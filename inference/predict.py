"""
Inference Pipeline for Retinal Disease Prediction

Complete inference script that:
1. Loads trained model(s)
2. Preprocesses input image
3. Applies TTA (Test-Time Augmentation)
4. Uses per-class optimal thresholds
5. Returns predictions with confidence levels

Usage:
    python inference/predict.py --image path/to/image.jpg --model path/to/model.pt
    python inference/predict.py --image path/to/image.jpg --ensemble outputs/ensemble_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier import MultiLabelClassifier
from models.chaosfex import HybridCNNChaosFEX
from models.ensemble import EnsemblePredictor
from data.preprocessing import RetinalPreprocessor
from inference.tta import TTAWrapper, LightTTA, create_tta_wrapper
from inference.threshold_optimizer import ThresholdOptimizer, DISEASE_NAMES


class RetinalDiseasePredictor:
    """
    Complete inference pipeline for retinal disease prediction
    
    Features:
    - Single model or ensemble prediction
    - Test-Time Augmentation (TTA)
    - Per-class optimal thresholds
    - Confidence levels and interpretability
    
    Args:
        model_path: Path to trained model (.pt file)
        threshold_path: Path to optimal thresholds (.json file)
        config_path: Path to config file
        use_tta: Enable Test-Time Augmentation
        tta_mode: 'full' (8 transforms) or 'light' (4 transforms)
        device: Compute device ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ensemble_config: Optional[str] = None,
        threshold_path: Optional[str] = None,
        config_path: str = 'config/config.yaml',
        use_tta: bool = True,
        tta_mode: str = 'light',
        device: Optional[str] = None
    ):
        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Load model(s)
        if ensemble_config:
            self.model = self._load_ensemble(ensemble_config)
            self.is_ensemble = True
        elif model_path:
            self.model = self._load_single_model(model_path)
            self.is_ensemble = False
        else:
            raise ValueError("Must provide either model_path or ensemble_config")
        
        # Wrap with TTA
        self.use_tta = use_tta
        if use_tta:
            print(f"✅ TTA enabled ({tta_mode} mode)")
            self.tta_model = create_tta_wrapper(self.model, mode=tta_mode)
        else:
            self.tta_model = None
        
        # Load thresholds
        if threshold_path and os.path.exists(threshold_path):
            self.threshold_optimizer = ThresholdOptimizer()
            self.thresholds = self.threshold_optimizer.load_thresholds(threshold_path)
            print(f"✅ Loaded per-class thresholds from {threshold_path}")
        else:
            # Default thresholds
            self.thresholds = np.full(28, 0.5)
            print("⚠️  Using default threshold (0.5) for all classes")
        
        # Preprocessor
        self.preprocessor = RetinalPreprocessor(target_size=224)
        
        # Normalization transform
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Disease names
        self.disease_names = DISEASE_NAMES
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_single_model(self, model_path: str) -> nn.Module:
        """Load a single trained model"""
        print(f"📁 Loading model from {model_path}")
        
        # Determine model type from path or config
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check if it's a state dict or full checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            model_config = {}
        
        # Create model
        architecture = model_config.get('architecture', 'efficientnet_b0')
        use_chaosfex = model_config.get('use_chaosfex', False)
        
        if use_chaosfex:
            model = HybridCNNChaosFEX(
                backbone_name=architecture,
                num_classes=28,
                n_chaos_neurons=model_config.get('chaos_neurons', 100)
            )
        else:
            model = MultiLabelClassifier(
                backbone_name=architecture,
                num_classes=28
            )
        
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_ensemble(self, ensemble_config_path: str) -> nn.Module:
        """Load ensemble of models"""
        print(f"📁 Loading ensemble from {ensemble_config_path}")
        
        with open(ensemble_config_path, 'r') as f:
            ensemble_config = yaml.safe_load(f)
        
        models = []
        weights = ensemble_config.get('weights', [])
        
        for i, model_info in enumerate(ensemble_config['models']):
            model_path = model_info['path']
            if os.path.exists(model_path):
                model = self._load_single_model(model_path)
                models.append(model)
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        if not models:
            raise ValueError("No models loaded for ensemble")
        
        # Create ensemble
        ensemble = EnsemblePredictor(
            models=models,
            weights=weights[:len(models)] if weights else None,
            method='weighted_average'
        )
        
        print(f"✅ Loaded ensemble with {len(models)} models")
        return ensemble
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image
        
        Returns:
            preprocessed: Preprocessed image for model (normalized tensor ready)
            original: Original image for display
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        
        # Apply retinal preprocessing (crop)
        preprocessed = self.preprocessor.preprocess(image)
        
        # Resize to model input size
        preprocessed = cv2.resize(preprocessed, (224, 224))
        
        return preprocessed, original
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        return_all: bool = False
    ) -> Dict:
        """
        Predict diseases from a retinal image
        
        Args:
            image_path: Path to retinal fundus image
            return_all: If True, return all class probabilities
            
        Returns:
            Dictionary with predictions, confidences, and interpretation
        """
        # Preprocess
        image, original = self.preprocess_image(image_path)
        
        # Get predictions
        if self.use_tta and self.tta_model is not None:
            # TTA prediction
            probs = self.tta_model(image)
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
        else:
            # Standard prediction
            x = self.normalize(image=image)['image']
            x = x.unsqueeze(0).to(self.device)
            
            if self.is_ensemble:
                probs = self.model.predict_proba(x)
            else:
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        
        # Apply per-class thresholds
        predictions = (probs >= self.thresholds).astype(int)
        
        # Build result
        result = {
            'image_path': image_path,
            'disease_risk': {
                'detected': bool(predictions[0]),
                'probability': float(probs[0]),
                'threshold': float(self.thresholds[0])
            },
            'detected_diseases': [],
            'all_probabilities': {},
            'summary': ''
        }
        
        # Process each disease
        high_confidence = []
        moderate_confidence = []
        low_confidence = []
        
        for i, (name, prob, pred, thresh) in enumerate(zip(
            self.disease_names, probs, predictions, self.thresholds
        )):
            result['all_probabilities'][name] = float(prob)
            
            if pred == 1:
                disease_info = {
                    'name': name,
                    'probability': float(prob),
                    'threshold': float(thresh),
                    'confidence': self._get_confidence_level(prob, thresh)
                }
                result['detected_diseases'].append(disease_info)
                
                # Categorize by confidence
                if prob >= 0.8:
                    high_confidence.append(name)
                elif prob >= 0.6:
                    moderate_confidence.append(name)
                else:
                    low_confidence.append(name)
        
        # Generate summary
        result['summary'] = self._generate_summary(
            high_confidence, moderate_confidence, low_confidence, probs[0]
        )
        
        # Sort detected diseases by probability
        result['detected_diseases'].sort(key=lambda x: x['probability'], reverse=True)
        
        return result
    
    def _get_confidence_level(self, prob: float, threshold: float) -> str:
        """Categorize prediction confidence"""
        margin = prob - threshold
        
        if prob >= 0.85:
            return 'HIGH'
        elif prob >= 0.7:
            return 'MODERATE-HIGH'
        elif prob >= 0.55:
            return 'MODERATE'
        elif margin > 0:
            return 'LOW'
        else:
            return 'BELOW_THRESHOLD'
    
    def _generate_summary(
        self,
        high: List[str],
        moderate: List[str],
        low: List[str],
        disease_risk_prob: float
    ) -> str:
        """Generate human-readable summary"""
        lines = []
        
        # Overall assessment
        if disease_risk_prob >= 0.8:
            lines.append("⚠️ HIGH PROBABILITY OF RETINAL DISEASE DETECTED")
        elif disease_risk_prob >= 0.5:
            lines.append("⚡ MODERATE PROBABILITY OF RETINAL DISEASE")
        else:
            lines.append("✅ LOW PROBABILITY OF RETINAL DISEASE")
        
        lines.append("")
        
        # High confidence detections
        if high:
            lines.append(f"🔴 High confidence detections: {', '.join(high)}")
        
        # Moderate confidence
        if moderate:
            lines.append(f"🟡 Moderate confidence: {', '.join(moderate)}")
        
        # Low confidence
        if low:
            lines.append(f"🟢 Low confidence (needs review): {', '.join(low)}")
        
        if not high and not moderate and not low:
            lines.append("No specific diseases detected above threshold.")
        
        lines.append("")
        lines.append("⚕️ DISCLAIMER: This is an AI prediction. Please consult an ophthalmologist for diagnosis.")
        
        return "\n".join(lines)
    
    def predict_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """Predict on multiple images"""
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(image_paths) if show_progress else image_paths
        
        for path in iterator:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        
        return results
    
    def print_prediction(self, result: Dict):
        """Pretty print prediction result"""
        print("\n" + "=" * 70)
        print(f"📷 Image: {result['image_path']}")
        print("=" * 70)
        
        # Disease Risk
        dr = result['disease_risk']
        status = "🔴 DETECTED" if dr['detected'] else "🟢 NOT DETECTED"
        print(f"\n🏥 Disease Risk: {status} (probability: {dr['probability']:.2%})")
        
        # Detected diseases
        if result['detected_diseases']:
            print(f"\n📋 Detected Conditions ({len(result['detected_diseases'])}):")
            print("-" * 50)
            for disease in result['detected_diseases']:
                conf_emoji = {'HIGH': '🔴', 'MODERATE-HIGH': '🟠', 
                             'MODERATE': '🟡', 'LOW': '🟢'}.get(disease['confidence'], '⚪')
                print(f"  {conf_emoji} {disease['name']}: {disease['probability']:.2%} "
                      f"[{disease['confidence']}]")
        else:
            print("\n✅ No diseases detected above threshold")
        
        # Summary
        print("\n" + "-" * 70)
        print(result['summary'])
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Retinal Disease Prediction')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (.pt)')
    parser.add_argument('--ensemble', type=str, default=None,
                       help='Path to ensemble config (.yaml)')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path to optimal thresholds (.json)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable Test-Time Augmentation')
    parser.add_argument('--tta-mode', type=str, default='light',
                       choices=['full', 'light'],
                       help='TTA mode')
    parser.add_argument('--output', type=str, default=None,
                       help='Save result to JSON file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not args.model and not args.ensemble:
        parser.error("Must provide either --model or --ensemble")
    
    # Create predictor
    predictor = RetinalDiseasePredictor(
        model_path=args.model,
        ensemble_config=args.ensemble,
        threshold_path=args.thresholds,
        config_path=args.config,
        use_tta=not args.no_tta,
        tta_mode=args.tta_mode,
        device=args.device
    )
    
    # Predict
    result = predictor.predict(args.image)
    
    # Print result
    predictor.print_prediction(result)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Result saved to {args.output}")


if __name__ == '__main__':
    main()
