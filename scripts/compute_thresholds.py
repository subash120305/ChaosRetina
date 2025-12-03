"""
Compute Optimal Thresholds After Training

This script:
1. Loads a trained model
2. Runs predictions on validation set
3. Finds per-class optimal thresholds
4. Saves thresholds for use during inference

Run this after training to prepare for inference:
    python scripts/compute_thresholds.py --model outputs/classifier_xxx/fold_0/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import RFMiDDataset, get_dataloaders
from models.classifier import MultiLabelClassifier
from models.chaosfex import HybridCNNChaosFEX
from inference.threshold_optimizer import ThresholdOptimizer, DISEASE_NAMES


def load_model(model_path: str, device: torch.device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}
    
    architecture = config.get('architecture', 'efficientnet_b0')
    use_chaosfex = config.get('use_chaosfex', False)
    
    if use_chaosfex:
        model = HybridCNNChaosFEX(
            backbone_name=architecture,
            num_classes=28,
            n_chaos_neurons=config.get('chaos_neurons', 100)
        )
    else:
        model = MultiLabelClassifier(
            backbone_name=architecture,
            num_classes=28
        )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get predictions and ground truth from dataloader"""
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Getting predictions"):
        images, labels = batch
        images = images.to(device)
        
        logits = model(images)
        probs = torch.sigmoid(logits)
        
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='Compute Optimal Thresholds')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--method', type=str, default='f1',
                       choices=['f1', 'youden', 'precision', 'recall'],
                       help='Threshold optimization method')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for thresholds (default: same dir as model)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Validation fold to use')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Computing Optimal Per-Class Thresholds")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"\n📁 Loading model: {args.model}")
    model = load_model(args.model, device)
    
    # Get validation data
    data_config = config.get('paths', {})
    training_config = config.get('training', {})
    
    _, val_loader = get_dataloaders(
        csv_path=data_config.get('train_labels', 'data/RFMiD_Training_Labels.csv'),
        images_dir=data_config.get('train_images', 'data/Training'),
        batch_size=training_config.get('batch_size', 16),
        num_workers=4,
        fold=args.fold,
        num_folds=training_config.get('num_folds', 5),
        task='multilabel'
    )
    
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    # Get predictions
    y_true, y_pred = get_predictions(model, val_loader, device)
    
    # Find optimal thresholds
    print(f"\n🔍 Finding optimal thresholds (method: {args.method})")
    optimizer = ThresholdOptimizer(method=args.method)
    thresholds = optimizer.find_optimal_thresholds(y_true, y_pred)
    
    # Print summary
    optimizer.print_summary()
    
    # Save thresholds
    if args.output:
        output_path = args.output
    else:
        model_dir = Path(args.model).parent
        output_path = model_dir / 'optimal_thresholds.json'
    
    optimizer.save_thresholds(str(output_path))
    
    print(f"\n✅ Thresholds saved to {output_path}")
    print("\nUse with inference:")
    print(f"  python inference/predict.py --image IMG.jpg --model {args.model} --thresholds {output_path}")


if __name__ == '__main__':
    main()
