import torch
import pandas as pd
import numpy as np
import yaml
import argparse
import random
from pathlib import Path
from torch.utils.data import DataLoader

# Import project modules
import sys
import os
sys.path.append(os.getcwd())

from data.dataset import RFMiDDataset, get_val_transforms
from models.classifier import MultiLabelClassifier
from models.detector import DiseaseDetector

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_disease_names(labels, disease_columns):
    """Convert binary labels to list of disease names."""
    names = []
    for i, label in enumerate(labels):
        if label == 1:
            names.append(disease_columns[i])
    if not names:
        return ["Healthy"]
    return names

def main():
    parser = argparse.ArgumentParser(description='Visualize Random Predictions')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to trained classifier .pth')
    parser.add_argument('--detector_path', type=str, required=True, help='Path to trained detector .pth')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to check')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    disease_columns = config['dataset']['disease_columns']
    num_classes = len(disease_columns)
    
    # ---------------------------------------------------------
    # 1. Load Models
    # ---------------------------------------------------------
    print("Loading models...")
    
    # Classifier
    classifier = MultiLabelClassifier(backbone_name="densenet121", num_classes=num_classes, pretrained=False)
    ckpt_cls = torch.load(args.classifier_path, map_location=device)
    classifier.load_state_dict(ckpt_cls['model_state_dict'] if 'model_state_dict' in ckpt_cls else ckpt_cls)
    classifier.to(device)
    classifier.eval()
    
    # Detector
    detector = DiseaseDetector(backbone_name="efficientnet_b0", pretrained=False)
    ckpt_det = torch.load(args.detector_path, map_location=device)
    detector.load_state_dict(ckpt_det['model_state_dict'] if 'model_state_dict' in ckpt_det else ckpt_det)
    detector.to(device)
    detector.eval()
    
    # ---------------------------------------------------------
    # 2. Load Test Data
    # ---------------------------------------------------------
    print("Loading test data...")
    test_csv = Path(config['paths']['test_labels'])
    test_img_dir = Path(config['paths']['test_images'])
    test_transform = get_val_transforms(config['dataset']['image_size'])
    
    dataset = RFMiDDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        mode="multilabel",
        transform=test_transform,
        disease_columns=disease_columns
    )
    
    # ---------------------------------------------------------
    # 3. Random Sampling & Prediction
    # ---------------------------------------------------------
    indices = random.sample(range(len(dataset)), args.num_samples)
    
    print("\n" + "="*100)
    print(f"{'IMAGE ID':<10} | {'ACTUAL (Ground Truth)':<35} | {'PREDICTED (Model)':<35} | {'RISK SCORE':<10}")
    print("="*100)
    
    with torch.no_grad():
        for idx in indices:
            # Get data
            image, labels = dataset[idx]
            image_id = dataset.df.iloc[idx][config['dataset']['image_id_column']]
            
            # Prepare input
            input_tensor = image.unsqueeze(0).to(device)
            
            # Run Inference
            # 1. Detector
            risk_logits = detector(input_tensor)
            risk_prob = torch.softmax(risk_logits, dim=1)[0, 1].item()
            
            # 2. Classifier
            cls_logits = classifier(input_tensor)
            cls_probs = torch.sigmoid(cls_logits)[0].cpu().numpy()
            
            # Process Results
            actual_diseases = get_disease_names(labels.numpy(), disease_columns)
            
            # Threshold predictions (e.g., > 0.5)
            pred_indices = np.where(cls_probs > 0.5)[0]
            pred_diseases = [disease_columns[i] for i in pred_indices]
            
            if not pred_diseases:
                if risk_prob > 0.5:
                    pred_diseases = ["Uncertain Disease"] # Detector says sick, classifier unsure
                else:
                    pred_diseases = ["Healthy"]
            
            # Format output
            actual_str = ", ".join(actual_diseases)
            pred_str = ", ".join(pred_diseases)
            risk_str = f"{risk_prob:.4f}"
            
            # Color/Mark match (simple text based)
            match_mark = "✅" if set(actual_diseases) == set(pred_diseases) else "⚠️"
            if actual_diseases == ["Healthy"] and pred_diseases == ["Healthy"]: match_mark = "✅"
            
            print(f"{image_id:<10} | {actual_str:<35} | {pred_str:<35} | {risk_str:<10} {match_mark}")

    print("="*100)
    print("Note: 'Risk Score' is from the Binary Detector (0=Healthy, 1=Disease).")
    print("      'Predicted' is from the Multi-label Classifier.")

if __name__ == '__main__':
    main()
