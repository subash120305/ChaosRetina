import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

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

def evaluate_model(model, dataloader, device, is_binary=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            if is_binary:
                # For binary, we want probability of class 1 (Disease)
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                # For multi-label, sigmoid
                probs = torch.sigmoid(outputs)
                
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    return np.concatenate(all_preds), np.concatenate(all_targets)

def main():
    parser = argparse.ArgumentParser(description='Final Evaluation')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to trained classifier .pth')
    parser.add_argument('--detector_path', type=str, required=True, help='Path to trained detector .pth')
    parser.add_argument('--output_dir', type=str, default='outputs/final_results', help='Directory to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("\n[1/4] Loading Test Data...")
    test_csv = Path(config['paths']['test_labels'])
    test_img_dir = Path(config['paths']['test_images'])
    
    # Get disease columns
    disease_columns = config['dataset']['disease_columns']
    num_classes = len(disease_columns)
    print(f"Number of classes: {num_classes}")
    
    # Transforms
    test_transform = get_val_transforms(config['dataset']['image_size'])
    
    # Test Dataset
    test_dataset = RFMiDDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        mode="multilabel", # We use multilabel mode to get all labels
        transform=test_transform,
        disease_columns=disease_columns
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # ---------------------------------------------------------
    # 2. Evaluate Classifier (Multi-label)
    # ---------------------------------------------------------
    print("\n[2/4] Evaluating Classifier...")
    
    # Initialize model structure (DenseNet121 as per your training)
    # Note: We need to match the architecture used in training
    # You used densenet121 for classifier
    classifier = MultiLabelClassifier(
        backbone_name="densenet121", 
        num_classes=num_classes,
        pretrained=False # Loading weights manually
    )
    
    # Load weights
    print(f"Loading classifier weights from: {args.classifier_path}")
    checkpoint = torch.load(args.classifier_path, map_location=device)
    
    # Handle state_dict loading (remove 'module.' prefix if present, handle full checkpoint dict)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    
    # Run inference
    cls_preds, cls_targets = evaluate_model(classifier, test_loader, device, is_binary=False)
    
    # Calculate Metrics
    try:
        cls_auroc = roc_auc_score(cls_targets, cls_preds, average='macro')
        print(f"Classifier Test AUROC: {cls_auroc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUROC: {e}")

    # ---------------------------------------------------------
    # 3. Evaluate Detector (Binary)
    # ---------------------------------------------------------
    print("\n[3/4] Evaluating Detector...")
    
    # Initialize model structure (EfficientNet_B0 as per your training)
    detector = DiseaseDetector(
        backbone_name="efficientnet_b0",
        pretrained=False
    )
    
    # Load weights
    print(f"Loading detector weights from: {args.detector_path}")
    checkpoint = torch.load(args.detector_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    detector.load_state_dict(state_dict)
    detector.to(device)
    
    # For detector, we need binary targets (Disease_Risk)
    # The dataset is in multilabel mode, so we extract Disease_Risk column if available
    # Or derive it (if any label is 1, then risk is 1)
    # Let's check if Disease_Risk is in the dataset dataframe
    if 'Disease_Risk' in test_dataset.df.columns:
        det_targets = test_dataset.df['Disease_Risk'].values
    else:
        # Derive from labels (assuming labels are 0/1)
        det_targets = np.max(cls_targets, axis=1)
        
    # Run inference
    det_preds, _ = evaluate_model(detector, test_loader, device, is_binary=True)
    
    # Calculate Metrics
    try:
        det_auroc = roc_auc_score(det_targets, det_preds)
        print(f"Detector Test AUROC: {det_auroc:.4f}")
    except Exception as e:
        print(f"Could not calculate Detector AUROC: {e}")
        
    # ---------------------------------------------------------
    # 4. Save Results
    # ---------------------------------------------------------
    print("\n[4/4] Saving Results...")
    
    # Save Classifier Predictions
    cls_df = pd.DataFrame(cls_preds, columns=disease_columns)
    cls_df.insert(0, "ID", test_dataset.df[config['dataset']['image_id_column']])
    cls_df.to_csv(output_dir / "classifier_predictions.csv", index=False)
    
    # Save Detector Predictions
    det_df = pd.DataFrame({
        "ID": test_dataset.df[config['dataset']['image_id_column']],
        "Disease_Risk_Prob": det_preds,
        "Disease_Risk_Target": det_targets
    })
    det_df.to_csv(output_dir / "detector_predictions.csv", index=False)
    
    print(f"\n✅ Done! Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
