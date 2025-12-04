import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import argparse
from pathlib import Path
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_roc_curve(targets, probs, save_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc

def plot_confusion_matrix(targets, preds, save_path, classes=["Healthy", "Disease"]):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate Final Report and Graphs')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--results_dir', type=str, default='outputs/final_results', help='Directory with prediction CSVs')
    args = parser.parse_args()
    
    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Generating report in: {results_dir}")
    
    # ======================================================================================
    # 1. DETECTOR ANALYSIS (Binary)
    # ======================================================================================
    print("\n" + "="*50)
    print("BINARY DETECTOR ANALYSIS")
    print("="*50)
    
    det_df = pd.read_csv(results_dir / "detector_predictions.csv")
    targets = det_df["Disease_Risk_Target"].values
    probs = det_df["Disease_Risk_Prob"].values
    preds = (probs > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Plots
    plot_roc_curve(targets, probs, plots_dir / "detector_roc_curve.png", title="Detector ROC Curve (Disease vs Healthy)")
    plot_confusion_matrix(targets, preds, plots_dir / "detector_confusion_matrix.png")
    
    # Save metrics to text file
    with open(results_dir / "detector_metrics.txt", "w") as f:
        f.write("BINARY DETECTOR METRICS\n")
        f.write("=======================\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUROC:     {roc_auc_score(targets, probs):.4f}\n")

    # ======================================================================================
    # 2. CLASSIFIER ANALYSIS (Multi-label)
    # ======================================================================================
    print("\n" + "="*50)
    print("MULTI-LABEL CLASSIFIER ANALYSIS")
    print("="*50)
    
    # Load predictions
    cls_preds_df = pd.read_csv(results_dir / "classifier_predictions.csv")
    
    # Load Ground Truth (Need to reload dataset to get actual labels)
    test_csv = Path(config['paths']['test_labels'])
    disease_columns = config['dataset']['disease_columns']
    
    gt_df = pd.read_csv(test_csv)
    
    # Ensure ID alignment
    # Assuming the CSVs are in same order (which they are from run_final_eval)
    # But let's be safe and merge on ID
    merged_df = pd.merge(gt_df, cls_preds_df, on=config['dataset']['image_id_column'], suffixes=('_true', '_pred'))
    
    # Calculate per-class metrics
    class_metrics = []
    
    for disease in disease_columns:
        if f"{disease}_true" not in merged_df.columns:
            continue
            
        y_true = merged_df[f"{disease}_true"].values
        y_score = merged_df[f"{disease}_pred"].values
        y_pred = (y_score > 0.5).astype(int)
        
        # Skip if no positive samples in test set for this disease
        if np.sum(y_true) == 0:
            continue
            
        try:
            auc_score = roc_auc_score(y_true, y_score)
        except:
            auc_score = 0.5
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        class_metrics.append({
            "Disease": disease,
            "AUROC": auc_score,
            "F1": f1,
            "Support": np.sum(y_true)
        })
    
    metrics_df = pd.DataFrame(class_metrics)
    print(metrics_df.to_string(index=False))
    
    # Save detailed report
    metrics_df.to_csv(results_dir / "classifier_class_metrics.csv", index=False)
    
    # Plot Class-wise AUROC
    plt.figure(figsize=(12, 8))
    sns.barplot(data=metrics_df, x="AUROC", y="Disease", palette="viridis")
    plt.title("Per-Class AUROC Performance")
    plt.xlabel("AUROC Score")
    plt.axvline(0.5, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(plots_dir / "classifier_per_class_auroc.png", dpi=300)
    plt.close()
    
    print(f"\n✅ Report generated! Check {results_dir} and {plots_dir}")

if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score # Import here to avoid circular dependency issues if any
    main()
