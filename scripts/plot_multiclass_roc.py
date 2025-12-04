import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from pathlib import Path
import yaml
import sys
import os
import itertools

# Add project root to path
sys.path.append(os.getcwd())

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Generate Multi-Class ROC Curve')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--results_dir', type=str, default='outputs/final_results', help='Directory with prediction CSVs')
    args = parser.parse_args()
    
    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Generating Multi-Class ROC in: {plots_dir}")
    
    # Load predictions and ground truth
    cls_preds_df = pd.read_csv(results_dir / "classifier_predictions.csv")
    test_csv = Path(config['paths']['test_labels'])
    gt_df = pd.read_csv(test_csv)
    
    # Merge
    merged_df = pd.merge(gt_df, cls_preds_df, on=config['dataset']['image_id_column'], suffixes=('_true', '_pred'))
    
    disease_columns = config['dataset']['disease_columns']
    
    # Setup Grid Plot (27 diseases -> 5 rows x 6 columns = 30 slots)
    n_cols = 6
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    print("Calculating ROC for each disease...")
    
    for i, disease in enumerate(disease_columns):
        ax = axes[i]
        
        if f"{disease}_true" not in merged_df.columns:
            ax.axis('off')
            continue
            
        y_true = merged_df[f"{disease}_true"].values
        y_score = merged_df[f"{disease}_pred"].values
        
        # Skip if no positive samples
        if np.sum(y_true) == 0:
            ax.text(0.5, 0.5, "No Samples", ha='center', va='center')
            ax.set_title(f"{disease}")
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot on the specific subplot
        ax.plot(fpr, tpr, color='darkorange', lw=2)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f"{disease} (AUC={roc_auc:.2f})", fontsize=10, fontweight='bold')
        
        # Only show labels on outer edges to save space
        if i % n_cols == 0:
            ax.set_ylabel('TPR', fontsize=8)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('FPR', fontsize=8)
            
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(disease_columns), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    save_path = plots_dir / "multiclass_roc_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {save_path}")

if __name__ == '__main__':
    main()
