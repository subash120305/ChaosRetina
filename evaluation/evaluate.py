"""
Evaluator for Multi-label Retinal Disease Classification

Handles:
- Model evaluation on test sets
- ROC curve generation and visualization
- Threshold optimization
- Per-class performance analysis
- Results export
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm

from .metrics import (
    compute_metrics, compute_multilabel_metrics, 
    find_optimal_thresholds, print_metrics_summary,
    get_confusion_matrices, DISEASE_NAMES
)


class Evaluator:
    """
    Comprehensive model evaluator for retinal disease classification
    
    Args:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        threshold: Classification threshold (or 'optimal' for auto)
        output_dir: Directory to save evaluation results
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        threshold: Union[float, str] = 0.5,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.output_dir = Path(output_dir) if output_dir else Path('evaluation_results')
        
        self.model.to(self.device)
        self.model.eval()
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    @torch.no_grad()
    def predict(
        self, 
        dataloader: DataLoader,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions for entire dataset
        
        Args:
            dataloader: Test data loader
            return_labels: Whether to return ground truth labels
            
        Returns:
            y_pred_proba: Predicted probabilities
            y_true: Ground truth labels (if return_labels=True)
        """
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
                if return_labels and len(batch) > 1:
                    labels = batch[1]
                    all_labels.append(labels.numpy())
            else:
                images = batch.to(self.device)
            
            logits = self.model(images)
            probs = torch.sigmoid(logits)
            all_predictions.append(probs.cpu().numpy())
        
        y_pred_proba = np.concatenate(all_predictions, axis=0)
        
        if return_labels and len(all_labels) > 0:
            y_true = np.concatenate(all_labels, axis=0)
            return y_pred_proba, y_true
        
        return y_pred_proba, None
    
    def evaluate(
        self,
        dataloader: DataLoader,
        save_results: bool = True,
        generate_plots: bool = True
    ) -> Dict[str, float]:
        """
        Full evaluation pipeline
        
        Args:
            dataloader: Test data loader
            save_results: Save metrics to JSON
            generate_plots: Generate and save plots
            
        Returns:
            Dictionary of all metrics
        """
        # Generate predictions
        y_pred_proba, y_true = self.predict(dataloader, return_labels=True)
        
        if y_true is None:
            raise ValueError("DataLoader must provide labels for evaluation")
        
        # Find optimal thresholds if requested
        if self.threshold == 'optimal':
            thresholds = find_optimal_thresholds(y_true, y_pred_proba)
            y_pred = np.zeros_like(y_pred_proba)
            for i in range(y_pred_proba.shape[1]):
                y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(int)
        else:
            thresholds = np.full(y_pred_proba.shape[1], self.threshold)
            y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_pred_proba, task='multilabel')
        
        # Add threshold info
        metrics['thresholds'] = thresholds.tolist() if isinstance(thresholds, np.ndarray) else thresholds
        
        # Print summary
        print_metrics_summary(metrics)
        
        # Save results
        if save_results:
            self._save_results(metrics, y_true, y_pred_proba, thresholds)
        
        # Generate plots
        if generate_plots:
            self._generate_roc_curves(y_true, y_pred_proba)
            self._generate_pr_curves(y_true, y_pred_proba)
            self._generate_confusion_matrices(y_true, y_pred)
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray
    ):
        """Save evaluation results to files"""
        
        # Save metrics JSON
        metrics_path = self.output_dir / 'metrics.json'
        
        # Convert numpy types for JSON serialization
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                metrics_serializable[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                metrics_serializable[k] = int(v)
            else:
                metrics_serializable[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"📁 Metrics saved to {metrics_path}")
        
        # Save predictions
        np.save(self.output_dir / 'y_true.npy', y_true)
        np.save(self.output_dir / 'y_pred_proba.npy', y_pred_proba)
        np.save(self.output_dir / 'thresholds.npy', thresholds)
        print(f"📁 Predictions saved to {self.output_dir}")
    
    def _generate_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        top_k: int = 10
    ):
        """Generate and save ROC curves"""
        
        # Compute AUC for each class to find top performers
        aucs = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append((i, roc_auc))
        
        # Sort by AUC and take top k
        aucs.sort(key=lambda x: x[1], reverse=True)
        top_classes = aucs[:top_k]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        for class_idx, class_auc in top_classes:
            fpr, tpr, _ = roc_curve(y_true[:, class_idx], y_pred_proba[:, class_idx])
            name = DISEASE_NAMES[class_idx] if class_idx < len(DISEASE_NAMES) else f'Class_{class_idx}'
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={class_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - Top {top_k} Classes by AUC', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=150)
        plt.close()
        print(f"📊 ROC curves saved to {self.output_dir / 'roc_curves.png'}")
    
    def _generate_pr_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        top_k: int = 10
    ):
        """Generate and save Precision-Recall curves"""
        
        # Compute AP for each class
        aps = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
                ap = auc(recall, precision)
                aps.append((i, ap))
        
        aps.sort(key=lambda x: x[1], reverse=True)
        top_classes = aps[:top_k]
        
        plt.figure(figsize=(12, 8))
        
        for class_idx, class_ap in top_classes:
            precision, recall, _ = precision_recall_curve(y_true[:, class_idx], y_pred_proba[:, class_idx])
            name = DISEASE_NAMES[class_idx] if class_idx < len(DISEASE_NAMES) else f'Class_{class_idx}'
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AP={class_ap:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - Top {top_k} Classes by AP', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=150)
        plt.close()
        print(f"📊 PR curves saved to {self.output_dir / 'pr_curves.png'}")
    
    def _generate_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 6
    ):
        """Generate and save confusion matrix plots"""
        
        cms = get_confusion_matrices(y_true, y_pred, DISEASE_NAMES)
        
        # Sort by support (most common classes)
        supports = []
        for i, name in enumerate(DISEASE_NAMES[:y_true.shape[1]]):
            supports.append((name, y_true[:, i].sum()))
        supports.sort(key=lambda x: x[1], reverse=True)
        
        top_names = [x[0] for x in supports[:top_k]]
        
        # Plot grid of confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, name in enumerate(top_names):
            if name in cms and idx < len(axes):
                cm = cms[name]
                ax = axes[idx]
                
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_title(f'{name}\n(N={cm.sum()})', fontsize=12)
                
                # Add text annotations
                thresh = cm.max() / 2
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f'{cm[i, j]}', 
                               ha='center', va='center',
                               color='white' if cm[i, j] > thresh else 'black')
                
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Neg', 'Pos'])
                ax.set_yticklabels(['Neg', 'Pos'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
        
        plt.suptitle('Confusion Matrices - Top Classes by Support', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150)
        plt.close()
        print(f"📊 Confusion matrices saved to {self.output_dir / 'confusion_matrices.png'}")
    
    def compare_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    ) -> Dict[float, Dict[str, float]]:
        """
        Compare metrics at different thresholds
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to compare
            
        Returns:
            Dictionary mapping threshold to metrics
        """
        results = {}
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            metrics = compute_metrics(y_true, y_pred, y_pred_proba, task='multilabel')
            
            results[thresh] = {
                'f1_macro': metrics.get('f1_macro', 0),
                'precision_macro': metrics.get('precision_macro', 0),
                'recall_macro': metrics.get('recall_macro', 0),
                'hamming_loss': metrics.get('hamming_loss', 0)
            }
        
        # Print comparison
        print("\n📊 Threshold Comparison:")
        print("-" * 60)
        print(f"{'Threshold':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 60)
        
        for thresh, m in results.items():
            print(f"{thresh:<12.2f} {m['f1_macro']:<12.4f} {m['precision_macro']:<12.4f} {m['recall_macro']:<12.4f}")
        
        return results
    
    def generate_report(
        self,
        dataloader: DataLoader,
        model_name: str = "model"
    ) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            dataloader: Test data loader
            model_name: Name of the model for report
            
        Returns:
            Path to generated report
        """
        # Run evaluation
        metrics = self.evaluate(dataloader, save_results=True, generate_plots=True)
        
        # Generate markdown report
        report_path = self.output_dir / f'{model_name}_evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Evaluation Report: {model_name}\n\n")
            f.write("## Summary Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            summary_keys = ['auc_macro', 'auc_micro', 'ap_macro', 'f1_macro',
                          'precision_macro', 'recall_macro', 'hamming_loss']
            
            for key in summary_keys:
                if key in metrics:
                    f.write(f"| {key} | {metrics[key]:.4f} |\n")
            
            f.write("\n## Per-Class AUC (Top 10)\n\n")
            f.write("| Disease | AUC |\n")
            f.write("|---------|-----|\n")
            
            auc_items = [(k, v) for k, v in metrics.items() 
                        if k.startswith('auc_') and k not in 
                        ['auc_macro', 'auc_micro', 'auc_weighted', 'auc_std']]
            auc_items.sort(key=lambda x: x[1], reverse=True)
            
            for key, val in auc_items[:10]:
                disease = key.replace('auc_', '')
                f.write(f"| {disease} | {val:.4f} |\n")
            
            f.write("\n## Visualizations\n\n")
            f.write("- ![ROC Curves](roc_curves.png)\n")
            f.write("- ![PR Curves](pr_curves.png)\n")
            f.write("- ![Confusion Matrices](confusion_matrices.png)\n")
        
        print(f"📄 Report saved to {report_path}")
        return str(report_path)


if __name__ == "__main__":
    print("Evaluator Test")
    print("=" * 50)
    print("Evaluator ready for use with trained models.")
