"""
Optimal Threshold Finder for Multi-label Classification

Finds the best classification threshold for each disease class
to maximize F1, precision, or other metrics.

Critical for imbalanced datasets where 0.5 threshold is suboptimal.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)
import warnings
import json
from pathlib import Path


# Disease names for RIADD (28 classes)
DISEASE_NAMES = [
    'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
    'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE',
    'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP'
]


class ThresholdOptimizer:
    """
    Finds optimal classification thresholds per class
    
    Methods:
    - F1 optimization: Maximizes F1 score (balance precision/recall)
    - Youden's J: Maximizes sensitivity + specificity - 1
    - Precision-based: Ensures minimum precision
    - Recall-based: Ensures minimum recall (for medical: better to over-detect)
    """
    
    def __init__(
        self,
        method: Literal['f1', 'youden', 'precision', 'recall'] = 'f1',
        min_precision: float = 0.5,
        min_recall: float = 0.5,
        search_resolution: int = 100
    ):
        self.method = method
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.search_resolution = search_resolution
        
        self.thresholds: Optional[np.ndarray] = None
        self.threshold_dict: Dict[str, float] = {}
        self.metrics_at_threshold: Dict[str, Dict] = {}
    
    def find_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        disease_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Find optimal threshold for each class
        
        Args:
            y_true: Ground truth labels (N, num_classes)
            y_pred_proba: Predicted probabilities (N, num_classes)
            disease_names: Names for reporting
            
        Returns:
            Optimal thresholds array (num_classes,)
        """
        if disease_names is None:
            disease_names = DISEASE_NAMES
        
        num_classes = y_true.shape[1]
        self.thresholds = np.zeros(num_classes)
        
        for i in range(num_classes):
            name = disease_names[i] if i < len(disease_names) else f'Class_{i}'
            
            # Skip if class has no positive samples
            if y_true[:, i].sum() == 0:
                self.thresholds[i] = 0.5
                self.threshold_dict[name] = 0.5
                continue
            
            # Find optimal threshold
            if self.method == 'f1':
                thresh, metrics = self._optimize_f1(y_true[:, i], y_pred_proba[:, i])
            elif self.method == 'youden':
                thresh, metrics = self._optimize_youden(y_true[:, i], y_pred_proba[:, i])
            elif self.method == 'precision':
                thresh, metrics = self._optimize_with_min_precision(
                    y_true[:, i], y_pred_proba[:, i]
                )
            elif self.method == 'recall':
                thresh, metrics = self._optimize_with_min_recall(
                    y_true[:, i], y_pred_proba[:, i]
                )
            else:
                thresh, metrics = self._optimize_f1(y_true[:, i], y_pred_proba[:, i])
            
            self.thresholds[i] = thresh
            self.threshold_dict[name] = thresh
            self.metrics_at_threshold[name] = metrics
        
        return self.thresholds
    
    def _optimize_f1(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[float, Dict]:
        """Optimize threshold for maximum F1 score"""
        best_thresh = 0.5
        best_f1 = 0
        best_metrics = {}
        
        for thresh in np.linspace(0.1, 0.9, self.search_resolution):
            y_binary = (y_pred >= thresh).astype(int)
            
            if y_binary.sum() == 0:
                continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1 = f1_score(y_true, y_binary, zero_division=0)
                prec = precision_score(y_true, y_binary, zero_division=0)
                rec = recall_score(y_true, y_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {'f1': f1, 'precision': prec, 'recall': rec}
        
        return best_thresh, best_metrics
    
    def _optimize_youden(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, Dict]:
        """Optimize using Youden's J statistic (sensitivity + specificity - 1)"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # Youden's J = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Compute metrics at this threshold
        y_binary = (y_pred >= best_thresh).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(y_true, y_binary, zero_division=0)
            prec = precision_score(y_true, y_binary, zero_division=0)
            rec = recall_score(y_true, y_binary, zero_division=0)
        
        return best_thresh, {
            'f1': f1, 'precision': prec, 'recall': rec,
            'youden_j': j_scores[best_idx]
        }
    
    def _optimize_with_min_precision(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, Dict]:
        """Find threshold that maximizes recall while maintaining min precision"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Find thresholds where precision >= min_precision
        valid_idx = np.where(precision[:-1] >= self.min_precision)[0]
        
        if len(valid_idx) == 0:
            # Fall back to F1 optimization
            return self._optimize_f1(y_true, y_pred)
        
        # Among valid, pick one with highest recall
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
        best_thresh = thresholds[best_idx]
        
        y_binary = (y_pred >= best_thresh).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(y_true, y_binary, zero_division=0)
            prec = precision_score(y_true, y_binary, zero_division=0)
            rec = recall_score(y_true, y_binary, zero_division=0)
        
        return best_thresh, {'f1': f1, 'precision': prec, 'recall': rec}
    
    def _optimize_with_min_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Find threshold that maximizes precision while maintaining min recall
        IMPORTANT for medical: Better to detect more (higher recall) even if some false positives
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Find thresholds where recall >= min_recall
        valid_idx = np.where(recall[:-1] >= self.min_recall)[0]
        
        if len(valid_idx) == 0:
            # Use lowest threshold to maximize recall
            return 0.3, {'f1': 0, 'precision': 0, 'recall': 0}
        
        # Among valid, pick one with highest precision
        best_idx = valid_idx[np.argmax(precision[valid_idx])]
        best_thresh = thresholds[best_idx]
        
        y_binary = (y_pred >= best_thresh).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(y_true, y_binary, zero_division=0)
            prec = precision_score(y_true, y_binary, zero_division=0)
            rec = recall_score(y_true, y_binary, zero_division=0)
        
        return best_thresh, {'f1': f1, 'precision': prec, 'recall': rec}
    
    def apply_thresholds(
        self,
        y_pred_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply per-class thresholds to probabilities
        
        Args:
            y_pred_proba: Predicted probabilities (N, num_classes)
            thresholds: Optional custom thresholds (uses stored if None)
            
        Returns:
            Binary predictions (N, num_classes)
        """
        if thresholds is None:
            if self.thresholds is None:
                raise ValueError("No thresholds found. Run find_optimal_thresholds first.")
            thresholds = self.thresholds
        
        y_pred = np.zeros_like(y_pred_proba)
        for i in range(y_pred_proba.shape[1]):
            y_pred[:, i] = (y_pred_proba[:, i] >= thresholds[i]).astype(int)
        
        return y_pred
    
    def save_thresholds(self, path: str):
        """Save thresholds to JSON file"""
        data = {
            'method': self.method,
            'thresholds': self.threshold_dict,
            'metrics': self.metrics_at_threshold
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Thresholds saved to {path}")
    
    def load_thresholds(self, path: str) -> np.ndarray:
        """Load thresholds from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.threshold_dict = data['thresholds']
        self.metrics_at_threshold = data.get('metrics', {})
        
        # Convert to array
        self.thresholds = np.array([
            self.threshold_dict.get(name, 0.5) 
            for name in DISEASE_NAMES
        ])
        
        return self.thresholds
    
    def print_summary(self):
        """Print threshold summary"""
        print("\n" + "=" * 60)
        print("OPTIMAL THRESHOLDS SUMMARY")
        print("=" * 60)
        print(f"Method: {self.method}")
        print("-" * 60)
        print(f"{'Disease':<15} {'Threshold':<12} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 60)
        
        for name, thresh in self.threshold_dict.items():
            metrics = self.metrics_at_threshold.get(name, {})
            f1 = metrics.get('f1', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            print(f"{name:<15} {thresh:<12.3f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f}")
        
        print("=" * 60)


def get_medical_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> np.ndarray:
    """
    Get thresholds optimized for medical use case
    
    For medical imaging, we prefer:
    - Higher recall (catch more diseases, even if some false positives)
    - Different strategies for common vs rare diseases
    """
    num_classes = y_true.shape[1]
    thresholds = np.zeros(num_classes)
    
    for i in range(num_classes):
        positive_rate = y_true[:, i].mean()
        
        if positive_rate > 0.1:
            # Common disease: use F1 optimization
            optimizer = ThresholdOptimizer(method='f1')
        elif positive_rate > 0.01:
            # Moderate: ensure minimum recall
            optimizer = ThresholdOptimizer(method='recall', min_recall=0.6)
        else:
            # Rare disease: lower threshold to catch more
            optimizer = ThresholdOptimizer(method='recall', min_recall=0.7)
        
        if y_true[:, i].sum() > 0:
            thresh, _ = optimizer._optimize_f1(y_true[:, i], y_pred_proba[:, i])
            thresholds[i] = thresh
        else:
            thresholds[i] = 0.5
    
    return thresholds


if __name__ == "__main__":
    print("Threshold Optimizer Test")
    print("=" * 50)
    
    # Simulated data
    np.random.seed(42)
    n_samples = 500
    n_classes = 28
    
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_pred_proba = np.random.rand(n_samples, n_classes)
    
    # Make predictions somewhat correlated with truth
    y_pred_proba = 0.3 * y_pred_proba + 0.7 * y_true + np.random.randn(n_samples, n_classes) * 0.1
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Test optimizer
    optimizer = ThresholdOptimizer(method='f1')
    thresholds = optimizer.find_optimal_thresholds(y_true, y_pred_proba)
    
    print(f"\nFound {len(thresholds)} thresholds")
    print(f"Range: [{thresholds.min():.3f}, {thresholds.max():.3f}]")
    print(f"Mean: {thresholds.mean():.3f}")
    
    optimizer.print_summary()
