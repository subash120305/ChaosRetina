"""
Evaluation Metrics for Multi-label Retinal Disease Classification

Comprehensive metrics including:
- Per-class and macro/micro AUC-ROC
- Precision, Recall, F1 at various thresholds
- Multi-label specific metrics (Hamming loss, subset accuracy)
- Confusion matrices and calibration curves
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    accuracy_score, hamming_loss,
    confusion_matrix, classification_report,
    multilabel_confusion_matrix
)
import warnings


# Disease names for RIADD (28 classes)
DISEASE_NAMES = [
    'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
    'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE',
    'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP'
]


def compute_auc_scores(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    disease_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute per-class and aggregate AUC-ROC scores
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred_proba: Predicted probabilities (N, num_classes)
        disease_names: Names of diseases for reporting
        
    Returns:
        Dictionary with AUC scores
    """
    if disease_names is None:
        disease_names = DISEASE_NAMES
    
    results = {}
    num_classes = y_true.shape[1]
    
    per_class_auc = []
    valid_classes = []
    
    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                per_class_auc.append(auc)
                valid_classes.append(i)
                if i < len(disease_names):
                    results[f'auc_{disease_names[i]}'] = auc
            except ValueError:
                pass
    
    # Aggregate metrics
    if len(per_class_auc) > 0:
        results['auc_macro'] = np.mean(per_class_auc)
        results['auc_std'] = np.std(per_class_auc)
    
    # Micro AUC (flattened)
    try:
        results['auc_micro'] = roc_auc_score(
            y_true[:, valid_classes].ravel(),
            y_pred_proba[:, valid_classes].ravel()
        )
    except ValueError:
        results['auc_micro'] = 0.0
    
    # Weighted AUC
    try:
        results['auc_weighted'] = roc_auc_score(
            y_true, y_pred_proba, average='weighted', multi_class='ovr'
        )
    except ValueError:
        results['auc_weighted'] = results.get('auc_macro', 0.0)
    
    return results


def compute_ap_scores(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    disease_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute Average Precision (AP) scores
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        disease_names: Names of diseases
        
    Returns:
        Dictionary with AP scores
    """
    if disease_names is None:
        disease_names = DISEASE_NAMES
    
    results = {}
    num_classes = y_true.shape[1]
    
    per_class_ap = []
    
    for i in range(num_classes):
        if y_true[:, i].sum() > 0:
            try:
                ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                per_class_ap.append(ap)
                if i < len(disease_names):
                    results[f'ap_{disease_names[i]}'] = ap
            except ValueError:
                pass
    
    if len(per_class_ap) > 0:
        results['ap_macro'] = np.mean(per_class_ap)
        results['ap_std'] = np.std(per_class_ap)
    
    # Micro AP
    try:
        results['ap_micro'] = average_precision_score(
            y_true.ravel(), y_pred_proba.ravel()
        )
    except ValueError:
        results['ap_micro'] = 0.0
    
    return results


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f1'
) -> np.ndarray:
    """
    Find optimal classification threshold per class
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Optimal threshold per class
    """
    num_classes = y_true.shape[1]
    thresholds = np.zeros(num_classes)
    
    for i in range(num_classes):
        best_score = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba[:, i] >= thresh).astype(int)
            
            if y_pred.sum() == 0 or y_true[:, i].sum() == 0:
                continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true[:, i], y_pred, average='binary', zero_division=0
                )
            
            if metric == 'f1':
                score = f1
            elif metric == 'precision':
                score = prec
            elif metric == 'recall':
                score = rec
            else:
                score = f1
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        thresholds[i] = best_thresh
    
    return thresholds


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    disease_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics
    
    Args:
        y_true: Ground truth binary labels (N, num_classes)
        y_pred: Predicted binary labels (N, num_classes)
        y_pred_proba: Optional predicted probabilities
        disease_names: Names of diseases
        
    Returns:
        Dictionary with all metrics
    """
    if disease_names is None:
        disease_names = DISEASE_NAMES
    
    results = {}
    
    # Multi-label specific metrics
    results['hamming_loss'] = hamming_loss(y_true, y_pred)
    results['subset_accuracy'] = accuracy_score(y_true, y_pred)  # Exact match
    
    # Per-class metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
    
    for i, name in enumerate(disease_names[:y_true.shape[1]]):
        results[f'precision_{name}'] = precision[i]
        results[f'recall_{name}'] = recall[i]
        results[f'f1_{name}'] = f1[i]
        results[f'support_{name}'] = support[i]
    
    # Aggregate metrics
    results['precision_macro'] = np.mean(precision)
    results['recall_macro'] = np.mean(recall)
    results['f1_macro'] = np.mean(f1)
    
    # Micro metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true.ravel(), y_pred.ravel(), average='binary', zero_division=0
        )
    
    results['precision_micro'] = prec_micro
    results['recall_micro'] = rec_micro
    results['f1_micro'] = f1_micro
    
    # AUC scores if probabilities provided
    if y_pred_proba is not None:
        auc_scores = compute_auc_scores(y_true, y_pred_proba, disease_names)
        results.update(auc_scores)
        
        ap_scores = compute_ap_scores(y_true, y_pred_proba, disease_names)
        results.update(ap_scores)
    
    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    task: str = 'multilabel'
) -> Dict[str, float]:
    """
    Main metric computation function
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (or probabilities if threshold needed)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        task: 'multilabel' or 'binary'
        
    Returns:
        Dictionary with metrics
    """
    # Convert probabilities to binary if needed
    if y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        if y_pred_proba is None:
            y_pred_proba = y_pred.copy()
        y_pred = (y_pred >= threshold).astype(int)
    
    if task == 'binary':
        # Binary classification (Disease_Risk)
        results = {}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
        
        results['precision'] = prec
        results['recall'] = rec
        results['f1'] = f1
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if y_pred_proba is not None:
            try:
                results['auc'] = roc_auc_score(y_true, y_pred_proba)
                results['ap'] = average_precision_score(y_true, y_pred_proba)
            except ValueError:
                results['auc'] = 0.0
                results['ap'] = 0.0
        
        return results
    
    else:
        # Multi-label classification
        return compute_multilabel_metrics(y_true, y_pred, y_pred_proba)


def get_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disease_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Get confusion matrices for each class
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        disease_names: Names of diseases
        
    Returns:
        Dictionary mapping disease name to confusion matrix
    """
    if disease_names is None:
        disease_names = DISEASE_NAMES
    
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    results = {}
    for i, name in enumerate(disease_names[:len(mcm)]):
        results[name] = mcm[i]
    
    return results


def compute_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and other calibration metrics
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    # Flatten for overall calibration
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred_proba.ravel()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (y_pred_flat > bin_lower) & (y_pred_flat <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_pred_flat[in_bin].mean()
            avg_accuracy = y_true_flat[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return {
        'ece': ece,
        'mce': 0.0  # Max calibration error (simplified)
    }


def print_metrics_summary(metrics: Dict[str, float], top_k: int = 10):
    """
    Print formatted summary of metrics
    
    Args:
        metrics: Dictionary of metrics
        top_k: Number of per-class metrics to show
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # Aggregate metrics
    print("\n📊 Aggregate Metrics:")
    print("-" * 40)
    
    agg_keys = ['auc_macro', 'auc_micro', 'ap_macro', 'f1_macro', 
                'precision_macro', 'recall_macro', 'hamming_loss', 'subset_accuracy']
    
    for key in agg_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    # Per-class AUC (top classes)
    print(f"\n📈 Per-class AUC (top {top_k}):")
    print("-" * 40)
    
    auc_items = [(k, v) for k, v in metrics.items() if k.startswith('auc_') 
                 and k not in ['auc_macro', 'auc_micro', 'auc_weighted', 'auc_std']]
    auc_items.sort(key=lambda x: x[1], reverse=True)
    
    for key, val in auc_items[:top_k]:
        disease = key.replace('auc_', '')
        print(f"  {disease}: {val:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test metrics computation
    np.random.seed(42)
    
    n_samples = 100
    n_classes = 28
    
    # Simulated data
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("Testing Metrics Computation")
    print("=" * 50)
    
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    print(f"\nComputed {len(metrics)} metrics")
    
    print_metrics_summary(metrics)
