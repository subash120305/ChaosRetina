# =============================================================================
# RIADD Modern - Ensemble Learning
# =============================================================================
"""
Ensemble learning for combining predictions from multiple models.

Ported from original riadd.aucmedi with improvements:
1. Cleaner API
2. Support for both averaging and stacking
3. Per-class meta-learners for better calibration
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple model predictions.
    
    Supports two methods:
    1. Averaging: Simple mean of predictions
    2. Stacking: Train meta-learner on predictions
    
    Example:
        # Collect predictions from multiple models
        predictions = {
            'efficientnet_fold0': preds1,  # [N, 28]
            'densenet_fold0': preds2,
            ...
        }
        
        # Train ensemble
        ensemble = EnsemblePredictor(method='stacking')
        ensemble.fit(predictions, ground_truth)
        
        # Predict
        final_preds = ensemble.predict(new_predictions)
    """
    
    def __init__(
        self,
        method: str = "stacking",
        meta_learner: str = "logistic_regression",
        balanced: bool = True,
        calibrate: bool = True
    ):
        """
        Args:
            method: 'averaging' or 'stacking'
            meta_learner: 'logistic_regression' or 'random_forest'
            balanced: Use class balancing for meta-learner
            calibrate: Apply probability calibration
        """
        self.method = method
        self.meta_learner_type = meta_learner
        self.balanced = balanced
        self.calibrate = calibrate
        
        # Meta-learners (one per class)
        self.meta_learners: Dict[int, Any] = {}
        
        # Feature columns (model names)
        self.feature_columns: List[str] = []
        self.num_classes: int = 0
        self.is_fitted: bool = False
    
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray
    ) -> "EnsemblePredictor":
        """
        Fit the ensemble.
        
        Args:
            predictions: Dict mapping model names to prediction arrays [N, C]
            ground_truth: Ground truth labels [N, C]
            
        Returns:
            Self
        """
        if self.method == "averaging":
            # No fitting needed for averaging
            self.feature_columns = list(predictions.keys())
            self.num_classes = ground_truth.shape[1]
            self.is_fitted = True
            return self
        
        # Stacking method
        self.feature_columns = sorted(predictions.keys())
        self.num_classes = ground_truth.shape[1]
        
        # Stack predictions as features
        # Shape: [N, num_models * num_classes] or [N, num_models] per class
        features = self._prepare_features(predictions)
        
        print(f"Training ensemble with {len(self.feature_columns)} models")
        print(f"Feature shape: {features.shape}")
        
        # Train one meta-learner per class
        for class_idx in range(self.num_classes):
            labels = ground_truth[:, class_idx]
            
            # Create meta-learner
            if self.meta_learner_type == "logistic_regression":
                meta_learner = LogisticRegression(
                    class_weight="balanced" if self.balanced else None,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42
                )
            else:
                meta_learner = RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced" if self.balanced else None,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Get features for this class
            class_features = self._get_class_features(features, class_idx)
            
            # Fit
            try:
                meta_learner.fit(class_features, labels)
                
                # Optional: Calibrate probabilities
                if self.calibrate:
                    meta_learner = CalibratedClassifierCV(
                        meta_learner,
                        cv="prefit",
                        method="isotonic"
                    )
                    meta_learner.fit(class_features, labels)
                
                self.meta_learners[class_idx] = meta_learner
                
            except Exception as e:
                print(f"Warning: Failed to fit meta-learner for class {class_idx}: {e}")
                self.meta_learners[class_idx] = None
        
        self.is_fitted = True
        return self
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            predictions: Dict mapping model names to prediction arrays [N, C]
            
        Returns:
            Ensemble predictions [N, C]
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        if self.method == "averaging":
            return self._predict_averaging(predictions)
        else:
            return self._predict_stacking(predictions)
    
    def _predict_averaging(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple averaging of predictions."""
        all_preds = [predictions[name] for name in self.feature_columns]
        return np.mean(all_preds, axis=0)
    
    def _predict_stacking(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stacked prediction using meta-learners."""
        features = self._prepare_features(predictions)
        
        n_samples = features.shape[0]
        final_preds = np.zeros((n_samples, self.num_classes))
        
        for class_idx in range(self.num_classes):
            meta_learner = self.meta_learners.get(class_idx)
            
            if meta_learner is None:
                # Fallback to averaging for this class
                class_preds = [predictions[name][:, class_idx] 
                              for name in self.feature_columns]
                final_preds[:, class_idx] = np.mean(class_preds, axis=0)
            else:
                class_features = self._get_class_features(features, class_idx)
                probs = meta_learner.predict_proba(class_features)
                final_preds[:, class_idx] = probs[:, 1]  # Probability of positive class
        
        return final_preds
    
    def _prepare_features(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Prepare feature matrix from predictions.
        
        For stacking, we use predictions from all models as features.
        """
        # Stack predictions from all models
        feature_list = []
        for name in self.feature_columns:
            if name not in predictions:
                raise ValueError(f"Missing predictions for model: {name}")
            feature_list.append(predictions[name])
        
        # Shape: [N, num_models, num_classes]
        stacked = np.stack(feature_list, axis=1)
        
        # Reshape to [N, num_models * num_classes]
        n_samples = stacked.shape[0]
        return stacked.reshape(n_samples, -1)
    
    def _get_class_features(self, features: np.ndarray, class_idx: int) -> np.ndarray:
        """Extract features relevant to a specific class."""
        # For each model, get the prediction for this class
        n_models = len(self.feature_columns)
        n_samples = features.shape[0]
        
        class_features = np.zeros((n_samples, n_models))
        for model_idx in range(n_models):
            # Features are organized as [model1_class0, model1_class1, ..., model2_class0, ...]
            feature_idx = model_idx * self.num_classes + class_idx
            class_features[:, model_idx] = features[:, feature_idx]
        
        return class_features
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "method": self.method,
            "meta_learner_type": self.meta_learner_type,
            "balanced": self.balanced,
            "calibrate": self.calibrate,
            "feature_columns": self.feature_columns,
            "num_classes": self.num_classes,
            "meta_learners": self.meta_learners,
            "is_fitted": self.is_fitted
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "EnsemblePredictor":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        ensemble = cls(
            method=save_dict["method"],
            meta_learner=save_dict["meta_learner_type"],
            balanced=save_dict["balanced"],
            calibrate=save_dict["calibrate"]
        )
        ensemble.feature_columns = save_dict["feature_columns"]
        ensemble.num_classes = save_dict["num_classes"]
        ensemble.meta_learners = save_dict["meta_learners"]
        ensemble.is_fitted = save_dict["is_fitted"]
        
        return ensemble


def train_ensemble(
    predictions_dir: str,
    ground_truth_path: str,
    output_path: str,
    method: str = "stacking",
    meta_learner: str = "logistic_regression"
) -> EnsemblePredictor:
    """
    Train ensemble from saved predictions.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        ground_truth_path: Path to ground truth CSV
        output_path: Where to save the trained ensemble
        method: 'averaging' or 'stacking'
        meta_learner: 'logistic_regression' or 'random_forest'
        
    Returns:
        Trained EnsemblePredictor
    """
    import pandas as pd
    
    predictions_dir = Path(predictions_dir)
    
    # Load all predictions
    predictions = {}
    for pred_file in predictions_dir.glob("*.csv"):
        name = pred_file.stem
        df = pd.read_csv(pred_file, index_col=0)
        predictions[name] = df.values
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path, index_col=0)
    ground_truth = gt_df.values
    
    # Train ensemble
    ensemble = EnsemblePredictor(
        method=method,
        meta_learner=meta_learner
    )
    ensemble.fit(predictions, ground_truth)
    
    # Save
    ensemble.save(output_path)
    
    return ensemble


def simple_average(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Simple averaging of predictions.
    
    Args:
        predictions: List of prediction arrays, each [N, C]
        
    Returns:
        Averaged predictions [N, C]
    """
    return np.mean(predictions, axis=0)


def weighted_average(
    predictions: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Weighted averaging of predictions.
    
    Args:
        predictions: List of prediction arrays, each [N, C]
        weights: List of weights (should sum to 1)
        
    Returns:
        Weighted average predictions [N, C]
    """
    weights = np.array(weights) / np.sum(weights)
    
    result = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        result += weight * pred
    
    return result
