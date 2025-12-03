# =============================================================================
# RIADD Modern - PyTorch Dataset
# =============================================================================
"""
PyTorch Dataset for RFMiD (Retinal Fundus Multi-Disease) dataset.
Replaces AUCMEDI's DataGenerator with a clean, debuggable implementation.

Key improvements over original:
1. No threading bugs (fixes GitHub issues #17, #20)
2. Works on Windows
3. Clear error messages
4. Supports both multi-label classification and binary detection
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import RetinalPreprocessor
from .augmentation import get_train_transforms, get_val_transforms


class RFMiDDataset(Dataset):
    """
    PyTorch Dataset for RFMiD retinal images.
    
    Supports:
    - Multi-label classification (28 disease classes)
    - Binary detection (Disease_Risk: 0 or 1)
    - Custom transforms
    - Retinal-specific preprocessing
    
    Example:
        dataset = RFMiDDataset(
            csv_path="Training_Set/labels.csv",
            image_dir="Training_Set/Training",
            mode="multilabel",
            transform=get_train_transforms(224)
        )
        image, labels = dataset[0]
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        image_dir: Union[str, Path],
        mode: str = "multilabel",
        transform: Optional[Callable] = None,
        image_id_column: str = "ID",
        disease_columns: Optional[List[str]] = None,
        disease_risk_column: str = "Disease_Risk",
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        apply_retinal_preprocessing: bool = True,
        target_size: int = 224,
    ):
        """
        Args:
            csv_path: Path to CSV file with labels
            image_dir: Directory containing images
            mode: 'multilabel' for 28-class or 'binary' for Disease_Risk
            transform: Albumentations transform pipeline
            image_id_column: Column name for image IDs in CSV
            disease_columns: List of disease column names (for multilabel mode)
            disease_risk_column: Column name for binary disease risk
            image_extensions: Valid image file extensions
            apply_retinal_preprocessing: Whether to apply retinal-specific crops
            target_size: Target image size for preprocessing
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.mode = mode
        self.transform = transform
        self.image_id_column = image_id_column
        self.disease_risk_column = disease_risk_column
        self.image_extensions = image_extensions
        self.apply_retinal_preprocessing = apply_retinal_preprocessing
        
        # Default disease columns if not provided
        if disease_columns is None:
            self.disease_columns = [
                "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
                "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP",
                "ODE", "ST", "AION", "PT", "RT", "RS", "CRS", "EDN",
                "RPEC", "MHL", "RP", "OTHER"
            ]
        else:
            self.disease_columns = disease_columns
        
        # Initialize preprocessor
        if apply_retinal_preprocessing:
            self.preprocessor = RetinalPreprocessor(target_size=target_size)
        else:
            self.preprocessor = None
        
        # Load and validate data
        self._load_data()
    
    def _load_data(self) -> None:
        """Load CSV and validate image files exist."""
        # Load CSV
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        if self.image_id_column not in self.df.columns:
            raise ValueError(f"Column '{self.image_id_column}' not found in CSV. "
                           f"Available columns: {list(self.df.columns)}")
        
        if self.mode == "binary":
            if self.disease_risk_column not in self.df.columns:
                raise ValueError(f"Column '{self.disease_risk_column}' not found for binary mode")
        elif self.mode == "multilabel":
            missing_cols = [c for c in self.disease_columns if c not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing disease columns: {missing_cols}")
        
        # Check image directory
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Find image files and match with CSV
        self.samples = []
        missing_images = []
        
        for idx, row in self.df.iterrows():
            image_id = str(row[self.image_id_column])
            
            # Try to find image file
            image_path = self._find_image(image_id)
            
            if image_path is not None:
                # Get labels based on mode
                if self.mode == "binary":
                    label = int(row[self.disease_risk_column])
                else:  # multilabel
                    label = [int(row[col]) for col in self.disease_columns]
                
                self.samples.append({
                    "id": image_id,
                    "path": image_path,
                    "label": label
                })
            else:
                missing_images.append(image_id)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found. Examples: {missing_images[:5]}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image-label pairs found! "
                             f"Check image_dir: {self.image_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {self.csv_path.name}")
    
    def _find_image(self, image_id: str) -> Optional[Path]:
        """Find image file for given ID, trying various extensions."""
        # Clean image_id (remove any extension if present)
        image_id = str(image_id).split('.')[0]
        
        for ext in self.image_extensions:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        
        # Also try with uppercase extensions
        for ext in self.image_extensions:
            path = self.image_dir / f"{image_id}{ext.upper()}"
            if path.exists():
                return path
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample["path"]).convert("RGB")
            image = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {sample['path']}: {e}")
        
        # Apply retinal preprocessing
        if self.preprocessor is not None:
            image = self.preprocessor(image)
        
        # Apply augmentation transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Convert label to tensor
        label = sample["label"]
        if self.mode == "binary":
            label = torch.tensor(label, dtype=torch.long)
        else:  # multilabel
            label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array (useful for computing class weights)."""
        if self.mode == "binary":
            return np.array([s["label"] for s in self.samples])
        else:
            return np.array([s["label"] for s in self.samples])
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for analysis."""
        labels = self.get_labels()
        
        if self.mode == "binary":
            return {
                "negative": int((labels == 0).sum()),
                "positive": int((labels == 1).sum())
            }
        else:
            distribution = {}
            for i, col in enumerate(self.disease_columns):
                distribution[col] = int(labels[:, i].sum())
            return distribution


def get_dataloaders(
    config: Dict,
    train_csv: Union[str, Path],
    train_image_dir: Union[str, Path],
    val_csv: Optional[Union[str, Path]] = None,
    val_image_dir: Optional[Union[str, Path]] = None,
    mode: str = "multilabel"
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        train_csv: Path to training CSV
        train_image_dir: Path to training images
        val_csv: Path to validation CSV (optional)
        val_image_dir: Path to validation images (optional)
        mode: 'multilabel' or 'binary'
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    image_size = config["dataset"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    pin_memory = config["training"]["pin_memory"]
    aug_strength = config["regularization"]["augmentation_strength"]
    
    # Get disease columns from config
    disease_columns = config["dataset"]["disease_columns"]
    
    # Create training dataset
    train_transform = get_train_transforms(image_size, strength=aug_strength)
    train_dataset = RFMiDDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        mode=mode,
        transform=train_transform,
        disease_columns=disease_columns,
        target_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_csv is not None and val_image_dir is not None:
        val_transform = get_val_transforms(image_size)
        val_dataset = RFMiDDataset(
            csv_path=val_csv,
            image_dir=val_image_dir,
            mode=mode,
            transform=val_transform,
            disease_columns=disease_columns,
            target_size=image_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader


def compute_class_weights(labels: np.ndarray, mode: str = "multilabel") -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Uses inverse frequency weighting to give more importance to rare classes.
    
    Args:
        labels: Label array (N,) for binary or (N, num_classes) for multilabel
        mode: 'binary' or 'multilabel'
        
    Returns:
        Class weights tensor
    """
    if mode == "binary":
        # Binary class weights
        n_samples = len(labels)
        n_positive = labels.sum()
        n_negative = n_samples - n_positive
        
        weights = torch.tensor([
            n_samples / (2 * n_negative),
            n_samples / (2 * n_positive)
        ], dtype=torch.float32)
        
    else:  # multilabel
        # Per-class weights based on positive frequency
        n_samples = labels.shape[0]
        n_positive = labels.sum(axis=0)
        n_negative = n_samples - n_positive
        
        # Avoid division by zero
        n_positive = np.maximum(n_positive, 1)
        n_negative = np.maximum(n_negative, 1)
        
        # Weight = n_negative / n_positive (more weight on rare classes)
        weights = torch.tensor(n_negative / n_positive, dtype=torch.float32)
        
        # Clip extreme weights
        weights = torch.clamp(weights, min=0.1, max=10.0)
    
    return weights
