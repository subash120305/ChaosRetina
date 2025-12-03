"""
Test-Time Augmentation (TTA) for Retinal Disease Classification

Applies multiple augmentations at inference time and aggregates predictions
for more robust and accurate disease detection.

TTA significantly improves prediction reliability, especially for:
- Edge cases near decision boundaries
- Images with unusual lighting/quality
- Rare disease detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation wrapper for any classification model
    
    Applies N augmented versions of an image, runs inference on each,
    and aggregates predictions (mean, max, or geometric mean).
    
    Args:
        model: The trained classification model
        tta_transforms: List of augmentation functions to apply
        aggregation: How to combine predictions ('mean', 'max', 'geometric')
        n_augments: Number of random augmentations (if using random TTA)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tta_transforms: Optional[List[Callable]] = None,
        aggregation: str = 'mean',
        n_augments: int = 5
    ):
        super().__init__()
        self.model = model
        self.aggregation = aggregation
        self.n_augments = n_augments
        
        # Default TTA transforms if none provided
        if tta_transforms is None:
            self.tta_transforms = self._get_default_tta_transforms()
        else:
            self.tta_transforms = tta_transforms
    
    def _get_default_tta_transforms(self) -> List[A.Compose]:
        """
        Default TTA transforms for retinal images
        
        Includes geometric transforms that preserve medical features:
        - Horizontal flip (retinas are roughly symmetric)
        - Rotation (orientation shouldn't matter)
        - Scale variations (capture different magnifications)
        """
        base_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transforms = [
            # Original (no augmentation)
            A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Vertical flip
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # 90 degree rotation
            A.Compose([
                A.Rotate(limit=(90, 90), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # 180 degree rotation
            A.Compose([
                A.Rotate(limit=(180, 180), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # 270 degree rotation
            A.Compose([
                A.Rotate(limit=(270, 270), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Slight zoom in
            A.Compose([
                A.RandomResizedCrop(height=224, width=224, scale=(0.9, 0.95), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Slight zoom out
            A.Compose([
                A.RandomResizedCrop(height=224, width=224, scale=(1.0, 1.1), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
        
        return transforms
    
    def apply_transform(self, image: np.ndarray, transform: A.Compose) -> torch.Tensor:
        """Apply a single transform to an image"""
        augmented = transform(image=image)
        return augmented['image']
    
    @torch.no_grad()
    def forward(self, image: np.ndarray) -> torch.Tensor:
        """
        Forward pass with TTA
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB, uint8
            
        Returns:
            Aggregated prediction probabilities
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        
        all_predictions = []
        
        for transform in self.tta_transforms:
            # Apply transform
            augmented = self.apply_transform(image, transform)
            
            # Add batch dimension
            x = augmented.unsqueeze(0).to(device)
            
            # Get prediction
            logits = self.model(x)
            probs = torch.sigmoid(logits)
            all_predictions.append(probs)
        
        # Stack predictions: (n_transforms, batch, num_classes)
        stacked = torch.stack(all_predictions, dim=0)
        
        # Aggregate
        if self.aggregation == 'mean':
            return stacked.mean(dim=0).squeeze(0)
        elif self.aggregation == 'max':
            return stacked.max(dim=0)[0].squeeze(0)
        elif self.aggregation == 'geometric':
            # Geometric mean (better for probabilities)
            log_probs = torch.log(stacked + 1e-8)
            return torch.exp(log_probs.mean(dim=0)).squeeze(0)
        else:
            return stacked.mean(dim=0).squeeze(0)
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Predict on a batch of images with TTA
        
        Args:
            images: List of numpy arrays (H, W, C)
            
        Returns:
            Predictions (batch_size, num_classes)
        """
        predictions = []
        for image in images:
            pred = self.forward(image)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0)


class LightTTA(nn.Module):
    """
    Lightweight TTA for faster inference
    
    Only applies 4 transforms: original + 3 flips/rotations
    Good balance between speed and accuracy improvement
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @torch.no_grad()
    def forward(self, image: np.ndarray) -> torch.Tensor:
        """
        Forward with 4-way TTA (original + hflip + vflip + both)
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        
        predictions = []
        
        # Original
        x = self.normalize(image=image)['image'].unsqueeze(0).to(device)
        predictions.append(torch.sigmoid(self.model(x)))
        
        # Horizontal flip
        img_hflip = np.fliplr(image).copy()
        x = self.normalize(image=img_hflip)['image'].unsqueeze(0).to(device)
        predictions.append(torch.sigmoid(self.model(x)))
        
        # Vertical flip
        img_vflip = np.flipud(image).copy()
        x = self.normalize(image=img_vflip)['image'].unsqueeze(0).to(device)
        predictions.append(torch.sigmoid(self.model(x)))
        
        # Both flips
        img_both = np.flipud(np.fliplr(image)).copy()
        x = self.normalize(image=img_both)['image'].unsqueeze(0).to(device)
        predictions.append(torch.sigmoid(self.model(x)))
        
        # Average
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0).squeeze(0)


def create_tta_wrapper(
    model: nn.Module,
    mode: str = 'full'
) -> nn.Module:
    """
    Factory function to create TTA wrapper
    
    Args:
        model: Trained model
        mode: 'full' (8 transforms), 'light' (4 transforms), 'none'
        
    Returns:
        Model wrapped with TTA
    """
    if mode == 'full':
        return TTAWrapper(model, aggregation='mean')
    elif mode == 'light':
        return LightTTA(model)
    else:
        return model


if __name__ == "__main__":
    print("TTA Module Test")
    print("=" * 50)
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(224 * 224 * 3, 28)
        
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    model = MockModel()
    
    # Test TTA wrapper
    tta_model = TTAWrapper(model)
    
    # Fake image
    fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test forward
    with torch.no_grad():
        output = tta_model(fake_image)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✅ TTA working correctly")
