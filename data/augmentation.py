# =============================================================================
# RIADD Modern - Image Augmentation
# =============================================================================
"""
Image augmentation pipelines using Albumentations.
Provides transforms for training, validation, and test-time augmentation.

Augmentation strategy designed to:
1. Prevent overfitting (your concern)
2. Handle class imbalance through diverse augmentations
3. Maintain medical image validity (no unrealistic distortions)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, List


# ImageNet normalization (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    strength: str = "medium"
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size
        strength: Augmentation strength - 'light', 'medium', or 'strong'
        
    Returns:
        Albumentations Compose object
    """
    # Base transforms always applied
    base_transforms = [
        A.Resize(image_size, image_size),
    ]
    
    if strength == "light":
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]
        
    elif strength == "medium":
        aug_transforms = [
            # Geometric transforms (safe for retinal images)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-45, 45),
                shear=(-10, 10),
                p=0.5
            ),
            
            # Color transforms (mimic different camera settings)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Quality transforms (mimic different image qualities)
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Dropout transforms (encourage robustness)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(image_size // 20, image_size // 16),
                hole_width_range=(image_size // 20, image_size // 16),
                p=0.2
            ),
        ]
        
    elif strength == "strong":
        aug_transforms = [
            # More aggressive geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.15, 0.15),
                rotate=(-90, 90),
                shear=(-10, 10),
                p=0.6
            ),
            
            # Aggressive color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            
            # Quality degradation
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
            
            # Stronger dropout
            A.CoarseDropout(
                num_holes_range=(2, 12),
                hole_height_range=(image_size // 16, image_size // 12),
                hole_width_range=(image_size // 16, image_size // 12),
                p=0.3
            ),
            
            # Grid distortion (subtle, medical-safe)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=0.2
            ),
        ]
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    
    # Final normalization and tensor conversion
    final_transforms = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    
    return A.Compose(base_transforms + aug_transforms + final_transforms)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(
    image_size: int = 224,
    n_transforms: int = 5
) -> List[A.Compose]:
    """
    Get test-time augmentation transforms.
    
    Returns multiple transform pipelines to apply during inference.
    Predictions are averaged across all transforms.
    
    Args:
        image_size: Target image size
        n_transforms: Number of different transforms
        
    Returns:
        List of Albumentations Compose objects
    """
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    
    transforms_list = [
        # Original (no augmentation)
        A.Compose(base),
        
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # Vertical flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # 90 degree rotation
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # Slight brightness/contrast variation
        A.Compose([
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]
    
    return transforms_list[:n_transforms]


class MixupCutmix:
    """
    Mixup and CutMix augmentation for regularization.
    
    These techniques help prevent overfitting by:
    - Mixup: Linearly interpolates between two images and their labels
    - CutMix: Pastes a patch from one image onto another
    
    Reference:
    - Mixup: https://arxiv.org/abs/1710.09412
    - CutMix: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 0.5,
        switch_prob: float = 0.5
    ):
        """
        Args:
            mixup_alpha: Mixup beta distribution parameter
            cutmix_alpha: CutMix beta distribution parameter
            mix_prob: Probability of applying any mixing
            switch_prob: Probability of using CutMix vs Mixup
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
    
    def __call__(self, images, labels):
        """
        Apply mixup or cutmix to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, num_classes)
            
        Returns:
            Mixed images and labels
        """
        import torch
        import numpy as np
        
        batch_size = images.size(0)
        
        # Decide whether to apply mixing
        if np.random.random() > self.mix_prob:
            return images, labels, labels, 1.0
        
        # Choose between mixup and cutmix
        use_cutmix = np.random.random() < self.switch_prob
        
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Shuffle indices for mixing
        indices = torch.randperm(batch_size, device=images.device)
        
        if use_cutmix:
            # CutMix
            images_mixed = images.clone()
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual box area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / 
                       (images.size(-1) * images.size(-2)))
        else:
            # Mixup
            images_mixed = lam * images + (1 - lam) * images[indices]
        
        labels_a = labels
        labels_b = labels[indices]
        
        return images_mixed, labels_a, labels_b, lam
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix."""
        import numpy as np
        
        W = size[-1]
        H = size[-2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
