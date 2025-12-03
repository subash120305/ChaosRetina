# =============================================================================
# RIADD Modern - Retinal Image Preprocessing
# =============================================================================
"""
Retinal-specific preprocessing functions.
Ported from original riadd.aucmedi/scripts/retinal_crop.py

The RFMiD dataset contains images from 3 different microscopes with different
resolutions. This module handles the appropriate cropping for each type.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


class RetinalPreprocessor:
    """
    Preprocessor for retinal fundus images.
    
    Handles:
    1. Detection of microscope type based on image dimensions
    2. Center cropping to remove black borders
    3. Padding to square aspect ratio
    4. Resizing to target size
    
    Microscope types in RFMiD:
    - TOPCON 3D OCT-2000: 1424 x 2144 → crop to 1424 x 1424
    - TOPCON TRC-NW300: 1536 x 2048 → crop to 1536 x 1536  
    - Kowa VX-10α: 2848 x 4288 → crop specific region
    """
    
    def __init__(self, target_size: int = 224):
        """
        Args:
            target_size: Output image size (square)
        """
        self.target_size = target_size
        
        # Crop configurations for each microscope type
        # Format: (height, width) -> crop parameters
        self.crop_configs = {
            # After padding to square, these become the dimensions
            (2144, 2144): {"type": "center", "size": 1424},
            (2048, 2048): {"type": "center", "size": 1536},
            (4288, 4288): {"type": "region", "x_min": 248, "x_max": 3712, 
                          "y_min": 408, "y_max": 3872},
        }
    
    def _pad_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Pad image to square aspect ratio.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Square padded image
        """
        h, w = image.shape[:2]
        
        if h == w:
            return image
        
        # Determine padding
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        
        # Pad with zeros (black)
        if len(image.shape) == 3:
            padded = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((max_dim, max_dim), dtype=image.dtype)
        
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
        
        return padded
    
    def _center_crop(self, image: np.ndarray, size: int) -> np.ndarray:
        """
        Center crop image to specified size.
        
        Args:
            image: Input image (must be square)
            size: Crop size
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        if h < size or w < size:
            # Image too small, just return as-is
            return image
        
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        return image[start_h:start_h+size, start_w:start_w+size]
    
    def _region_crop(
        self, 
        image: np.ndarray, 
        x_min: int, 
        x_max: int, 
        y_min: int, 
        y_max: int
    ) -> np.ndarray:
        """
        Crop specific region from image.
        
        Args:
            image: Input image
            x_min, x_max: Horizontal crop bounds
            y_min, y_max: Vertical crop bounds
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        # Clamp to image bounds
        x_min = max(0, min(x_min, w))
        x_max = max(0, min(x_max, w))
        y_min = max(0, min(y_min, h))
        y_max = max(0, min(y_max, h))
        
        return image[y_min:y_max, x_min:x_max]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Alias for __call__ for explicit preprocessing."""
        return self.__call__(image)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline.
        
        Args:
            image: Input image (H, W, C) as numpy array
            
        Returns:
            Preprocessed image resized to target_size
        """
        # Step 1: Pad to square
        image = self._pad_to_square(image)
        
        h, w = image.shape[:2]
        
        # Step 2: Apply microscope-specific crop if recognized
        if (h, w) in self.crop_configs:
            config = self.crop_configs[(h, w)]
            
            if config["type"] == "center":
                image = self._center_crop(image, config["size"])
            elif config["type"] == "region":
                image = self._region_crop(
                    image, 
                    config["x_min"], 
                    config["x_max"],
                    config["y_min"], 
                    config["y_max"]
                )
        
        # Step 3: Resize to target size
        image = cv2.resize(
            image, 
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA if image.shape[0] > self.target_size else cv2.INTER_LINEAR
        )
        
        return image


def remove_black_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove black borders from retinal image.
    
    Some images have excessive black borders that waste resolution.
    This function detects and removes them.
    
    Args:
        image: Input image (H, W, C)
        threshold: Pixel intensity threshold for "black"
        
    Returns:
        Cropped image with minimal borders
    """
    # Convert to grayscale for border detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Find non-black pixels
    mask = gray > threshold
    
    # Find bounding box of non-black region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        # All black, return original
        return image
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add small margin
    margin = 5
    rmin = max(0, rmin - margin)
    rmax = min(image.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(image.shape[1], cmax + margin)
    
    return image[rmin:rmax, cmin:cmax]
