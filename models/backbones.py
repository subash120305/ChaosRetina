# =============================================================================
# RIADD Modern - Backbone Models
# =============================================================================
"""
CNN backbone models using the timm library.
Replaces AUCMEDI's architecture_dict with a cleaner, more flexible approach.

Key benefits:
- 700+ pretrained architectures available
- Consistent API across all models
- Easy to swap architectures with one line
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, List, Optional


# Models optimized for GTX 1650 4GB (sorted by memory usage)
RECOMMENDED_MODELS = {
    # Lightweight (< 2GB VRAM for batch_size=8)
    "efficientnet_b0": {"params": "5.3M", "memory": "low", "accuracy": "good"},
    "mobilenetv3_large_100": {"params": "5.4M", "memory": "low", "accuracy": "good"},
    "efficientnet_b1": {"params": "7.8M", "memory": "low", "accuracy": "good+"},
    
    # Medium (2-3GB VRAM for batch_size=8)
    "densenet121": {"params": "8.0M", "memory": "medium", "accuracy": "good+"},
    "resnet50": {"params": "25.6M", "memory": "medium", "accuracy": "good+"},
    "efficientnet_b2": {"params": "9.2M", "memory": "medium", "accuracy": "better"},
    "convnext_tiny": {"params": "28.6M", "memory": "medium", "accuracy": "better"},
    
    # Heavier (3-4GB VRAM for batch_size=8, may need batch_size=4)
    "efficientnetv2_s": {"params": "21.5M", "memory": "high", "accuracy": "best"},
    "swin_tiny_patch4_window7_224": {"params": "28.3M", "memory": "high", "accuracy": "best"},
    "densenet201": {"params": "20.0M", "memory": "high", "accuracy": "better"},
}


def create_backbone(
    name: str,
    pretrained: bool = True,
    features_only: bool = True,
    in_channels: int = 3
) -> Tuple[nn.Module, int]:
    """
    Create a backbone model from timm library.
    
    Args:
        name: Model name (e.g., 'efficientnet_b0', 'resnet50')
        pretrained: Whether to load pretrained ImageNet weights
        features_only: If True, returns features without classification head
        in_channels: Number of input channels (3 for RGB)
        
    Returns:
        Tuple of (model, num_features)
        - model: The backbone model
        - num_features: Output feature dimension
        
    Example:
        backbone, num_features = create_backbone("efficientnet_b0")
        # backbone(images) returns feature tensor of shape [B, num_features]
    """
    # Validate model exists
    if name not in timm.list_models():
        # Try to find similar names
        similar = [m for m in timm.list_models() if name.lower() in m.lower()][:5]
        raise ValueError(
            f"Model '{name}' not found in timm. "
            f"Similar models: {similar}"
        )
    
    if features_only:
        # Create model without classification head (returns features)
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,  # This removes the head, returns pooled features
            in_chans=in_channels
        )
        num_features = model.num_features
    else:
        # Create full model with head
        model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_channels
        )
        num_features = model.num_features
    
    return model, num_features


def list_available_backbones(filter_pattern: Optional[str] = None) -> List[str]:
    """
    List available backbone models.
    
    Args:
        filter_pattern: Optional pattern to filter models (e.g., 'efficient')
        
    Returns:
        List of model names
    """
    if filter_pattern:
        return timm.list_models(f"*{filter_pattern}*")
    return list(RECOMMENDED_MODELS.keys())


def get_backbone_info(name: str) -> dict:
    """
    Get information about a backbone model.
    
    Args:
        name: Model name
        
    Returns:
        Dictionary with model information
    """
    if name in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[name]
    
    # Get info from timm for unknown models
    try:
        model = timm.create_model(name, pretrained=False, num_classes=0)
        num_params = sum(p.numel() for p in model.parameters())
        return {
            "params": f"{num_params / 1e6:.1f}M",
            "num_features": model.num_features,
            "memory": "unknown",
            "accuracy": "unknown"
        }
    except Exception as e:
        return {"error": str(e)}


class BackboneWithFeatures(nn.Module):
    """
    Wrapper that extracts features from a backbone at multiple scales.
    Useful for feature pyramid networks or attention mechanisms.
    """
    
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 3, 4)
    ):
        """
        Args:
            name: Backbone model name
            pretrained: Use pretrained weights
            out_indices: Which feature levels to extract
        """
        super().__init__()
        
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )
        self.feature_info = self.backbone.feature_info
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors at different scales
        """
        return self.backbone(x)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze backbone parameters.
    Useful for transfer learning - freeze initially, then unfreeze.
    
    Args:
        model: The backbone model
        freeze: If True, freeze parameters; if False, unfreeze
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def get_parameter_groups(
    model: nn.Module,
    backbone_lr_scale: float = 0.1
) -> List[dict]:
    """
    Get parameter groups with different learning rates.
    
    Backbone (pretrained) gets lower learning rate.
    Head (randomly initialized) gets higher learning rate.
    
    Args:
        model: Model with .backbone and .head attributes
        backbone_lr_scale: Scale factor for backbone learning rate
        
    Returns:
        List of parameter group dicts for optimizer
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    return [
        {"params": backbone_params, "lr_scale": backbone_lr_scale},
        {"params": head_params, "lr_scale": 1.0}
    ]
