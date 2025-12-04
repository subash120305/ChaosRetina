# =============================================================================
# RIADD Modern - Multi-Label Classifier
# =============================================================================
"""
Multi-label classifier for 28 retinal disease classes.
Uses sigmoid activation for independent per-class predictions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .backbones import create_backbone


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier for retinal disease detection.
    
    Architecture:
        [Image] → [Backbone CNN] → [Global Pool] → [Dropout] → [FC] → [Sigmoid] → [28 probabilities]
    
    Each disease is predicted independently (multi-label, not multi-class).
    An image can have multiple diseases simultaneously.
    
    Example:
        model = MultiLabelClassifier("efficientnet_b0", num_classes=28)
        logits = model(images)  # [B, 28]
        probs = torch.sigmoid(logits)  # [B, 28] probabilities
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 28,
        pretrained: bool = True,
        dropout: float = 0.4,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            backbone_name: Name of the backbone CNN (from timm)
            num_classes: Number of disease classes (28 for RFMiD)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in classification head
            hidden_dim: Optional hidden layer dimension. If None, direct projection.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Create backbone
        self.backbone, num_features = create_backbone(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )
        
        # Classification head
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes)
            )
        
        # Initialize head weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Initialize bias slightly negative for rare classes
                    # This helps with class imbalance at the start
                    nn.init.constant_(m.bias, -0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Logits [B, num_classes] (apply sigmoid for probabilities)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        Useful for ensemble learning.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Features [B, num_features]
        """
        return self.backbone(x)
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone for transfer learning.
        
        Args:
            freeze: If True, freeze backbone weights
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_param_groups(self, base_lr: float) -> list:
        """
        Get parameter groups with differential learning rates.
        
        Backbone gets lower LR (pretrained), head gets higher LR (random init).
        
        Args:
            base_lr: Base learning rate for head
            
        Returns:
            List of param group dicts for optimizer
        """
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.head.parameters(), "lr": base_lr}
        ]


class MultiLabelClassifierWithAttention(nn.Module):
    """
    Multi-label classifier with attention mechanism.
    
    Attention helps the model focus on disease-relevant regions.
    Particularly useful for rare diseases with subtle features.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 28,
        pretrained: bool = True,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Create backbone with feature maps (not pooled)
        self.backbone = create_backbone_with_features(backbone_name, pretrained)
        num_features = self.backbone.num_features
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Get feature maps
        features = self.backbone.forward_features(x)  # [B, C, H, W]
        
        # Compute attention weights
        attn = self.attention(features)  # [B, 1, H, W]
        
        # Apply attention
        features = features * attn
        
        # Pool and classify
        features = self.pool(features).flatten(1)  # [B, C]
        logits = self.head(features)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            attn = self.attention(features)
        return attn


def create_backbone_with_features(name: str, pretrained: bool = True):
    """Create backbone that returns feature maps instead of pooled features."""
    import timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    model.num_features = model.num_features
    return model


def get_regularized_classifier(
    backbone_name: str,
    num_classes: int = 28,
    dropout: float = 0.4,
    pretrained: bool = True
) -> MultiLabelClassifier:
    """
    Factory function to create a regularized classifier.
    Matches the signature expected by train_classifiers.py
    """
    return MultiLabelClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
