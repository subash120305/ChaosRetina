# =============================================================================
# RIADD Modern - Binary Disease Detector
# =============================================================================
"""
Binary detector for Disease_Risk classification.
Predicts whether an image has ANY disease (1) or is healthy (0).
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import create_backbone


class DiseaseDetector(nn.Module):
    """
    Binary classifier for disease risk detection.
    
    Architecture:
        [Image] → [Backbone CNN] → [Global Pool] → [Dropout] → [FC] → [Softmax] → [2 probabilities]
    
    Output:
        - Class 0: No disease risk (healthy)
        - Class 1: Disease risk present
    
    Example:
        model = DiseaseDetector("efficientnet_b0")
        logits = model(images)  # [B, 2]
        probs = torch.softmax(logits, dim=1)  # [B, 2]
        disease_prob = probs[:, 1]  # Probability of disease
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.4,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            backbone_name: Name of the backbone CNN (from timm)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in classification head
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = 2  # Binary classification
        
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
                nn.Linear(hidden_dim, 2)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 2)
            )
        
        # Initialize head weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Logits [B, 2] (apply softmax for probabilities)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get disease probability.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Disease probability [B] (probability of class 1)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # Probability of disease
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
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
        
        Args:
            base_lr: Base learning rate for head
            
        Returns:
            List of param group dicts for optimizer
        """
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.head.parameters(), "lr": base_lr}
        ]


class DiseaseDetectorBCE(nn.Module):
    """
    Binary detector using BCE loss (single output neuron).
    
    Alternative to 2-class softmax, sometimes works better for binary tasks.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        
        # Create backbone
        self.backbone, num_features = create_backbone(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )
        
        # Single output for binary classification
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Logits [B, 1] (apply sigmoid for probability)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(-1)  # [B]
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get disease probability."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
