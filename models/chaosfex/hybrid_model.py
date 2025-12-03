"""
Hybrid CNN-ChaosFEX Model for Retinal Disease Classification

Combines deep learning feature extraction with chaos-based feature transformation
for improved classification of imbalanced multi-label retinal diseases.

Architecture:
1. CNN Backbone (timm) -> Deep features
2. ChaosFEX Layer -> Chaotic feature transformation
3. Combined classifier head -> Multi-label predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from ..backbones import create_backbone
from .chaos_features import ChaosFEXExtractor


class HybridCNNChaosFEX(nn.Module):
    """
    Hybrid model combining CNN features with ChaosFEX chaotic features
    
    The model extracts deep features from a CNN backbone, transforms them
    through chaotic dynamics, and combines both representations for 
    classification. This helps with:
    - Handling class imbalance through nonlinear feature transformation
    - Capturing complex patterns via chaotic dynamics
    - Improving generalization through diverse feature representations
    
    Args:
        backbone_name: Name of timm backbone
        num_classes: Number of output classes (28 for multi-label, 2 for binary)
        n_chaos_neurons: Number of chaotic neurons
        chaos_map_type: Type of chaotic map ('GLS', 'Logistic', 'Hybrid')
        chaos_b: GLS map parameter
        chaos_iterations: Number of chaotic iterations
        fusion_method: How to combine features ('concat', 'add', 'attention')
        pretrained: Use pretrained backbone weights
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b0',
        num_classes: int = 28,
        n_chaos_neurons: int = 100,
        chaos_map_type: str = 'GLS',
        chaos_b: float = 0.1,
        chaos_iterations: int = 500,
        fusion_method: str = 'concat',
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.n_chaos_neurons = n_chaos_neurons
        
        # CNN backbone for deep feature extraction
        self.backbone, self.backbone_dim = create_backbone(
            backbone_name, 
            pretrained=pretrained
        )
        
        # ChaosFEX extractor (runs on CPU)
        self.chaos_extractor = ChaosFEXExtractor(
            n_neurons=n_chaos_neurons,
            map_type=chaos_map_type,
            b=chaos_b,
            max_iterations=chaos_iterations
        )
        self.chaos_dim = self.chaos_extractor.get_output_dim()
        
        # Feature processing layers
        self.backbone_bn = nn.BatchNorm1d(self.backbone_dim)
        self.chaos_projection = nn.Sequential(
            nn.Linear(self.chaos_dim, self.backbone_dim),
            nn.BatchNorm1d(self.backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion and classification
        if fusion_method == 'concat':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.backbone_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif fusion_method == 'add':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.backbone_dim * 2, self.backbone_dim),
                nn.Tanh(),
                nn.Linear(self.backbone_dim, 2),
                nn.Softmax(dim=1)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from CNN backbone"""
        return self.backbone(x)
    
    def extract_chaos_features(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """
        Transform backbone features through chaotic dynamics
        
        Note: ChaosFEX runs on CPU as it uses NumPy for chaotic map computation
        """
        device = backbone_features.device
        features_np = backbone_features.detach().cpu().numpy()
        
        chaos_features = self.chaos_extractor.extract_features_batch(features_np)
        
        return torch.from_numpy(chaos_features).float().to(device)
    
    def fuse_features(
        self, 
        backbone_features: torch.Tensor, 
        chaos_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse backbone and chaos features based on fusion method"""
        
        backbone_features = self.backbone_bn(backbone_features)
        chaos_features = self.chaos_projection(chaos_features)
        
        if self.fusion_method == 'concat':
            return torch.cat([backbone_features, chaos_features], dim=1)
        elif self.fusion_method == 'add':
            return backbone_features + chaos_features
        elif self.fusion_method == 'attention':
            combined = torch.cat([backbone_features, chaos_features], dim=1)
            weights = self.attention(combined)
            return weights[:, 0:1] * backbone_features + weights[:, 1:2] * chaos_features
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (batch_size, 3, H, W)
            return_features: If True, also return intermediate features
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
            features: (optional) Tuple of (backbone_features, chaos_features, fused_features)
        """
        # Extract backbone features
        backbone_features = self.extract_backbone_features(x)
        
        # Extract chaos features
        chaos_features = self.extract_chaos_features(backbone_features)
        
        # Fuse features
        fused_features = self.fuse_features(backbone_features, chaos_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, (backbone_features, chaos_features, fused_features)
        
        return logits


class ChaosFEXClassifier(nn.Module):
    """
    Simpler ChaosFEX classifier that applies chaos transformation only
    
    This is a lightweight version that uses ChaosFEX as the primary
    feature transformation, followed by a simple classifier.
    Useful when computational resources are limited.
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b0',
        num_classes: int = 28,
        n_chaos_neurons: int = 50,
        chaos_map_type: str = 'GLS',
        pretrained: bool = True
    ):
        super().__init__()
        
        # Backbone
        self.backbone, self.backbone_dim = create_backbone(
            backbone_name, pretrained=pretrained
        )
        
        # ChaosFEX
        self.chaos_extractor = ChaosFEXExtractor(
            n_neurons=n_chaos_neurons,
            map_type=chaos_map_type
        )
        self.chaos_dim = self.chaos_extractor.get_output_dim()
        
        # Simple classifier on chaos features
        self.classifier = nn.Sequential(
            nn.Linear(self.chaos_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get backbone features
        features = self.backbone(x)
        
        # Transform through chaos
        device = features.device
        features_np = features.detach().cpu().numpy()
        chaos_features = self.chaos_extractor.extract_features_batch(features_np)
        chaos_features = torch.from_numpy(chaos_features).float().to(device)
        
        # Classify
        return self.classifier(chaos_features)


def create_hybrid_classifier(
    config: dict,
    num_classes: int = 28
) -> HybridCNNChaosFEX:
    """
    Factory function to create HybridCNNChaosFEX from config
    
    Args:
        config: Configuration dictionary with 'chaosfex' section
        num_classes: Number of output classes
        
    Returns:
        HybridCNNChaosFEX model
    """
    chaos_config = config.get('chaosfex', {})
    
    return HybridCNNChaosFEX(
        backbone_name=config.get('architecture', 'efficientnet_b0'),
        num_classes=num_classes,
        n_chaos_neurons=chaos_config.get('n_neurons', 100),
        chaos_map_type=chaos_config.get('map_type', 'GLS'),
        chaos_b=chaos_config.get('b', 0.1),
        chaos_iterations=chaos_config.get('max_iterations', 500),
        fusion_method=chaos_config.get('fusion', 'concat'),
        pretrained=config.get('pretrained', True),
        dropout_rate=config.get('dropout', 0.3)
    )


if __name__ == "__main__":
    print("Hybrid CNN-ChaosFEX Model Test")
    print("=" * 50)
    
    # Test hybrid model
    model = HybridCNNChaosFEX(
        backbone_name='efficientnet_b0',
        num_classes=28,
        n_chaos_neurons=50,
        fusion_method='concat'
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Backbone dim: {model.backbone_dim}")
    print(f"Chaos dim: {model.chaos_dim}")
    
    # Test with features
    logits, features = model(x, return_features=True)
    backbone_f, chaos_f, fused_f = features
    print(f"\nBackbone features: {backbone_f.shape}")
    print(f"Chaos features: {chaos_f.shape}")
    print(f"Fused features: {fused_f.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
