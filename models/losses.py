# =============================================================================
# RIADD Modern - Loss Functions
# =============================================================================
"""
Loss functions optimized for multi-label classification with class imbalance.

Key improvements over original focal loss:
1. Asymmetric Loss - better for multi-label imbalance
2. Label smoothing - reduces overconfidence
3. Class weighting - handles rare diseases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    
    Paper: "Asymmetric Loss For Multi-Label Classification" (2021)
    https://arxiv.org/abs/2009.14119
    
    Key insight: In multi-label, most labels are negative (0).
    Standard BCE treats positives and negatives equally.
    Asymmetric loss down-weights easy negatives more aggressively.
    
    This typically improves mAP by 2-4% on imbalanced datasets.
    
    Args:
        gamma_neg: Focusing parameter for negatives (higher = more down-weighting)
        gamma_pos: Focusing parameter for positives
        clip: Probability clipping to prevent log(0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Model outputs before sigmoid [B, C]
            targets: Ground truth labels [B, C] (0 or 1)
            
        Returns:
            Loss value
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Positive and negative parts
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric clipping for negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Asymmetric focusing
        pos_weight = (1 - probs_pos) ** self.gamma_pos
        neg_weight = probs_neg ** self.gamma_neg
        
        # Loss calculation
        loss_pos = targets * pos_weight * torch.log(probs_pos.clamp(min=1e-8))
        loss_neg = (1 - targets) * neg_weight * torch.log(probs_neg.clamp(min=1e-8))
        
        loss = -loss_pos - loss_neg
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    
    Paper: "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
    
    Focuses learning on hard examples by down-weighting easy ones.
    
    Args:
        alpha: Weighting factor (can be per-class tensor)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        For multi-label: logits [B, C], targets [B, C]
        For binary/multiclass: logits [B, C], targets [B] (class indices)
        """
        # Check if multi-label or multi-class
        if targets.dim() == logits.dim():
            # Multi-label case
            return self._focal_loss_multilabel(logits, targets)
        else:
            # Multi-class case
            return self._focal_loss_multiclass(logits, targets)
    
    def _focal_loss_multilabel(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for multi-label classification."""
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Binary cross-entropy with focal weighting
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        loss = focal_weight * bce
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _focal_loss_multiclass(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for multi-class classification."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for imbalanced multi-label classification.
    
    Applies per-class weights based on class frequency.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Args:
            pos_weight: Weight for positive samples per class [C]
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted BCE loss."""
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for multi-label classification.
    
    Softens labels: 1 → (1 - smoothing), 0 → smoothing
    Helps prevent overconfidence and improves generalization.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean"
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute label-smoothed BCE loss."""
        # Smooth labels
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            reduction=self.reduction
        )


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with weights.
    
    Example:
        loss_fn = CombinedLoss({
            'asymmetric': (AsymmetricLoss(), 1.0),
            'smoothing': (LabelSmoothingLoss(0.1), 0.5)
        })
    """
    
    def __init__(self, losses: Dict[str, tuple]):
        """
        Args:
            losses: Dict of {name: (loss_fn, weight)}
        """
        super().__init__()
        
        self.loss_fns = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.loss_fns[name] = loss_fn
            self.weights[name] = weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0
        
        for name, loss_fn in self.loss_fns.items():
            loss = loss_fn(logits, targets)
            total_loss = total_loss + self.weights[name] * loss
        
        return total_loss


def get_loss_function(
    config: Union[Dict[str, Any], str],
    mode: str = "classifier",
    pos_weight: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Factory function to create loss based on config.
    
    Args:
        config: Configuration dictionary OR loss type string
        mode: 'classifier' for multi-label, 'detector' for binary
        pos_weight: Optional class weights for imbalance
        
    Returns:
        Loss function module
    """
    # Handle string input directly
    if isinstance(config, str):
        loss_type = config
        loss_config = {}
    else:
        loss_config = config.get("loss", {})
        
        if mode == "classifier":
            loss_type = loss_config.get("classifier_loss", "asymmetric")
        else:
            loss_type = loss_config.get("detector_loss", "focal")
    
    if loss_type == "asymmetric":
        asl_config = loss_config.get("asymmetric", {})
        return AsymmetricLoss(
            gamma_neg=asl_config.get("gamma_neg", 4),
            gamma_pos=asl_config.get("gamma_pos", 1),
            clip=asl_config.get("clip", 0.05)
        )
    
    elif loss_type == "focal":
        focal_config = loss_config.get("focal", {})
        return FocalLoss(
            alpha=focal_config.get("alpha", 0.25),
            gamma=focal_config.get("gamma", 2.0)
        )
    
    elif loss_type == "bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == "smoothing":
        smoothing = loss_config.get("label_smoothing", 0.1)
        return LabelSmoothingLoss(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class MixupLoss(nn.Module):
    """
    Loss function wrapper for mixup/cutmix training.
    
    When using mixup/cutmix, labels are mixed:
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    """
    
    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion
    
    def forward(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Compute mixed loss.
        
        Args:
            logits: Model predictions
            targets_a: First set of targets
            targets_b: Second set of targets (mixed)
            lam: Mixing coefficient
            
        Returns:
            Mixed loss value
        """
        return lam * self.criterion(logits, targets_a) + \
               (1 - lam) * self.criterion(logits, targets_b)
