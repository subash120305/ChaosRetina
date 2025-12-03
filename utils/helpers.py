# =============================================================================
# RIADD Modern - Helper Utilities
# =============================================================================
"""
Common utility functions used across the project.
"""

import os
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find config relative to this file's location
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings can slow down training but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA GPU or CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    
    return device


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def format_number(n: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        n: Number to format
        
    Returns:
        Formatted string
    """
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """
    
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors a metric and stops training if no improvement.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like AUROC, 'min' for loss
            verbose: Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
        
        return self.should_stop


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    scheduler: Optional[Any] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Save path
        scheduler: Optional learning rate scheduler
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint path
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary with epoch and metrics
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {})
    }
