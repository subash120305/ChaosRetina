# =============================================================================
# RIADD Modern - Trainer
# =============================================================================
"""
Training loop with best practices for preventing overfitting.

Features:
- Mixed precision training (saves GPU memory)
- Gradient accumulation (simulates larger batches)
- Learning rate scheduling
- Early stopping
- Comprehensive logging
- Mixup/CutMix regularization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.cuda.amp import GradScaler, autocast # Deprecated
from typing import Dict, Optional, Any, Callable
from pathlib import Path
import time
import json

from utils.helpers import (
    AverageMeter, EarlyStopping, save_checkpoint, ensure_dir
)


class Trainer:
    """
    Training loop handler.
    
    Handles:
    - Training and validation loops
    - Mixed precision (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Logging
    
    Example:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            config=config,
            device=device
        )
        trainer.fit(epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        scheduler: Optional[Any] = None,
        mixup_fn: Optional[Callable] = None,
        experiment_name: str = "experiment"
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            config: Configuration dictionary
            device: Device to train on
            scheduler: Learning rate scheduler
            mixup_fn: Optional mixup/cutmix function
            experiment_name: Name for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixup_fn = mixup_fn
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Training settings from config
        train_config = config.get("training", {})
        self.accumulation_steps = train_config.get("accumulation_steps", 1)
        self.use_amp = train_config.get("use_amp", True) and device.type == "cuda"
        self.clip_grad_norm = train_config.get("clip_grad_norm", 1.0)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Early stopping
        es_config = train_config
        self.early_stopping = EarlyStopping(
            patience=es_config.get("early_stopping_patience", 15),
            min_delta=es_config.get("early_stopping_min_delta", 0.001),
            mode="max",  # We maximize validation AUROC
            verbose=True
        )
        
        # Logging
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "learning_rate": []
        }
        
        # Create output directories
        self.output_dir = Path(config["paths"]["output_dir"]) / experiment_name
        self.models_dir = ensure_dir(self.output_dir / "models")
        self.logs_dir = ensure_dir(self.output_dir / "logs")
        
        # Best metric tracking
        self.best_val_auroc = 0.0
        self.best_epoch = 0
        
        # Wandb logging
        self.use_wandb = config.get("logging", {}).get("wandb", {}).get("enabled", False)
        self.wandb_run = None
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        loss_meter = AverageMeter("Loss")
        
        self.optimizer.zero_grad()
        
        num_batches = len(self.train_loader)
        log_interval = max(1, num_batches // 10)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Apply mixup/cutmix if available
            if self.mixup_fn is not None:
                images, targets_a, targets_b, lam = self.mixup_fn(images, targets)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    if self.mixup_fn is not None:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
                    loss = loss / self.accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                if self.mixup_fn is not None:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.clip_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.clip_grad_norm
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            loss_meter.update(loss.item() * self.accumulation_steps, images.size(0))
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] - Loss: {loss_meter.avg:.4f}")
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_auroc": 0.0}
        
        self.model.eval()
        loss_meter = AverageMeter("Val Loss")
        
        all_outputs = []
        all_targets = []
        
        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            loss_meter.update(loss.item(), images.size(0))
            
            # Collect predictions for metric computation
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        # Compute metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self._compute_metrics(all_outputs, all_targets)
        metrics["val_loss"] = loss_meter.avg
        
        return metrics
    
    def _compute_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Convert to probabilities
        if outputs.dim() == 1 or outputs.size(1) == 1:
            # Binary classification
            probs = torch.sigmoid(outputs).numpy()
            targets_np = targets.numpy()
            
            try:
                auroc = roc_auc_score(targets_np, probs)
            except ValueError:
                auroc = 0.5
        elif outputs.size(1) == 2:
            # Binary classification with 2 outputs (CrossEntropy)
            # Apply softmax and take probability of class 1
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            targets_np = targets.numpy()
            
            try:
                auroc = roc_auc_score(targets_np, probs)
            except ValueError:
                auroc = 0.5
        else:
            # Multi-label classification
            probs = torch.sigmoid(outputs).numpy()
            targets_np = targets.numpy()
            
            try:
                # Macro-averaged AUROC
                auroc = roc_auc_score(
                    targets_np, 
                    probs, 
                    average="macro",
                    multi_class="ovr"
                )
            except ValueError:
                # Some classes may have no positive samples
                auroc = 0.5
            
            try:
                # Mean Average Precision
                mAP = average_precision_score(
                    targets_np,
                    probs,
                    average="macro"
                )
            except ValueError:
                mAP = 0.0
        
        return {
            "val_auroc": auroc,
            "val_mAP": mAP if 'mAP' in dir() else auroc
        }
    
    def fit(self, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"  Training: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Accumulation steps: {self.accumulation_steps}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.2e}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]
            val_auroc = val_metrics["val_auroc"]
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_auroc)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auroc"].append(val_auroc)
            self.history["learning_rate"].append(current_lr)
            
            # Log
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_epoch = epoch
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.models_dir / "best_model.pth",
                    self.scheduler
                )
                print(f"  ★ New best model saved! AUROC: {val_auroc:.4f}")
            
            # Early stopping check
            if self.early_stopping(val_auroc, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            print()
        
        # Training complete
        total_time = time.time() - start_time
        print(f"{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}")
        
        # Save final model and history
        save_checkpoint(
            self.model,
            self.optimizer,
            epochs - 1,
            {"val_auroc": val_auroc},
            self.models_dir / "final_model.pth",
            self.scheduler
        )
        
        self._save_history()
        
        return self.history
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.logs_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved to {history_path}")


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    train_config = config.get("training", {})
    lr = train_config.get("learning_rate", 1e-4)
    weight_decay = train_config.get("weight_decay", 0.01)
    
    # Use differential learning rates if model supports it
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: int
) -> Any:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer
        config: Configuration dictionary
        steps_per_epoch: Number of batches per epoch
        
    Returns:
        Scheduler
    """
    train_config = config.get("training", {})
    scheduler_type = train_config.get("scheduler", "cosine")
    epochs = train_config.get("epochs", 100)
    warmup_epochs = train_config.get("warmup_epochs", 5)
    
    if scheduler_type == "cosine":
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs,
            T_mult=1,
            eta_min=1e-7
        )
    elif scheduler_type == "step":
        # Step decay
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_type == "plateau":
        # Reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler
