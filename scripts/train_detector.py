"""
Train Binary Disease Detector for RIADD Dataset

This script trains the binary disease risk detector (Disease_Risk: 0 or 1).
Simpler task than multi-label but important for overall pipeline.

Usage:
    python scripts/train_detector.py --config config/config.yaml
    python scripts/train_detector.py --architecture resnet50 --epochs 30
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import RFMiDDataset, get_dataloaders
from models.detector import DiseaseDetector
from training.trainer import Trainer, create_optimizer, create_scheduler
from evaluation.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Binary Disease Detector')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--architecture', type=str, default='efficientnet_b0',
                       help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    
    return parser.parse_args()


def train_detector_fold(
    fold: int,
    num_folds: int,
    config: dict,
    args,
    device: torch.device,
    output_dir: Path
) -> dict:
    """Train detector for one fold"""
    
    print(f"\n{'='*50}")
    print(f"DETECTOR FOLD {fold + 1}/{num_folds}")
    print('='*50)
    
    # Data loaders (binary task)
    data_config = config.get('paths', {})
    
    train_loader, val_loader = get_dataloaders(
        csv_path=data_config.get('train_labels', 'data/RFMiD_Training_Labels.csv'),
        images_dir=data_config.get('train_images', 'data/Training'),
        batch_size=args.batch_size,
        num_workers=4,
        fold=fold,
        num_folds=num_folds,
        task='binary'  # Only Disease_Risk
    )
    
    print(f"[DATA] Training: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)}")
    
    # Model
    model = DiseaseDetector(
        backbone_name=args.architecture,
        pretrained=True
    )
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    # Calculate steps per epoch for scheduler
    steps_per_epoch = len(train_loader) // config.get('training', {}).get('accumulation_steps', 1)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch)
    
    # Update config with correct output directory for this run
    fold_config = config.copy()
    fold_config['paths'] = config.get('paths', {}).copy()
    fold_config['paths']['output_dir'] = str(output_dir)

    # Trainer with BCE loss for binary
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        config=fold_config,
        device=device,
        scheduler=scheduler,
        experiment_name=f"fold_{fold}"
    )
    
    fold_dir = output_dir / f'fold_{fold}'
    
    history = trainer.fit(
        epochs=args.epochs
    )
    
    # Get best validation metrics
    best_val_loss = min(history.get('val_loss', [float('inf')]))
    
    return {
        'fold': fold,
        'best_val_loss': best_val_loss,
        'history': history
    }


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("RIADD Modern - Binary Disease Detector Training")
    print("="*60)
    
    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path('outputs') / f'detector_{args.architecture}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    all_results = []
    for fold in range(args.folds):
        results = train_detector_fold(fold, args.folds, config, args, device, output_dir)
        all_results.append(results)
    
    # Summary
    losses = [r['best_val_loss'] for r in all_results]
    
    print("\n" + "="*60)
    print("DETECTOR TRAINING COMPLETE")
    print("="*60)
    print(f"[RESULTS] Mean Val Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    
    # Save summary
    summary = {
        'architecture': args.architecture,
        'num_folds': args.folds,
        'mean_loss': float(np.mean(losses)),
        'std_loss': float(np.std(losses))
    }
    
    with open(output_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f)


if __name__ == '__main__':
    main()
