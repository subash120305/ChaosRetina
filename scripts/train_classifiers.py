"""
Train Multi-label Classifier for RIADD Dataset

This script trains individual classifiers for the 28 retinal diseases.
Supports multiple architectures, k-fold cross-validation, and ChaosFEX hybrid models.

Usage:
    python scripts/train_classifiers.py --config config/config.yaml
    python scripts/train_classifiers.py --architecture efficientnet_b0 --folds 5
    python scripts/train_classifiers.py --use-chaosfex --chaos-neurons 100
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
from data.augmentation import get_train_transforms, get_val_transforms
from data.preprocessing import RetinalPreprocessor
from models.classifier import MultiLabelClassifier, get_regularized_classifier
from models.chaosfex import HybridCNNChaosFEX, create_hybrid_classifier
from models.losses import get_loss_function
from training.trainer import Trainer
from evaluation.evaluate import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-label Classifier')
    
    # Config
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    
    # Override options
    parser.add_argument('--architecture', type=str, default=None,
                       help='Backbone architecture (overrides config)')
    parser.add_argument('--folds', type=int, default=None,
                       help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # ChaosFEX options
    parser.add_argument('--use-chaosfex', action='store_true',
                       help='Use hybrid CNN-ChaosFEX model')
    parser.add_argument('--chaos-neurons', type=int, default=100,
                       help='Number of chaotic neurons')
    parser.add_argument('--chaos-map', type=str, default='GLS',
                       choices=['GLS', 'Logistic', 'Hybrid'],
                       help='Chaotic map type')
    
    # Training options
    parser.add_argument('--loss', type=str, default='asymmetric',
                       choices=['asymmetric', 'focal', 'bce', 'smoothing'],
                       help='Loss function')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for models and logs')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for WandB')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device():
    """Setup compute device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU available, using CPU")
    return device


def create_model(config: dict, args, num_classes: int = 28) -> nn.Module:
    """Create model based on configuration"""
    
    architecture = args.architecture or config.get('architecture', 'efficientnet_b0')
    
    if args.use_chaosfex:
        print(f"🔮 Creating Hybrid CNN-ChaosFEX model with {args.chaos_neurons} neurons")
        
        chaos_config = {
            'architecture': architecture,
            'pretrained': True,
            'dropout': config.get('training', {}).get('dropout', 0.3),
            'chaosfex': {
                'n_neurons': args.chaos_neurons,
                'map_type': args.chaos_map,
                'b': config.get('chaosfex', {}).get('b', 0.1),
                'max_iterations': config.get('chaosfex', {}).get('max_iterations', 500),
                'fusion': config.get('chaosfex', {}).get('fusion', 'concat')
            }
        }
        
        model = create_hybrid_classifier(chaos_config, num_classes=num_classes)
    else:
        print(f"🏗️  Creating classifier with {architecture} backbone")
        model = get_regularized_classifier(
            backbone_name=architecture,
            num_classes=num_classes,
            dropout=config.get('training', {}).get('dropout', 0.3),
            pretrained=True
        )
    
    return model


def train_fold(
    fold: int,
    config: dict,
    args,
    device: torch.device,
    output_dir: Path
) -> dict:
    """Train a single fold"""
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.folds or config.get('training', {}).get('num_folds', 5)}")
    print('='*60)
    
    # Get data loaders
    data_config = config.get('paths', {})
    training_config = config.get('training', {})
    
    batch_size = args.batch_size or training_config.get('batch_size', 8)
    num_workers = training_config.get('num_workers', 4)
    
    train_loader, val_loader = get_dataloaders(
        csv_path=data_config.get('train_labels', 'data/RFMiD_Training_Labels.csv'),
        images_dir=data_config.get('train_images', 'data/Training'),
        batch_size=batch_size,
        num_workers=num_workers,
        fold=fold,
        num_folds=args.folds or training_config.get('num_folds', 5),
        task='multilabel'
    )
    
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config, args, num_classes=28)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Total parameters: {total_params:,}")
    print(f"📐 Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    epochs = args.epochs or training_config.get('epochs', 50)
    lr = args.lr or training_config.get('learning_rate', 1e-4)
    
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=args.loss,
        learning_rate=lr,
        weight_decay=training_config.get('weight_decay', 1e-5),
        scheduler_type=training_config.get('scheduler', 'cosine'),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        use_amp=training_config.get('use_amp', True),
        use_wandb=not args.no_wandb and config.get('wandb', {}).get('enabled', False),
        wandb_project=config.get('wandb', {}).get('project', 'riadd-modern'),
        wandb_run_name=f"{args.experiment_name or 'classifier'}_fold{fold}"
    )
    
    # Train
    fold_output = output_dir / f'fold_{fold}'
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=str(fold_output)
    )
    
    # Evaluate best model
    best_model_path = fold_output / 'best_model.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    evaluator = Evaluator(
        model=model,
        device=device,
        threshold='optimal',
        output_dir=str(fold_output / 'evaluation')
    )
    
    metrics = evaluator.evaluate(val_loader, generate_plots=True)
    
    return {
        'fold': fold,
        'history': history,
        'metrics': metrics,
        'best_auc': metrics.get('auc_macro', 0)
    }


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("RIADD Modern - Multi-label Classifier Training")
    print("="*60)
    
    # Load config
    config = load_config(args.config)
    print(f"📁 Loaded config from {args.config}")
    
    # Setup
    device = setup_device()
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    architecture = args.architecture or config.get('architecture', 'efficientnet_b0')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('paths', {}).get('output', 'outputs'))
        output_dir = output_dir / f'classifier_{architecture}_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Training
    num_folds = args.folds or config.get('training', {}).get('num_folds', 5)
    all_results = []
    
    for fold in range(num_folds):
        fold_results = train_fold(fold, config, args, device, output_dir)
        all_results.append(fold_results)
        
        print(f"\n✅ Fold {fold + 1} complete - Best AUC: {fold_results['best_auc']:.4f}")
    
    # Summary
    aucs = [r['best_auc'] for r in all_results]
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\n📊 Cross-validation Results ({num_folds} folds):")
    print(f"   Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"   Best fold: {np.argmax(aucs) + 1} (AUC: {np.max(aucs):.4f})")
    
    # Save summary
    summary = {
        'architecture': architecture,
        'use_chaosfex': args.use_chaosfex,
        'num_folds': num_folds,
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'fold_aucs': aucs
    }
    
    with open(output_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f)
    
    print(f"\n📁 Results saved to {output_dir}")


if __name__ == '__main__':
    main()
