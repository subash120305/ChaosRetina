"""
Full Training Pipeline for RIADD Dataset

Orchestrates the complete training process:
1. Train multiple classifier architectures with k-fold CV
2. Train detector models
3. Create ensemble from best models
4. Final evaluation

Usage:
    python scripts/run_full_pipeline.py --config config/config.yaml
    python scripts/run_full_pipeline.py --quick  # Single fold, fewer epochs for testing
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='RIADD Full Training Pipeline')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 1 fold, fewer epochs')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--skip-detector', action='store_true',
                       help='Skip detector training')
    parser.add_argument('--use-chaosfex', action='store_true',
                       help='Use ChaosFEX hybrid models')
    
    return parser.parse_args()


def run_classifier_training(config: dict, output_dir: Path, args) -> dict:
    """Train multi-label classifiers"""
    from scripts.train_classifiers import train_fold, create_model, setup_device
    
    print("\n" + "="*70)
    print("PHASE 1: MULTI-LABEL CLASSIFIER TRAINING")
    print("="*70)
    
    device = setup_device()
    architectures = config.get('training', {}).get('architectures', ['efficientnet_b0'])
    num_folds = 1 if args.quick else config.get('training', {}).get('num_folds', 5)
    
    if args.quick:
        architectures = architectures[:1]  # Only first architecture
    
    all_results = {}
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training classifier: {arch}")
        print('='*60)
        
        # Create mock args for train_fold
        class MockArgs:
            architecture = arch
            folds = num_folds
            epochs = 5 if args.quick else None
            batch_size = None
            lr = None
            use_chaosfex = args.use_chaosfex
            chaos_neurons = 100
            chaos_map = 'GLS'
            loss = 'asymmetric'
            no_wandb = args.no_wandb
            experiment_name = f"classifier_{arch}"
        
        mock_args = MockArgs()
        arch_dir = output_dir / 'classifiers' / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        
        fold_results = []
        for fold in range(num_folds):
            try:
                result = train_fold(fold, config, mock_args, device, arch_dir)
                fold_results.append(result)
            except Exception as e:
                print(f"⚠️  Fold {fold} failed: {e}")
                continue
        
        if fold_results:
            aucs = [r['best_auc'] for r in fold_results]
            all_results[arch] = {
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'fold_aucs': aucs
            }
    
    return all_results


def run_detector_training(config: dict, output_dir: Path, args) -> dict:
    """Train binary detectors"""
    print("\n" + "="*70)
    print("PHASE 2: BINARY DETECTOR TRAINING")
    print("="*70)
    
    # Use simpler detector training
    from scripts.train_detector import train_detector_fold
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architectures = config.get('training', {}).get('architectures', ['efficientnet_b0'])
    num_folds = 1 if args.quick else 3  # Fewer folds for detector
    
    if args.quick:
        architectures = architectures[:1]
    
    all_results = {}
    
    for arch in architectures:
        print(f"\n Training detector: {arch}")
        
        class MockArgs:
            architecture = arch
            batch_size = 16
            lr = 1e-4
            epochs = 3 if args.quick else 20
            config = 'config/config.yaml'
            no_wandb = args.no_wandb
        
        mock_args = MockArgs()
        det_dir = output_dir / 'detectors' / arch
        det_dir.mkdir(parents=True, exist_ok=True)
        
        fold_results = []
        for fold in range(num_folds):
            try:
                result = train_detector_fold(fold, num_folds, config, mock_args, device, det_dir)
                fold_results.append(result)
            except Exception as e:
                print(f"⚠️  Detector fold {fold} failed: {e}")
        
        if fold_results:
            losses = [r['best_val_loss'] for r in fold_results]
            all_results[arch] = {
                'mean_loss': float(np.mean(losses)),
                'std_loss': float(np.std(losses))
            }
    
    return all_results


def create_ensemble_config(classifier_results: dict, output_dir: Path):
    """Create ensemble configuration from trained models"""
    print("\n" + "="*70)
    print("PHASE 3: ENSEMBLE CONFIGURATION")
    print("="*70)
    
    # Sort architectures by AUC
    sorted_archs = sorted(
        classifier_results.items(),
        key=lambda x: x[1]['mean_auc'],
        reverse=True
    )
    
    print("\n📊 Model Performance Ranking:")
    for i, (arch, metrics) in enumerate(sorted_archs):
        print(f"  {i+1}. {arch}: AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    
    # Create ensemble config
    ensemble_config = {
        'models': [],
        'weights': [],
        'method': 'weighted_average'
    }
    
    total_auc = sum(m['mean_auc'] for _, m in sorted_archs)
    
    for arch, metrics in sorted_archs:
        weight = metrics['mean_auc'] / total_auc
        ensemble_config['models'].append({
            'architecture': arch,
            'path': str(output_dir / 'classifiers' / arch / 'fold_0' / 'best_model.pt')
        })
        ensemble_config['weights'].append(weight)
    
    # Save ensemble config
    ensemble_path = output_dir / 'ensemble_config.yaml'
    with open(ensemble_path, 'w') as f:
        yaml.dump(ensemble_config, f)
    
    print(f"\n📁 Ensemble config saved to {ensemble_path}")
    
    return ensemble_config


def generate_final_report(
    classifier_results: dict,
    detector_results: dict,
    output_dir: Path,
    args
):
    """Generate final training report"""
    
    report_path = output_dir / 'TRAINING_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# RIADD Modern Training Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Quick mode: {args.quick}\n")
        f.write(f"- ChaosFEX enabled: {args.use_chaosfex}\n")
        f.write(f"- WandB logging: {not args.no_wandb}\n\n")
        
        f.write("## Multi-label Classifier Results\n\n")
        f.write("| Architecture | Mean AUC | Std AUC |\n")
        f.write("|--------------|----------|--------|\n")
        
        for arch, metrics in classifier_results.items():
            f.write(f"| {arch} | {metrics['mean_auc']:.4f} | {metrics['std_auc']:.4f} |\n")
        
        if detector_results:
            f.write("\n## Binary Detector Results\n\n")
            f.write("| Architecture | Mean Loss | Std Loss |\n")
            f.write("|--------------|-----------|----------|\n")
            
            for arch, metrics in detector_results.items():
                f.write(f"| {arch} | {metrics['mean_loss']:.4f} | {metrics['std_loss']:.4f} |\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review per-class metrics in individual fold directories\n")
        f.write("2. Run ensemble predictions on test set\n")
        f.write("3. Generate submission file for evaluation\n")
    
    print(f"\n📄 Report saved to {report_path}")


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("RIADD MODERN - FULL TRAINING PIPELINE")
    print("="*70)
    
    if args.quick:
        print("⚡ Quick mode enabled - reduced training for testing")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path('outputs') / f'full_pipeline_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Phase 1: Classifiers
    try:
        classifier_results = run_classifier_training(config, output_dir, args)
    except Exception as e:
        print(f"❌ Classifier training failed: {e}")
        classifier_results = {}
    
    # Phase 2: Detectors
    detector_results = {}
    if not args.skip_detector:
        try:
            detector_results = run_detector_training(config, output_dir, args)
        except Exception as e:
            print(f"❌ Detector training failed: {e}")
    
    # Phase 3: Ensemble
    if classifier_results:
        try:
            create_ensemble_config(classifier_results, output_dir)
        except Exception as e:
            print(f"⚠️  Ensemble config failed: {e}")
    
    # Generate report
    generate_final_report(classifier_results, detector_results, output_dir, args)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
