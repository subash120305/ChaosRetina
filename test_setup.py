# =============================================================================
# RIADD Modern - Phase 1-2 Test Script
# =============================================================================
"""
Run this script to verify the basic setup works before proceeding.

Usage:
    cd c:\Repos\ch\riadd_modern
    python test_setup.py

Expected output:
    - Config loaded successfully
    - GPU detection works
    - Dataset can be created (if you've added data)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test all imports work."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not installed: {e}")
        print("    Run: pip install torch torchvision")
        return False
    
    try:
        import timm
        print(f"  ✓ timm {timm.__version__}")
    except ImportError as e:
        print(f"  ✗ timm not installed: {e}")
        print("    Run: pip install timm")
        return False
    
    try:
        import albumentations as A
        print(f"  ✓ albumentations {A.__version__}")
    except ImportError as e:
        print(f"  ✗ albumentations not installed: {e}")
        print("    Run: pip install albumentations")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ pandas not installed: {e}")
        return False
    
    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML not installed: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ OpenCV not installed: {e}")
        print("    Run: pip install opencv-python")
        return False
    
    print("  All core imports successful!\n")
    return True


def test_config():
    """Test config loading."""
    print("=" * 60)
    print("TEST 2: Configuration")
    print("=" * 60)
    
    try:
        from utils.helpers import load_config
        config = load_config()
        print(f"  ✓ Config loaded from config/config.yaml")
        print(f"    - Data directory: {config['paths']['data_dir']}")
        print(f"    - Image size: {config['dataset']['image_size']}")
        print(f"    - Batch size: {config['training']['batch_size']}")
        print(f"    - Architectures: {config['architectures']['classifiers']}")
        print(f"    - ChaosFEX enabled: {config['chaosfex']['enabled']}")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\n" + "=" * 60)
    print("TEST 3: GPU")
    print("=" * 60)
    
    try:
        from utils.helpers import get_device
        device = get_device()
        
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device: {torch.cuda.get_device_name(0)}")
            print(f"    - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Quick memory test
            try:
                x = torch.randn(32, 3, 224, 224).cuda()
                del x
                torch.cuda.empty_cache()
                print(f"  ✓ GPU memory allocation works")
            except Exception as e:
                print(f"  ⚠ GPU memory test failed: {e}")
        else:
            print(f"  ⚠ CUDA not available, will use CPU")
        return True
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        return False


def test_model_creation():
    """Test model creation with timm."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Creation")
    print("=" * 60)
    
    try:
        import timm
        import torch
        
        # Test creating a small model
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=28)
        print(f"  ✓ EfficientNet-B0 created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    - Parameters: {total_params / 1e6:.1f}M")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"    - Input shape: {x.shape}")
        print(f"    - Output shape: {y.shape}")
        print(f"  ✓ Forward pass successful")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_augmentation():
    """Test augmentation pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Augmentation Pipeline")
    print("=" * 60)
    
    try:
        import numpy as np
        from data.augmentation import get_train_transforms, get_val_transforms
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Test train transforms
        train_tf = get_train_transforms(224, "medium")
        result = train_tf(image=dummy_image)
        print(f"  ✓ Train transform created")
        print(f"    - Input shape: {dummy_image.shape}")
        print(f"    - Output shape: {result['image'].shape}")
        
        # Test val transforms
        val_tf = get_val_transforms(224)
        result = val_tf(image=dummy_image)
        print(f"  ✓ Validation transform created")
        print(f"    - Output shape: {result['image'].shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test retinal preprocessing."""
    print("\n" + "=" * 60)
    print("TEST 6: Retinal Preprocessing")
    print("=" * 60)
    
    try:
        import numpy as np
        from data.preprocessing import RetinalPreprocessor
        
        preprocessor = RetinalPreprocessor(target_size=224)
        
        # Test different image sizes (simulating different microscopes)
        test_sizes = [
            (1424, 2144, 3),  # TOPCON 3D OCT-2000
            (1536, 2048, 3),  # TOPCON TRC-NW300
            (2848, 4288, 3),  # Kowa VX-10α
            (800, 800, 3),    # Unknown/generic
        ]
        
        for size in test_sizes:
            dummy = np.random.randint(0, 255, size, dtype=np.uint8)
            result = preprocessor(dummy)
            print(f"  ✓ {size} → {result.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(skip_if_no_data: bool = True):
    """Test dataset loading (requires actual data)."""
    print("\n" + "=" * 60)
    print("TEST 7: Dataset Loading")
    print("=" * 60)
    
    from utils.helpers import load_config
    config = load_config()
    
    data_dir = Path(config["paths"]["data_dir"])
    train_dir = data_dir / config["dataset"]["train_folder"]
    
    # Check if data exists
    if not train_dir.exists():
        if skip_if_no_data:
            print(f"  ⚠ Training data not found at: {train_dir}")
            print(f"    Add your RFMiD dataset to: {data_dir}")
            print(f"    Skipping dataset test...")
            return True
        else:
            print(f"  ✗ Training data not found: {train_dir}")
            return False
    
    # Look for CSV file
    csv_files = list(train_dir.glob("*.csv"))
    if not csv_files:
        print(f"  ⚠ No CSV files found in {train_dir}")
        return True
    
    csv_path = csv_files[0]
    
    # Look for image directory
    possible_image_dirs = ["Training", "images", "Images", "."]
    image_dir = None
    for dirname in possible_image_dirs:
        test_dir = train_dir / dirname if dirname != "." else train_dir
        if test_dir.exists():
            # Check if it has images
            image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
            if image_files:
                image_dir = test_dir
                break
    
    if image_dir is None:
        print(f"  ⚠ No image directory found in {train_dir}")
        return True
    
    try:
        from data.dataset import RFMiDDataset
        from data.augmentation import get_train_transforms
        
        # Create dataset
        transform = get_train_transforms(224, "light")
        dataset = RFMiDDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            mode="multilabel",
            transform=transform,
            target_size=224
        )
        
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Get class distribution
        dist = dataset.get_class_distribution()
        print(f"    - Classes: {len(dist)}")
        print(f"    - Sample distribution (first 5):")
        for i, (k, v) in enumerate(dist.items()):
            if i >= 5:
                break
            print(f"      {k}: {v}")
        
        # Test getting a sample
        image, label = dataset[0]
        print(f"  ✓ Sample retrieved")
        print(f"    - Image shape: {image.shape}")
        print(f"    - Label shape: {label.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  RIADD MODERN - SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("GPU", test_gpu()))
    results.append(("Model", test_model_creation()))
    results.append(("Augmentation", test_augmentation()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Dataset", test_dataset()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("  🎉 All tests passed! Ready for Phase 3 (Models)")
        print("\n  Next steps:")
        print("  1. Add your RFMiD dataset to ./dataset/")
        print("  2. Run this test again to verify dataset loading")
        print("  3. Proceed with model training")
    else:
        print("  ❌ Some tests failed. Please fix issues above.")
        print("\n  If missing packages, run:")
        print("    pip install -r requirements.txt")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
