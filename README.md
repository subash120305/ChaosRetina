# RIADD Modern

**Modernized Multi-label Retinal Disease Classification with PyTorch and ChaosFEX**

A complete rewrite of the RIADD challenge solution using modern deep learning practices, replacing the outdated TensorFlow 2.3/AUCMEDI stack with PyTorch 2.x and timm.

## 🔬 What This Project Does

Classifies 28 retinal diseases from fundus images using the RFMiD (Retinal Fundus Multi-disease Image Dataset):

- **Binary Detection**: Disease_Risk (healthy vs. diseased)
- **Multi-label Classification**: 28 specific disease classes including DR, ARMD, Glaucoma, etc.

### Key Improvements Over Original

| Feature | Original (riadd.aucmedi) | Modern (riadd_modern) |
|---------|-------------------------|----------------------|
| Framework | TensorFlow 2.3 | PyTorch 2.x |
| Python | 3.6 | 3.10+ |
| Models | AUCMEDI 0.1.0 | timm 1.0.22 |
| Augmentation | AUCMEDI Image_Augmentation | Albumentations 2.0.8 |
| Multi-label AUC | ~0.70 | Target: 0.85+ |
| Windows GPU | ⚠️ Issues | ✅ Native support |

## 🚀 Quick Start

### Installation

```bash
cd riadd_modern
pip install -r requirements.txt
```

### Verify Setup

```bash
python -c "from models import create_backbone; print('✅ Setup OK')"
```

### Train a Classifier

```bash
# Single model with default settings
python scripts/train_classifiers.py --config config/config.yaml

# With ChaosFEX hybrid model (RECOMMENDED)
python scripts/train_classifiers.py --use-chaosfex --chaos-neurons 100

# Quick test run (1 fold, 5 epochs)
python scripts/run_full_pipeline.py --quick

# Full training with all architectures
python scripts/run_full_pipeline.py --use-chaosfex
```

### After Training: Compute Optimal Thresholds

```bash
# This is CRITICAL for good predictions on imbalanced data
python scripts/compute_thresholds.py --model outputs/classifier_xxx/fold_0/best_model.pt
```

### Run Inference on New Image

```bash
# With TTA and optimal thresholds (RECOMMENDED)
python inference/predict.py \
    --image path/to/retinal_image.jpg \
    --model outputs/classifier_xxx/fold_0/best_model.pt \
    --thresholds outputs/classifier_xxx/fold_0/optimal_thresholds.json

# Or with ensemble
python inference/predict.py \
    --image path/to/retinal_image.jpg \
    --ensemble outputs/ensemble_config.yaml \
    --thresholds outputs/optimal_thresholds.json
```

## 📁 Project Structure

```
riadd_modern/
├── config/
│   └── config.yaml           # Central configuration
├── data/
│   ├── dataset.py            # RFMiD Dataset class
│   ├── preprocessing.py      # Retinal crop preprocessing
│   └── augmentation.py       # Albumentations augmentations
├── models/
│   ├── backbones.py          # timm backbone wrapper
│   ├── classifier.py         # Multi-label classifier
│   ├── detector.py           # Binary disease detector
│   ├── losses.py             # AsymmetricLoss, FocalLoss
│   ├── ensemble.py           # Ensemble predictions
│   └── chaosfex/             # ChaosFEX integration
│       ├── chaos_features.py # GLS map, feature extraction
│       └── hybrid_model.py   # CNN-ChaosFEX hybrid
├── training/
│   └── trainer.py            # Training loop with AMP, WandB
├── evaluation/
│   ├── metrics.py            # AUC, AP, per-class metrics
│   └── evaluate.py           # Evaluation pipeline
├── scripts/
│   ├── train_classifiers.py  # Train multi-label classifiers
│   ├── train_detector.py     # Train binary detector
│   └── run_full_pipeline.py  # Complete training pipeline
└── requirements.txt
```

## 🧪 Key Features

### 1. Modern Architecture Support

10+ pretrained architectures via timm:
- EfficientNet-B0/B2/B4
- DenseNet-121/169/201
- ResNet-50/101/152
- ConvNeXt-Tiny
- And more...

### 2. ChaosFEX Integration

Chaos-based feature extraction for handling class imbalance:

```python
from models.chaosfex import HybridCNNChaosFEX

model = HybridCNNChaosFEX(
    backbone_name='efficientnet_b0',
    num_classes=28,
    n_chaos_neurons=100,
    chaos_map_type='GLS',
    fusion_method='concat'
)
```

### 3. Test-Time Augmentation (TTA)

Critical for reliable predictions - averages multiple augmented versions:

```python
from inference import TTAWrapper, create_tta_wrapper

# Wrap any trained model with TTA
tta_model = create_tta_wrapper(model, mode='light')  # 4 transforms
# or mode='full' for 8 transforms (slower but more accurate)
```

### 4. Per-Class Optimal Thresholds

Essential for imbalanced multi-label - each disease gets its own threshold:

```python
from inference import ThresholdOptimizer

optimizer = ThresholdOptimizer(method='f1')  # or 'recall' for medical
thresholds = optimizer.find_optimal_thresholds(y_true, y_pred_proba)
optimizer.save_thresholds('optimal_thresholds.json')
```

### 5. Advanced Loss Functions

For imbalanced multi-label classification:

- **AsymmetricLoss**: Better handles positive/negative imbalance
- **WeightedFocalLoss**: Class-weighted focal loss
- **LabelSmoothingBCE**: Regularization through smoothing

### 6. Training Best Practices

- Mixed precision training (AMP)
- Cosine annealing with warm restarts
- Early stopping with patience
- K-fold cross-validation
- WandB logging for experiment tracking

## 📊 Expected Results

With proper training:

| Metric | Expected Range |
|--------|---------------|
| AUC-ROC (macro) | 0.80 - 0.88 |
| F1 (macro) | 0.45 - 0.55 |
| Precision (macro) | 0.50 - 0.60 |
| Recall (macro) | 0.40 - 0.50 |

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
training:
  batch_size: 8          # Adjust for your GPU
  epochs: 50
  learning_rate: 0.0001
  num_folds: 5
  architectures:
    - efficientnet_b0
    - densenet121
    - resnet50

chaosfex:
  enabled: true
  n_neurons: 100
  map_type: GLS
  b: 0.1
```

## 🖥️ Hardware Requirements

- **Minimum**: 4GB GPU (GTX 1650), batch_size=8
- **Recommended**: 8GB+ GPU, batch_size=16-32
- **CPU**: Works but slow (not recommended for training)

## 📚 Dataset

Download RFMiD from: [IEEE DataPort](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)

Place in data directory:
```
data/
├── Training/           # Training images
├── Validation/         # Validation images  
├── Test/               # Test images
├── RFMiD_Training_Labels.csv
├── RFMiD_Validation_Labels.csv
└── RFMiD_Testing_Labels.csv
```

## 🤝 Contributing

This is a modernization of the original RIADD challenge solution. Key areas for improvement:

1. Additional loss functions for extreme imbalance
2. Test-time augmentation (TTA)
3. Stacking ensemble methods
4. Semi-supervised learning approaches

## 📝 License

MIT License

## 🙏 Acknowledgments

- Original RIADD challenge organizers
- RFMiD dataset creators
- ChaosFEX-NGRC-RRDC project for chaos-based methods
- timm library by Ross Wightman
