# ChaosRetina 🔬
### Chaos-Based Feature Extraction for Multi-Label Retinal Disease Classification

[![Conference](https://img.shields.io/badge/ICKECS-2026-blue.svg)](https://ickecs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: RFMiD](https://img.shields.io/badge/Dataset-RFMiD-green.svg)](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)

> **Accepted at ICKECS 2026** — 4th IEEE International Conference on Knowledge Engineering and Communication Systems

---

## Abstract

**ChaosRetina** is a two-stage deep learning framework for automated multi-label retinal disease detection, integrating a novel **Chaos-Based Feature Extraction (ChaosFEX)** module with CNN backbones.

Standard CNNs extract features through learned linear transformations — missing the subtle, nonlinear patterns that differentiate visually similar retinal diseases. ChaosFEX addresses this by passing CNN feature vectors through **Generalized Luroth Series (GLS) chaotic maps**, amplifying small inter-class differences via sensitive dependence on initial conditions. The result is richer, more discriminative feature representations — especially for rare disease classes.

**Validated on RFMiD (3,200 fundus images, 27 disease classes):**

| Stage | Task | Metric | Score |
|---|---|---|---|
| Stage 1 — Binary Detector | Healthy vs. Disease | AUROC | **0.9572** |
| Stage 1 — Binary Detector | Healthy vs. Disease | Accuracy | **93.8%** |
| Stage 1 — Binary Detector | Healthy vs. Disease | F1-Score | **0.942** |
| Stage 2 — Multi-Label Classifier | 27 Disease Classes | Macro-AUROC | **0.8689** |
| Stage 2 — Multi-Label Classifier | 27 Disease Classes | Weighted AUROC | **0.913** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STAGE 1: Binary Gate                  │
│   Fundus Image → EfficientNet-B0 → Healthy / Diseased   │
└────────────────────────┬────────────────────────────────┘
                         │ (if Diseased)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              STAGE 2: HybridCNNChaosFEX                  │
│                                                          │
│  Image → EfficientNet-B0 Backbone → [1280-dim vector]   │
│                                            │             │
│                              ┌─────────────┘             │
│                              ▼                           │
│                    ChaosFEX Extractor                    │
│                    100 GLS Chaotic Neurons               │
│                    500 iterations each                   │
│                    → [MFT, MFR, ME, MEnt] × 100         │
│                    → [400-dim chaos features]            │
│                              │                           │
│  [1280 CNN] ──── Concat ──── [1280 projected chaos]     │
│                    │                                     │
│              [2560-dim fused]                            │
│                    │                                     │
│         FC(512) → ReLU → Dropout → FC(27)               │
│                    │                                     │
│         Sigmoid → 27 disease probabilities              │
│                    │                                     │
│         Per-class Threshold Optimization                 │
│                    │                                     │
│         Final Multi-Label Prediction                    │
└─────────────────────────────────────────────────────────┘
```

---

## ChaosFEX Module

The core novelty. ChaosFEX transforms CNN feature vectors through chaotic dynamics:

**GLS Map:**
```
x_{n+1} = (x_n + b × x_n²) mod 1     [b = 0.1]
```

For each of 100 chaotic neurons, we run 500 iterations and extract 4 statistical features:

| Feature | Symbol | Description |
|---|---|---|
| Mean Firing Time | MFT | Average gap between threshold crossings |
| Mean Firing Rate | MFR | Crossings per 500 iterations |
| Mean Energy | ME | Mean squared trajectory value |
| Mean Entropy | MEnt | Shannon entropy of trajectory distribution |

**100 neurons × 4 features = 400-dimensional ChaosFEX representation**

ChaosFEX is **parameter-free** — it adds 0 trainable parameters while enriching the feature space through mathematically grounded nonlinear dynamics.

---

## Project Structure

```
ChaosRetina/
├── config/
│   └── config.yaml                 # All hyperparameters and paths
├── models/
│   ├── chaosfex/
│   │   ├── chaos_features.py       # GLS/Logistic maps, ChaosFEXExtractor
│   │   └── hybrid_model.py         # HybridCNNChaosFEX (main model)
│   ├── backbones.py                # timm backbone factory
│   ├── classifier.py               # MultiLabelClassifier
│   ├── detector.py                 # DiseaseDetector (binary)
│   ├── ensemble.py                 # EnsemblePredictor (stacking)
│   └── losses.py                   # AsymmetricLoss, FocalLoss, BCE, Mixup
├── training/
│   └── trainer.py                  # Training loop (AMP, grad accum, early stop)
├── inference/
│   ├── predict.py                  # Inference pipeline
│   ├── threshold_optimizer.py      # Per-class optimal threshold search
│   └── tta.py                      # Test-Time Augmentation
├── evaluation/
│   ├── evaluate.py                 # Full evaluation suite
│   └── metrics.py                  # AUROC, F1, Hamming, per-class metrics
├── scripts/
│   ├── generate_visualizations.py  # t-SNE, ROC curves, performance bars
│   └── architecture_efficiency_benchmark.py  # B0 vs B3 vs ViT benchmark
├── outputs/
│   ├── final_results/              # classifier_predictions.csv, metrics CSVs
│   └── visualizations/             # Generated plots (PNG)
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/subash120305/ChaosRetina.git
cd ChaosRetina

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place RFMiD dataset
# Download from: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid
dataset/
├── Training_Set/Training_Set/
│   ├── Training/          # fundus images
│   └── RFMiD_Training_Labels.csv
├── Evaluation_Set/Evaluation_Set/
│   ├── Validation/
│   └── RFMiD_Validation_Labels.csv
└── Test_Set/Test_Set/
    ├── Test/
    └── RFMiD_Testing_Labels.csv
```

---

## Usage

### Generate Visualizations (Feature Space, ROC, Performance)
```bash
python scripts/generate_visualizations.py
# outputs → outputs/visualizations/
```

### Architecture Efficiency Benchmark
```bash
python scripts/architecture_efficiency_benchmark.py
# Compares EfficientNet-B0, B3, and ViT-B/16 on latency, params, VRAM
```

### Run Evaluation
```bash
python evaluation/evaluate.py --config config/config.yaml
```

---

## Key Design Decisions

### Why EfficientNet-B0 over larger backbones?

Our benchmark shows B0 delivers the optimal accuracy-per-FLOP tradeoff:

| Model | Params | Latency/img | VRAM |
|---|---|---|---|
| **EfficientNet-B0 (ours)** | **4.0M** | **1.94 ms** | **112 MB** |
| EfficientNet-B3 | 10.7M | 6.68 ms | 292 MB |
| ViT-B/16 | 85.8M | 16.85 ms | 400 MB |

ChaosFEX compensates for smaller backbone capacity through nonlinear feature enrichment — a lighter backbone with richer features outperforms heavier backbones alone.

### Why not ViT?
Vision Transformers require pre-training on >100K images (Dosovitskiy et al., 2021). RFMiD has 3,200 images. CNNs' inductive biases (local connectivity, translation invariance) are also critical for localised retinal lesion detection.

### Handling Class Imbalance
- **Focal Loss** (detector): focuses training on hard examples
- **Weighted BCE** (classifier): higher penalty for misclassifying rare diseases
- **Per-class threshold optimization**: rare diseases use lower thresholds (0.2–0.3) vs common ones (0.5)

---

## Results

### Binary Disease Detector

| Metric | Score |
|---|---|
| AUROC | **0.9572** |
| Accuracy | **93.8%** |
| F1-Score | **0.942** |

### Multi-Label Classifier (27 Diseases)

| Metric | Score |
|---|---|
| Macro-AUROC | **0.8689** |
| Weighted AUROC | **0.913** |
| Hamming Accuracy | **94.6%** |
| Macro-F1 | **0.285** |

### Feature Separation (t-SNE)
ChaosFEX produces measurably better-separated feature clusters compared to CNN alone — see `outputs/visualizations/chaosfex_feature_comparison.png`.

---

## Citation

If you use ChaosRetina in your research, please cite:

```bibtex
@inproceedings{chaosretina2026,
  title     = {ChaosRetina: Chaos-Based Feature Extraction for Multi-Label Retinal Disease Classification},
  booktitle = {4th IEEE International Conference on Knowledge Engineering and Communication Systems (ICKECS)},
  year      = {2026},
  month     = {April}
}
```

---

## References

1. Gulshan et al. — "Deep learning for diabetic retinopathy detection" — JAMA 2016
2. Pachade et al. — "RFMiD: Retinal Fundus Multi-Disease Image Dataset" — Data 2021
3. Lin et al. — "Focal Loss for Dense Object Detection" — ICCV 2017
4. Ridnik et al. — "Asymmetric Loss for Multi-Label Classification" — ICCV 2021
5. Tan & Le — "EfficientNet: Rethinking Model Scaling for CNNs" — ICML 2019
6. Dosovitskiy et al. — "An Image is Worth 16x16 Words: Transformers for Image Recognition" — ICLR 2021

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Developed at School of Computing and Information Technology | Guided by Dr. Sindhu P Menon*
