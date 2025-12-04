# ChaosRetina: Chaos-Based Feature Extraction for Rare Retinal Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## 🔬 Abstract

**ChaosRetina** is a novel deep learning framework designed for the robust classification of rare retinal diseases using the RFMiD dataset. By integrating **Chaos-Based Feature Extraction** into modern Convolutional Neural Networks, this project addresses the limitations of static feature learning in medical imaging.

The core innovation is the **ChaosFEX** module, which introduces deterministic chaotic dynamics (via Generalized Logistic Maps) into the feature space. This approach mimics the non-linear complexity of biological systems, allowing the model to capture subtle, irregular patterns in retinal lesions that are critical for diagnosing rare conditions but are often missed by standard backbones.

**Key Performance (RFMiD Test Set):**
- **Binary Detection (Healthy vs. Disease):** 95.72% AUROC
- **Multi-Label Classification (27 Diseases):** 86.89% AUROC

## 🚀 Key Features

- **Chaotic Feature Extraction (ChaosFEX):** A plug-and-play module that injects chaotic maps (GLS/Logistic) into the feature space to improve separability.
- **Hybrid Architecture:** Combines EfficientNet/DenseNet backbones with chaotic neurons.
- **Robust Training Pipeline:** Implements Asymmetric Loss, Focal Loss, and dynamic class balancing to handle the severe class imbalance in the RFMiD dataset.
- **Dual-Stage Pipeline:**
  1.  **Binary Detector:** Filters healthy vs. sick patients with high sensitivity.
  2.  **Multi-Label Classifier:** Diagnoses specific conditions (DR, ARMD, MH, etc.) for at-risk patients.

## 📂 Project Structure

```
ChaosRetina/
├── config/             # Configuration files (Hyperparameters, Paths)
├── data/               # Dataset loaders and preprocessing logic
├── models/             # Model architectures
│   ├── chaosfex/       # Chaotic dynamics implementation (The Core Novelty)
│   ├── backbones.py    # CNN Backbone definitions
│   ├── classifier.py   # Multi-label classifier wrapper
│   └── detector.py     # Binary detector wrapper
├── scripts/            # Training and Evaluation scripts
│   ├── run_full_pipeline.py  # Main entry point
│   ├── train_classifiers.py  # Classifier training loop
│   ├── train_detector.py     # Detector training loop
│   └── generate_report.py    # Metrics and Visualization
├── training/           # Training utilities (Trainer class, Metrics)
└── outputs/            # Saved models, logs, and predictions
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ChaosRetina.git
    cd ChaosRetina
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    Ensure the RFMiD dataset is placed in the `dataset/` directory as follows:
    ```
    dataset/
    ├── Training_Set/
    ├── Validation_Set/
    └── Test_Set/
    ```

## 🏃‍♂️ Usage

### 1. Full Training Pipeline
To train both the detector and classifier from scratch using the settings in `config.yaml`:

```bash
python scripts/run_full_pipeline.py --config config/config.yaml
```

### 2. Evaluation
To evaluate trained models on the test set and generate a performance report:

```bash
python scripts/generate_report.py --config config/config.yaml
```

### 3. Inference / Visualization
To visualize model predictions on random test samples:

```bash
python scripts/visualize_predictions.py --config config/config.yaml --classifier_path outputs/models/classifier_best.pth --detector_path outputs/models/detector_best.pth
```

## 🧠 The ChaosFEX Module

The core innovation of this project lies in `models/chaosfex`. Instead of standard ReLU activations in the final dense layers, we utilize a **Chaotic Map Layer**:

$$ x_{n+1} = r x_n (1 - x_n) $$

This introduces a controlled non-linearity that helps the model escape local minima and learn more generalized features for rare disease classes.

## 📊 Results

| Model Component | Metric | Score |
|----------------|--------|-------|
| **Binary Detector** | AUROC | **0.9572** |
| **Multi-Label Classifier** | AUROC (Macro) | **0.8689** |

*Detailed confusion matrices and ROC curves can be found in the `outputs/final_results/plots` directory.*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Research Purposes.*
