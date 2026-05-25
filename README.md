# Multi-Layer Defense Pipeline for Skin Cancer Classification

## 📋 Project Overview

This project implements a **3-layer defense pipeline** to protect a skin lesion classification model (HAM10000 dataset) against adversarial attacks (PGD - Projected Gradient Descent). The model distinguishes between **benign** and **malignant** skin lesions using a ResNet34 architecture enhanced with **CBAM (Convolutional Block Attention Module)** attention mechanism.

## 🎯 Problem Statement

Deep learning models for medical image classification are vulnerable to **adversarial attacks** - small, imperceptible perturbations to input images that cause misclassification. This is particularly critical in healthcare, where misdiagnosis could have serious consequences.

## 🔒 3-Layer Defense Architecture

```
Input Image → [Layer 1] → [Layer 2] → [Layer 3] → Prediction
                  ↓           ↓           ↓
            Preprocessing  Adversarial   OOD Detection
              (Blur)        Training     (Mahalanobis)
```

### Layer 1: Preprocessing Defense
- **Technique**: Gaussian Blur + CLAHE + Median Blur
- **Purpose**: Remove high-frequency noise that adversarial attacks exploit
- **Effect**: Slight 0.2% clean accuracy drop, but improves robustness

### Layer 2: Adversarial Training (PGD-AT)
- **Attack**: PGD (Projected Gradient Descent) - 7 iterations
- **Training**: Mix of clean + adversarial images (1:1 ratio)
- **Duration**: 10 additional epochs
- **Result**: PGD attack accuracy improved from **18.8% → 68.8%** (before Layer 1)
- **Combined defense**: PGD + Layer 1 achieves **81.2%**

### Layer 3: OOD Detection (Mahalanobis Distance)
- **Goal**: Detect adversarial examples before classification
- **Method**: Mahalanobis distance in feature space
- **Result**: AUROC of only **0.5244** - not effective (model too robust)

## 📊 Dataset: HAM10000

| Class | Count |
|-------|-------|
| Benign (nv, bkl, df, vasc) | 8,061 |
| Malignant (mel, bcc, akiec) | 1,953 |
| **Total** | **10,014** |

**Class imbalance handled with weighted loss** (class weights: benign=0.62, malignant=2.56)

## 🏗️ Model Architecture

```
ResNet34 (pretrained) + CBAM Attention
├── Channel Attention Module (squeeze-and-excitation)
├── Spatial Attention Module (7×7 conv)
└── Binary classifier head (2 classes)
```

**Parameters**: 21.3M (3.3% more than baseline ResNet34)

## 📈 Results

| Metric | Baseline ResNet34 | ResNet34 + CBAM | + Adversarial Training |
|--------|-------------------|-----------------|------------------------|
| Clean Accuracy | 83.03% | 85.56% | **93.8%** |
| PGD Attack (no defense) | - | 18.8% | **68.8%** |
| PGD + Layer 1 Defense | - | 25.0% | **81.2%** |
| Malignant Recall | 36% | 47.3% | - |

**Key achievement**: +50% improvement in PGD attack robustness!

## 📁 Project Structure

```
HAM10000_binary/           # Dataset (organized)
├── benign/                # 8,061 images
└── malignant/            # 1,953 images

cbam_results/              # Saved models and results
├── best_cbam_model.pth            # Baseline CBAM
├── model_with_adv_training.pth    # After adversarial training
├── final_results_layer2_complete.pkl
├── FINAL_COMPLETE_RESULTS.pkl
└── best_cbam_model_retrained.pth

Simulation.ipynb           # Main notebook with full pipeline
```

## 🚀 Usage

### 1. Mount Google Drive & Load Dataset
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Train Baseline CBAM Model
```python
# ResNet34 + CBAM, 10 epochs
python simulation.ipynb  # Cell 20-22
```

### 3. Apply Layer 1 Defense (Preprocessing)
```python
defense = PreprocessingDefense()  # Gaussian Blur + CLAHE + Median Blur
```

### 4. Train Layer 2 (Adversarial Training)
```python
# PGD attack parameters: ε=8/255, α=2/255, 7 iterations
# 10 additional epochs with clean + adversarial loss
```

### 5. Evaluate Layer 3 (OOD Detection)
```python
# Mahalanobis distance in feature space
# Fails due to model robustness
```

## 📊 Performance Visualization

- **Training curves**: Loss and accuracy over epochs
- **Confusion matrices**: Baseline vs CBAM comparison
- **ROC curves**: OOD detection performance
- **PGD attack impact**: Before/after defense comparison

## 🔬 Key Findings

1. **CBAM improves clean accuracy**: 83.03% → 85.56% (+2.53%)
2. **Malignant recall improvement**: 36% → 47.3% (+11.3%)
3. **Adversarial training dramatically improves robustness**: 18.8% → 68.8% against PGD
4. **Preprocessing adds further robustness**: 68.8% → 81.2% with Layer 1
5. **OOD detection fails when model is too robust** - attacked features become "normal" in feature space

## 💾 Saved Files (in Google Drive)

| File | Size | Description |
|------|------|-------------|
| `best_cbam_model.pth` | 82 MB | Baseline CBAM model |
| `best_cbam_model_retrained.pth` | 82 MB | Retrained baseline |
| `model_with_adv_training.pth` | 82 MB | After adversarial training |
| `FINAL_COMPLETE_RESULTS.pkl` | Small | All final metrics |

## 🛠️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
pandas>=1.3.0
```

## 🎓 Learning Outcomes

- Implementing **CBAM attention mechanism** for medical image classification
- Building **PGD adversarial attacks** from scratch
- **Adversarial training** for model robustness
- **Mahalanobis distance** for OOD detection
- Handling **class imbalance** with weighted loss
- **3-layer defense pipeline** design and evaluation

## 📚 References

- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (PGD)
- [A Simple Unified Framework for Detecting Out-of-Distribution Samples](https://arxiv.org/abs/1807.03888) (Mahalanobis OOD)
- [HAM10000 dataset](https://arxiv.org/abs/1803.10417)

## 👨‍💻 Author

Final year project on adversarial robustness for medical image classification.

---

**Status**: ✅ Complete - Layer 1 & 2 successful, Layer 3 analysis complete
