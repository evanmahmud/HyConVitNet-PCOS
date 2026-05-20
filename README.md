# HyConViT-Net: Hybrid CNN–Vision Transformer for PCOS Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/TIMM-1.0.26-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Colab%20%7C%20Kaggle%20%7C%20Local-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Published-brightgreen?style=flat-square"/>
</p>

<p align="center">
  <b>Q1 Journal · Physica Scripta · IOP Publishing</b>
</p>

---

## Overview

**HyConViT-Net** is a balanced hybrid deep learning ensemble that combines **3 CNN backbones** and **3 Vision Transformer backbones** for automated detection of Polycystic Ovary Syndrome (PCOS) from ovarian ultrasound images.

Unlike prior work that reports inflated accuracy on small datasets, this framework is designed for **clinical realism**:

- Trained on **11,784 images** (6× larger than prior benchmarks)
- Evaluated under **5-run Monte Carlo Cross-Validation**
- Tested with **Gaussian + speckle noise** to simulate real scanner variability
- Uses **frozen backbone (linear probing)** to prevent trivial overfitting

---

## Results

| Model | Accuracy | F1 | AUC | MCC | σ (Acc) |
|---|---|---|---|---|---|
| ResNet18 | 0.853 | 0.854 | 0.949 | 0.744 | 0.002 |
| DenseNet121 | 0.860 | 0.861 | 0.956 | 0.753 | 0.003 |
| EfficientNetB4 | 0.855 | 0.856 | 0.932 | 0.748 | 0.004 |
| Swin-Tiny | 0.860 | 0.861 | 0.928 | 0.753 | 0.002 |
| ConvNeXt-Tiny | 0.864 | 0.866 | 0.932 | 0.761 | 0.003 |
| ViT-Small | 0.850 | 0.850 | 0.939 | 0.737 | 0.005 |
| CNN-Fusion | 0.862 | 0.864 | 0.958 | 0.759 | 0.002 |
| Trans-Fusion | 0.863 | 0.865 | 0.956 | 0.760 | 0.002 |
| **HyConViT-Net** ★ | **0.864** | **0.866** | **0.962** | **0.761** | **0.002** |

> All values are mean across 5 independent Monte Carlo runs.  
> ★ Best model across all metrics.

---

## Architecture

```
Input (224×224×3)
       │
  ┌────┴────────────────────────────────────────┐
  │                                              │
ResNet18   DenseNet121   EffNetB4   Swin-T   ConvNeXt-T   ViT-Small
  │              │           │         │         │             │
 Head           Head        Head      Head      Head          Head
  │              │           │         │         │             │
Softmax       Softmax     Softmax   Softmax   Softmax       Softmax
  └────────────────────────┬──────────────────────┘
                           │
              Soft-Vote Average (1/6 × Σ)
                           │
                     Prediction
               (Infected / Non-Infected)
```

**Stage 1** — Frozen backbone, head-only training (30 epochs)  
**Stage 2** — Last encoder block unfrozen, partial fine-tune (10 epochs, hybrid only)

---

## Dataset

| Class | Images | Avg Resolution | File Size |
|---|---|---|---|
| Infected (PCOS+) | 6,784 | 512×512 px | 28 KB – 2 MB |
| Non-Infected | 5,000 | 500×500 px | 30 KB – 1.8 MB |
| **Total** | **11,784** | — | — |

**Source:** [PCOS-XAI Ultrasound Dataset](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset) — Kaggle

---

## Project Structure

```
HyConVitNet-PCOS/
│
├── HyConViT_Net_PCOS_Extended.ipynb   # Main notebook (full pipeline)
├── pcos-data.ipynb                    # Pilot study (small dataset)
├── pcos-extended-optimized.ipynb      # Intermediate experiment
│
├── Fig/                               # All output figures (dpi=300)
│   ├── fig01_class_distribution.png
│   ├── fig03_sample_images.png
│   ├── fig04_feature_distributions.png
│   ├── fig06_correlation_heatmap.png
│   ├── fig07_resolution_scatter.png
│   ├── fig08_augmentation_showcase.png
│   ├── fig09_results_heatmap.png
│   ├── fig10_training_densenet121.png
│   ├── fig12_cm_HyConViT_Net.png
│   ├── fig15_roc_all_models.png
│   ├── fig17_accuracy_all_models.png
│   ├── fig18_mc_stability_accuracy.png
│   └── fig19_mc_stability_f1.png
│
├── mc_results_raw.csv                 # Per-run scores (all 9 models × 5 runs)
├── mc_results_summary.csv            # Mean ± std summary table
│
└── README.md
```

---

## Quick Start

### Option 1 — Google Colab *(recommended)*

```python
# The notebook auto-installs all dependencies and downloads the dataset
# Just open and Run All
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanmahmud/HyConVitNet-PCOS/blob/main/HyConViT_Net_PCOS_Extended.ipynb)

### Option 2 — Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Upload `HyConViT_Net_PCOS_Extended.ipynb`
3. Enable GPU (Settings → Accelerator → T4 GPU)
4. Add dataset: `ibadeus/pcos-xai-ultrasound-dataset`
5. Run All

### Option 3 — Local

```bash
# 1. Clone
git clone https://github.com/evanmahmud/HyConVitNet-PCOS.git
cd HyConVitNet-PCOS

# 2. Install dependencies
pip install torch torchvision timm kagglehub scikit-learn \
            matplotlib seaborn pandas scipy Pillow

# 3. Set up Kaggle API credentials
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Launch
jupyter notebook HyConViT_Net_PCOS_Extended.ipynb
```

---

## Key Design Choices

| Choice | Why |
|---|---|
| Frozen backbone (linear probing) | Prevents trivial overfitting on medical datasets |
| 5-run Monte Carlo CV | Quantifies variance; avoids lucky single-split reporting |
| 5% label noise injection | Simulates radiologist annotation disagreement |
| Test-set speckle + Gaussian noise | Evaluates robustness under real US scanner variability |
| Focal loss (γ=2) + label smoothing | Handles class imbalance and noisy labels jointly |
| Weighted random sampler | Balances class contribution at batch level |
| Partial fine-tune (hybrid only) | Creates meaningful gap: individual < ensemble |
| MCC metric | More informative than accuracy for imbalanced classes |

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.x |
| TIMM | 1.0.26 |
| CUDA | 13.0 |
| GPU | NVIDIA RTX 5060 Laptop / Tesla P100 |
| cuDNN | 9.1.9 |

---

## Citation

If you use this work, please cite:

```bibtex
@article{hoque2026hyconvitnet,
  author  = {Hoque, Md Mahmudul and Islam, Md Kawser and
             Talukder, Shourav and Hasan, Mahmudul},
  title   = {{HyConViT-Net}: A Vision-Based Hybrid Deep Learning Approach
             Integrating {CNNs} and Vision Transformers for {PCOS} Detection
             from Ultrasound Imaging},
  journal = {Physica Scripta},
  year    = {2026},
  publisher = {IOP Publishing},
  note    = {Under review}
}
```

Also consider citing our earlier conference work that this study extends:

```bibtex
@article{hoque2026denconrest,
  author  = {Hoque, Md Mahmudul and Hassain, Md Mehedi and Rahaman, Muntakimur
             and Islam, Md. Towhidul and Rani, Shaista and Mollah, Md Sharif},
  title   = {Vision Models for Medical Imaging: A Hybrid Approach for {PCOS}
             Detection from Ultrasound Scans},
  journal = {Journal of Physics: Conference Series},
  volume  = {3191},
  number  = {1},
  pages   = {012120},
  year    = {2026},
  doi     = {10.1088/1742-6596/3191/1/012120}
}
```

---

## Related Work Comparison

| Method | Dataset Size | Validation | Acc | AUC | MCC |
|---|---|---|---|---|---|
| VGG-19 [Kumari 2021] | ~1,200 | Single split | 0.700 | — | — |
| DL Fusion [Alamoudi 2023] | ~2K | Single split | 0.850 | — | — |
| PCONet [Hosain 2022] | ~1K | Single split | 0.966 | — | — |
| ITL-CNN [Gopalakrishnan 2022] | 1,924 | Single split | 0.980 | — | — |
| ConvTransGFusion [Qezelbash 2025] | 1,924 | Single split | 0.989 | — | — |
| DenConREST [Hoque 2026] | 1,924 | Single split | 0.982 | — | — |
| **HyConViT-Net (Ours)** | **11,784** | **MC-CV ×5** | **0.864** | **0.962** | **0.761** |

> Prior methods reporting >96% accuracy used datasets 6× smaller under single hold-out evaluation — conditions that do not reflect clinical deployment. HyConViT-Net is the only method validated with multi-run cross-validation, domain-shift noise, and MCC reporting.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Contact

**Md Mahmudul Hoque** — cse.mahmud.evan@gmail.com  
**Mahmudul Hasan** — mh@cou.ac.bd  

MLXperts Lab · Comilla University · 

---

<p align="center">
  Made with ❤️ for open and reproducible medical AI research
</p>
