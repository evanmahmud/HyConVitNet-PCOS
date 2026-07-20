<div align="center">

# HyConDViT-Net

### A Vision-Based Hybrid Deep Learning Approach Integrating CNNs, Vision Transformers, and Detection Backbones for PCOS Detection from Ultrasound Imaging

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-1.0-4B8BBE)](https://github.com/huggingface/pytorch-image-models)
[![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](HyConDViT_Net_PCOS_v3_Main.ipynb)

*A reproducible, tri-family hybrid deep-learning framework for the automated
detection of Polycystic Ovary Syndrome from ovarian ultrasound imagery — trained,
validated, and interpreted under a rigorous, clinically honest evaluation protocol.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements Summary](#requirements-summary)
- [System Specifications](#system-specifications)
- [Getting Started](#getting-started)
- [Timing and Computational Cost](#timing-and-computational-cost)
- [Methodology at a Glance](#methodology-at-a-glance)
- [Interpretability](#interpretability)
- [Citation](#citation)
- [Authors and Ownership](#authors-and-ownership)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**HyConDViT-Net** is a hybrid deep-learning framework that fuses **nine
ImageNet-pretrained backbones** drawn from three architecturally distinct families —
**convolutional networks**, **vision transformers**, and **detection backbones** —
into a single calibrated classifier for PCOS detection from ovarian ultrasound
images.

Unlike the many PCOS-classification studies that report near-perfect accuracy on
fewer than two thousand images under a single train/test split — a regime in which
pretrained networks memorise rather than generalise — HyConDViT-Net is trained on a
corpus of **11,784 images**, evaluated under **five-run Monte Carlo cross-validation**
with **bootstrap confidence intervals** and **leave-one-out cross-validation**, and
independently validated on a **fully held-out external corpus of 3,856 images**. The
result is a set of performance figures that are defensible, reproducible, and
representative of behaviour under genuine clinical conditions.

The framework does not merely average its constituents. It learns a **weighted
fusion** over the nine backbones, **temperature-calibrates** each architectural
family, and **optimises the decision threshold** on held-out validation data so that
the operating point is aligned with the clinical priority of minimising missed
diagnoses.

---

## Key Features

| | |
|---|---|
| **Tri-family hybrid** | 3 CNN + 3 Transformer + 3 Detection backbones, fused into one model |
| **Learnable fusion** | Mixture weights on the probability simplex, not uniform voting |
| **Per-family calibration** | Temperature scaling per architectural family |
| **Threshold optimisation** | Decision threshold fitted on validation data (θ\* ≈ 0.36) |
| **Frozen-backbone probing** | Only 1.2–11 % of parameters trainable — overfitting suppressed |
| **Honest evaluation** | 5-run Monte Carlo CV + bootstrap CIs + LOOCV + external validation |
| **Interpretability** | Grad-CAM, attention rollout, and expert radiological review |
| **Deployable** | Knowledge distillation into a single-backbone student (≈9× cheaper) |
| **Reproducible** | Deterministic seeding, auto dataset download, run-level checkpointing |

---

## Architecture

```
                                  ┌─────────────────────────────┐
                                  │      Ovarian Ultrasound     │
                                  │        Image (224×224)      │
                                  └──────────────┬──────────────┘
                                                 │
        ┌────────────────────────┬───────────────┼───────────────┬────────────────────────┐
        ▼                        ▼               ▼               ▼                        ▼
  ┌───────────┐          ┌───────────┐    ┌───────────┐   ┌───────────┐          ┌───────────┐
  │  CNN ×3   │          │ Transformer│   │ Detection │   │    ...     │          │    ...    │
  │ ResNet50  │          │  ViT-B/16  │   │ RT-DETR   │   │  (frozen   │          │  (frozen  │
  │ DenseNet  │          │  Swin-B    │   │ YOLO-NAS  │   │  backbones │          │ backbones)│
  │ EffNetB5  │          │  PiT-B     │   │ EffDet    │   │   + MLP)   │          │           │
  └─────┬─────┘          └─────┬─────┘    └─────┬─────┘   └─────┬─────┘          └─────┬─────┘
        │  softmax(z/τ_cnn)     │ softmax(z/τ_tr) │ softmax(z/τ_det)  ...                  ...
        └────────────────────────┴───────────────┴───────────────┴────────────────────────┘
                                                 │
                                  ┌──────────────▼──────────────┐
                                  │   Learnable Weighted Fusion │
                                  │   p̂ = Σ wᵢ · pᵢ,  w = softmax(α) │
                                  │   threshold θ* (val-optimised)  │
                                  └──────────────┬──────────────┘
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   Infected  /  Non-Infected │
                                  └─────────────────────────────┘
```

The nine constituents, organised by family:

| Family | Models | timm backbone | Trainable |
|---|---|---|:---:|
| **Convolutional** | ResNet50 | `resnet50` | 9.1 % |
| | DenseNet201 | `densenet201` | 11.0 % |
| | EfficientNetB5 | `tf_efficientnet_b5` | 7.7 % |
| **Transformer** | ViT-Base/16 | `vit_base_patch16_224` | 1.2 % |
| | Swin-Base | `swin_base_patch4_window7_224` | 1.5 % |
| | PiT-B | `pit_b_224` | 1.8 % |
| **Detection** | RT-DETR-R50 | `resnet50d` | 9.1 % |
| | YOLO-NAS-B4 | `tf_efficientnet_b4` | 10.7 % |
| | EfficientDet-ECA50 | `ecaresnet50d` | 9.1 % |

---

## Results

Mean over five Monte Carlo runs. **HyConDViT-Net** is the full nine-model hybrid;
**HyConViT-Net (6)** is the CNN + Transformer variant.

| Model | Accuracy | Precision | Recall | F1 | AUC | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Best individual (RT-DETR-R50) | 0.9511 | 0.9830 | 0.9312 | 0.9564 | 0.9924 | 0.9026 |
| Trans-Fusion | 0.9641 | 0.9885 | 0.9487 | 0.9682 | 0.9955 | 0.9281 |
| HyConViT-Net (6) | 0.9500 | 0.9901 | 0.9223 | 0.9550 | 0.9949 | 0.9017 |
| **HyConDViT-Net (9)** | **0.9708** | 0.9637 | **0.9866** | **0.9750** | **0.9955** | **0.9404** |

**Highlights**

- The learnable, calibrated fusion lifts **recall from the 0.81–0.95 band** of the
  individual and uniformly averaged configurations **to 0.9866** — the highest
  sensitivity in the study — at the cost of only a modest precision trade-off.
- **External validation**: 0.992 accuracy on 3,856 independently acquired images
  (every configuration exceeded 0.979).
- **Knowledge distillation**: a single RT-DETR-R50 student reached **0.9654** test
  accuracy — a **+0.0118** gain over the standalone backbone — recovering most of the
  teacher's performance at roughly one-ninth of the inference cost.
- **Radiological validation**: two independent obstetrician–gynaecologists confirmed
  that the model's class-activation maps concentrate on the ovarian stroma and
  perifollicular regions relevant to PCOS assessment.

---

## Repository Structure

```
.
├── HyConDViT_Net_PCOS_v3_Main.ipynb   # End-to-end pipeline (self-contained)
├── Fig/                               # Generated figures (EDA, results, XAI)
│   ├── fig02_feature_distributions_*.png
│   ├── fig03_*.png / fig07_*.png       # EDA: samples, correlation, resolution
│   ├── fig06_results_heatmap.png
│   ├── fig07_cm_individual.png / fig08_cm_hybrid.png
│   ├── fig09_roc_curves.png / fig10_pr_curves.png
│   ├── fig11_threshold_sensitivity.png
│   ├── fig12_mc_stability.png / fig13_accuracy_bar.png
│   ├── fig14_gradcam_*.png / fig15_vit_attention_rollout.png
│   └── fig17_external_validation.png
├── requirements.txt                   # Python dependencies
├── LICENSE
└── README.md
```

---

## Requirements Summary

| Category | Requirement |
|---|---|
| **Language** | Python 3.12 |
| **Deep-learning framework** | PyTorch 2.x + torchvision ≥ 0.15 |
| **Model zoo** | timm ≥ 1.0.0 (all nine pretrained backbones) |
| **Numerics / data** | NumPy ≥ 1.24, pandas ≥ 2.0, SciPy ≥ 1.11 |
| **ML utilities** | scikit-learn ≥ 1.3 (splitting, metrics) |
| **Imaging** | Pillow ≥ 10.0 |
| **Visualisation** | Matplotlib ≥ 3.7, seaborn ≥ 0.13 |
| **Dataset access** | kagglehub ≥ 0.3.4 (automatic download) |
| **Accelerator** | CUDA-capable GPU, ≥ 8 GB VRAM |
| **Disk** | ≈ 5 GB (datasets + checkpoints + figures) |

Install everything with:

```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt</b> (click to expand)</summary>

```
torch>=2.0
torchvision>=0.15
timm>=1.0.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
Pillow>=10.0
matplotlib>=3.7
seaborn>=0.13
kagglehub>=0.3.4
```
</details>

---

## System Specifications

The framework was developed and benchmarked on the following configuration. It is
designed to run on a single consumer-grade GPU and detects its runtime environment
(Colab, Kaggle, Jupyter, or local) automatically.

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5060 |
| **GPU memory** | 8,151 MiB (GDDR6) |
| **Peak training footprint** | ≈ 2.9 GiB |
| **Driver / CUDA** | Driver 595.71.05 · CUDA 13.2 |
| **Operating system** | Linux |
| **Python** | 3.12.13 |
| **Mixed precision** | `bfloat16` automatic mixed precision |
| **Determinism** | Global seeding of `random`, `numpy`, `torch`, and CUDA |

> **Memory efficiency.** The nine-model hybrid is engineered to fit comfortably on
> an 8 GB card: constituents are constructed, evaluated, and moved to host memory
> sequentially; fusion learning runs under a no-gradient context so that only nine
> mixture parameters retain gradients; and accelerator memory is reclaimed between
> models. The peak footprint of ≈ 2.9 GiB leaves ample headroom for concurrent
> workloads.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Datasets — downloaded automatically

Both corpora are fetched by the notebook via `kagglehub`; **no manual download is
required.** The notebook resolves data paths for whichever environment it detects.

| Role | Kaggle dataset | Images |
|---|---|---|
| **Primary** (train / val / test) | [`ibadeus/pcos-xai-ultrasound-dataset`](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset) | 11,784 |
| **External** (held-out validation) | [`anaghachoudhari/pcos-detection-using-ultrasound-images`](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images) | 3,856 |

*(Kaggle authentication may be required on first download; see the
[kagglehub documentation](https://github.com/Kaggle/kagglehub).)*

### 3. Run the pipeline

```bash
jupyter lab HyConDViT_Net_PCOS_v3_Main.ipynb
```

Run all cells top to bottom. The pipeline executes, in order:

1. Environment detection and automatic dataset download
2. Exploratory data analysis and image-level statistics
3. Augmentation (fivefold, applied before partitioning)
4. Individual-backbone training (frozen-backbone linear probing)
5. Partial fine-tuning of the terminal blocks
6. Hybrid fusion, temperature calibration, and threshold optimisation
7. Five-run Monte Carlo cross-validation with bootstrap confidence intervals
8. Leave-one-out cross-validation
9. Knowledge distillation into a single-backbone student
10. External validation on the held-out corpus
11. Interpretability visualisation (Grad-CAM, attention rollout)

Each Monte Carlo run is checkpointed to disk, so an interruption forfeits at most a
single run.

---

## Timing and Computational Cost

Wall-clock times measured on a single NVIDIA RTX 5060.

| Stage | Time | Notes |
|---|---:|---|
| **Five-run Monte Carlo cross-validation** | **1,829.4 min (≈ 30.5 h)** | ≈ 6.1 h per run |
| **Full notebook** (MC-CV + LOOCV + KD + external + all figures) | **2,435.5 min (≈ 40.6 h)** | end to end |
| Per Monte Carlo run | ≈ 6.1 h | 9 backbones + fusion + evaluation |
| Knowledge distillation | 15 epochs | single-backbone student |
| Peak GPU memory | ≈ 2.9 GiB | of 8,151 MiB available |

> **Data footprint per run.** Each run uses a stratified 70/15/15 split at the level
> of original images (8,248 / 1,768 / 1,768), which — after fivefold augmentation —
> yields 41,240 training, 8,840 validation, and 8,840 test instances.

---

## Methodology at a Glance

- **Frozen-backbone linear probing** — backbones are held fixed; only the classifier
  heads and (later) a single terminal block per backbone are trained. This confines
  the trainable-parameter budget to 1.2–11 % and makes any hybrid gain attributable
  to representational diversity rather than added capacity.
- **Augmentation before partitioning** — the corpus is expanded fivefold *before*
  splitting, and all augmented copies of an image are kept in the same partition, so
  train, validation, and test share one difficulty distribution and no leakage
  occurs.
- **Imbalance handling** — a weighted random sampler at the batch level and a focal
  objective with class weights at the loss level.
- **Label-noise injection** — a small fraction of training labels is flipped per run,
  reflecting documented inter-observer disagreement in gynaecological ultrasound.
- **Mixed precision** — training runs in `bfloat16` with a gradient-finiteness guard,
  which removes the numerical overflow that half precision can induce in certain
  backbones.

---

## Interpretability

The repository generates gradient-based class-activation maps for a representative
member of each family and for the fused model, attention-rollout maps for the
transformer constituent, and a correct-versus-incorrect gallery. The activation maps
were reviewed by two practising obstetrician–gynaecologists, who confirmed that the
highlighted regions correspond to clinically meaningful ovarian morphology.

---

## Citation

If you use this work, please cite:

```bibtex
@article{hoque2026hycondvit,
  title   = {HyConDViT-Net: A Vision-Based Hybrid Deep Learning Approach Integrating
             CNNs, Vision Transformers, and Detection Backbones for PCOS Detection
             from Ultrasound Imaging},
  author  = {Hoque, Md Mahmudul and Hasan, Mahmudul},
  year    = {2026}
}
```

---

## Authors and Ownership

This work is owned and maintained by:

| | Author | Role | Contact |
|---|---|---|---|
| **Owner** | **Md Mahmudul Hoque** | Lead author · Conceptualization, methodology, and implementation | [cse.mahmud.evan@gmail.com](mailto:cse.mahmud.evan@gmail.com) |
| **Owner** | **Dr. Mahmudul Hasan** | Principal supervisor · Project administration and review | [mh@cou.ac.bd](mailto:mh@cou.ac.bd) |

**Affiliations** — Department of Computer Science and Engineering, CCN University of
Science and Technology, Cumilla, Bangladesh · MLXperts Lab, Cumilla, Bangladesh ·
Department of Computer Science and Engineering, Comilla University, Cumilla,
Bangladesh.

For questions, collaboration, or licensing enquiries, please contact the owners at
the addresses above.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

We thank the contributors of the publicly available ovarian ultrasound corpora, and
Dr. Ummy Habiba Rekha and Dr. Tanjina Jerin for their independent radiological review
of the model's class-activation maps.
