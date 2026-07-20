<div align="center">

# HyConDViT-Net

### A Vision-Based Hybrid Deep Learning Approach Integrating CNNs, Vision Transformers, and Detection Backbones for PCOS Detection from Ultrasound Imaging

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-1.0-4B8BBE)](https://github.com/huggingface/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](HyConDViT_Net_PCOS_v3_Main.ipynb)

*A reproducible, tri-family hybrid deep-learning framework for the automated
detection of Polycystic Ovary Syndrome from ovarian ultrasound imagery — trained,
validated, and interpreted under a rigorous, clinically honest evaluation protocol.*

</div>

---

## Overview

**HyConDViT-Net** is a hybrid deep-learning framework that fuses **nine
ImageNet-pretrained backbones** drawn from three architecturally distinct families
— **convolutional networks**, **vision transformers**, and **detection backbones** —
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

| Family | Models | timm backbone |
|---|---|---|
| **Convolutional** | ResNet50, DenseNet201, EfficientNetB5 | `resnet50`, `densenet201`, `tf_efficientnet_b5` |
| **Transformer** | ViT-Base/16, Swin-Base, PiT-B | `vit_base_patch16_224`, `swin_base_patch4_window7_224`, `pit_b_224` |
| **Detection** | RT-DETR-R50, YOLO-NAS-B4, EfficientDet-ECA50 | `resnet50d`, `tf_efficientnet_b4`, `ecaresnet50d` |

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
├── requirements.txt                   # Python dependencies
├── LICENSE
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.12
- A CUDA-capable GPU with ≥ 8 GB memory (the study used an NVIDIA RTX 5060; peak
  training footprint ≈ 2.9 GiB)
- Core libraries: `torch`, `torchvision`, `timm`, `numpy`, `pandas`,
  `scikit-learn`, `scipy`, `Pillow`, `matplotlib`, `seaborn`, `kagglehub`

```bash
pip install -r requirements.txt
```

### Datasets

Both corpora are downloaded automatically by the notebook via `kagglehub` — no
manual download is required. The notebook detects its environment (Colab, Kaggle,
Jupyter, or local) and resolves the data paths accordingly.

| Role | Kaggle dataset | Images |
|---|---|---|
| **Primary** (train / val / test) | [`ibadeus/pcos-xai-ultrasound-dataset`](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset) | 11,784 |
| **External** (held-out validation) | [`anaghachoudhari/pcos-detection-using-ultrasound-images`](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images) | 3,856 |

### Running

Open the notebook and run all cells top to bottom:

```bash
jupyter lab HyConDViT_Net_PCOS_v3_Main.ipynb
```

The pipeline executes, in order: environment detection and dataset download →
exploratory data analysis → augmentation → individual-backbone training →
partial fine-tuning → hybrid fusion and calibration → Monte Carlo cross-validation →
leave-one-out cross-validation → knowledge distillation → external validation →
interpretability visualisation. Each Monte Carlo run is checkpointed, so an
interruption forfeits at most a single run.

> **Runtime.** The full five-run study takes ≈ 30.5 h of continuous GPU computation;
> the complete notebook (including LOOCV, distillation, external validation, and all
> figures) takes ≈ 40.6 h on a single RTX 5060.

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
transformer constituent, and a correct-versus-incorrect gallery. The activation
maps were reviewed by two practising obstetrician–gynaecologists, who confirmed that
the highlighted regions correspond to clinically meaningful ovarian morphology.

---

## Citation

If you use this work, please cite:

```bibtex
@article{hoque2026hycondvit,
  title   = {HyConDViT-Net: A Vision-Based Hybrid Deep Learning Approach Integrating
             CNNs, Vision Transformers, and Detection Backbones for PCOS Detection
             from Ultrasound Imaging},
  author  = {Hoque, Md Mahmudul and Islam, Md Kawser and Talukder, Shourav and
             Akand, Abdullah Rakib and Hasan, Mahmudul},
  year    = {2026}
}
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

We thank the contributors of the publicly available ovarian ultrasound corpora, and
Dr. Ummy Habiba Rekha and Dr. Tanjina Jerin for their independent radiological
review of the model's class-activation maps.
