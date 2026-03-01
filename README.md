# High-Density Object Segmentation with Soft-NMS

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Problem Statement

Many real-world vision systems operate in environments where objects are **densely packed and heavily occluded** — retail shelf monitoring, warehouse inventory counting, crowd analysis, and biological cell imaging. In such settings, traditional object detection systems fail because overlapping objects blur visual boundaries, and standard Non-Maximum Suppression (NMS) aggressively discards valid detections.

This project targets **instance-level detection and segmentation of 1–50 heavily overlapping objects in static images** (retail shelves, crowds, etc.) using **Soft-NMS** to handle occlusion. By replacing hard suppression with continuous score decay, Soft-NMS preserves detections of genuinely distinct but overlapping objects that would otherwise be suppressed.

### Key Innovation

Replacing standard Hard NMS with **Soft-NMS** (Gaussian/Linear score decay) in the post-processing stage of modern object detectors, specifically targeting dense retail/crowd scenes where the assumption of well-separated objects breaks down.

---

## Methods Overview

The project implements a progressive pipeline across four modeling paradigms:

| Level | Method | Description |
|-------|--------|-------------|
| **Baseline** | Blob/Contour Detection | Adaptive thresholding + morphological ops + watershed splitting |
| **Classical CV** | Watershed + Graph Segmentation | Distance-transform watershed, Felzenszwalb segmentation, retail priors |
| **Deep Learning** | Soft-NMS + Pretrained Detector | MobileNetV3/ResNet50 backbone with Soft-NMS post-processing |
| **Hybrid/Edge** | ONNX Edge Pipeline | Lightweight ONNX model on edge device + server refinement |

---

## Dataset

### Primary: SKU-110K (Subset)
- **Source:** Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR 2019
- **Images:** Filtered subset containing 1–50 objects per image
- **Annotations:** Bounding boxes in CSV format
- **Splits:** 70% train / 15% val / 15% test (no shelf overlap across splits)

### Secondary: Synthetic Overlapping Shapes
- **Images:** 500 generated images (256×256 px)
- **Objects:** 1–50 random geometric shapes per image
- **Occlusion levels:** 0%, 25%, 50%, 75% (controlled)
- **Format:** COCO-style JSON annotations + instance masks

---

## Repository Structure

```
Bell_Labs/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore                  # Git ignore rules
├── data/
│   ├── raw/                    # Original dataset files
│   ├── processed/              # Cleaned/filtered data
│   └── synthetic/              # Generated test images
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── eda.py                  # Exploratory data analysis
│   ├── baseline/
│   │   ├── heuristic.py        # Blob/contour baseline
│   │   └── classical_cv.py     # Watershed / graph segmentation
│   ├── models/
│   │   ├── soft_nms.py         # Soft-NMS (Gaussian + Linear)
│   │   ├── detector.py         # DL model wrapper
│   │   └── edge_pipeline.py    # ONNX edge inference
│   ├── evaluation/
│   │   └── metrics.py          # mAP, count MAE, FPS
│   └── utils/
│       └── visualization.py    # Plotting helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_classical_cv.ipynb
│   ├── 04_dl_model.ipynb
│   └── 05_experiments.ipynb
├── experiments/
│   └── experiment_log.md       # Running log of experiments
├── reports/
│   ├── figures/                # Exported plots (PNG/PDF)
│   └── latex/                  # LaTeX report source
└── figures/                    # General figures
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for deep learning experiments

### Install Dependencies

```bash
# Clone the repository
git clone <repo-url> Bell_Labs
cd Bell_Labs

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Generate synthetic dataset
python -m src.synthetic_generator

# 2. Run exploratory data analysis
python -m src.eda

# 3. Run baseline experiments
python experiments/run_baselines.py

# 4. Run DL inference with Soft-NMS
python -m src.models.detector --image path/to/image.jpg --nms soft_gaussian
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **GPU** | — (CPU-only OK for baselines) | NVIDIA GPU, ≥8 GB VRAM |
| **Storage** | 5 GB | 20 GB (full SKU-110K) |
| **Environment** | Local machine | Google Colab (T4 GPU) |

---

## Key References

1. Bodla et al., "Soft-NMS — Improving Object Detection With One Line of Code", **ICCV 2017** · [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
2. Bolya et al., "YOLACT: Real-time Instance Segmentation", **ICCV 2019** · [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf)
3. Liu et al., "Adaptive NMS: Refining Pedestrian Detection in a Crowd", **CVPR 2019**
4. Goldman et al., "Precise Detection in Densely Packed Scenes", **CVPR 2019** (SKU-110K)
5. He et al., "Mask R-CNN", **ICCV 2017**

---

## License

This project is developed for academic research purposes.
