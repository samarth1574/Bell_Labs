# 🔬 High-Density Object Segmentation with Soft-NMS

> **Improving Dense Object Detection Through Non-Maximum Suppression Alternatives**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Standard **Non-Maximum Suppression (NMS)** uses a hard IoU threshold to discard overlapping detections — but in dense scenes with **1–50 tightly packed objects** (retail shelves, warehouses, inventory), this greedy binary suppression discards *genuinely distinct* overlapping objects, causing **systematic undercounting**.

**Our approach**: Replace Hard NMS with **Soft-NMS** (Bodla et al., 2017), which applies continuous score decay instead of binary suppression:

```
Hard NMS:     s_i = 0                         if IoU(M, b_i) ≥ N_t
Soft-NMS:     s_i = s_i · exp(-IoU² / σ)      (Gaussian decay, σ = 0.5)
```

This preserves detections of genuinely distinct overlapping objects while still suppressing true duplicates — **adding < 1ms overhead** to inference.

---

## Methods

We implement a **four-level progressive pipeline**, from simple baselines to production-ready edge inference:

| Level | Method | Module | Purpose |
|-------|--------|--------|---------|
| 1 | **Heuristic Baseline** | `src/baseline/heuristic.py` | Adaptive threshold + contours + watershed |
| 2 | **Classical CV** | `src/baseline/classical_cv.py` | Watershed, Felzenszwalb, Retail Priors |
| 3 | **Deep Learning + Soft-NMS** | `src/models/detector.py` | Faster R-CNN + Soft-NMS post-processing |
| 4 | **Hybrid Edge Pipeline** | `src/models/edge_pipeline.py` | INT8 ONNX on edge → server refinement |

### Key Innovation: Soft-NMS Module

`src/models/soft_nms.py` implements three NMS variants as special cases of a general rescoring function:

- **Gaussian** (σ=0.5): smooth decay, no threshold — best for dense scenes
- **Linear**: decay proportional to IoU above threshold
- **Hard**: standard binary suppression (baseline)

---

## Datasets

### SKU-110K (Filtered Subset)
- **Source**: Goldman et al. (2019) retail shelf imagery
- **Filter**: 1–50 objects per image (our target density range)
- **Splits**: 70% train / 15% val / 15% test (no image leakage)
- **Loader**: `python -m src.data_loader`

### Synthetic Overlapping Shapes
- **Size**: 500 images (256×256 px), 1–50 objects each
- **Occlusion levels**: 0%, 25%, 50%, 75% (evenly distributed)
- **Format**: COCO annotations + per-instance masks
- **Generator**: `python -m src.synthetic_generator`

---

## Repository Structure

```
Bell_Labs/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── data/
│   ├── raw/                          # SKU-110K annotations
│   ├── processed/                    # Train/val/test split JSONs
│   └── synthetic/                    # Generated images + COCO annotations
│
├── src/
│   ├── data_loader.py                # SKU-110K download, parse, filter, split
│   ├── synthetic_generator.py        # Synthetic dataset with controlled occlusion
│   ├── eda.py                        # 5 EDA analyses + plots
│   │
│   ├── baseline/
│   │   ├── heuristic.py              # Blob/contour + watershed baseline
│   │   └── classical_cv.py           # Watershed, GraphSeg, RetailPrior
│   │
│   ├── models/
│   │   ├── soft_nms.py               # Soft-NMS (Gaussian/Linear/Hard)
│   │   ├── detector.py               # DenseObjectDetector (Faster R-CNN wrapper)
│   │   └── edge_pipeline.py          # ONNX export + edge inference
│   │
│   ├── evaluation/
│   │   └── metrics.py                # IoU, AP, mAP, Count MAE, FPS, full_eval
│   │
│   └── utils/
│       └── visualization.py          # Plotting helpers + style config
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_baseline.ipynb             # Heuristic baseline evaluation
│   └── 03_classical_cv.ipynb         # Classical CV comparison
│
├── experiments/
│   ├── experiment_log.md             # Timestamped experiment records
│   └── run_baselines.py              # Automated baseline runner
│
├── reports/
│   ├── architecture_plan.md          # Full pipeline design document
│   ├── compile_figures.py            # Publication figure generator
│   ├── figures/                      # Generated plots (PNG + PDF)
│   └── latex/
│       ├── main.tex                  # IEEEtran report
│       ├── references.bib            # 5 key paper citations
│       └── soft_nms_theory.tex       # Soft-NMS mathematical analysis
│
└── figures/                          # Additional visualisations
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install
```bash
# Clone
git clone <repo-url> && cd Bell_Labs

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Generate Data
```bash
# Generate synthetic dataset (500 images, 4 occlusion levels)
python -m src.synthetic_generator

# Download and prepare SKU-110K annotations
python -m src.data_loader
```

### Run Analyses
```bash
# EDA plots → reports/figures/
python -m src.eda --dataset all

# Run all baseline experiments
python experiments/run_baselines.py --verbose

# Run Soft-NMS unit tests
python -m src.models.soft_nms

# Run evaluation metrics tests
python -m src.evaluation.metrics --verbose

# Compile publication figures
python reports/compile_figures.py
```

### Deep Learning Inference
```bash
# Run detector with Soft-NMS on an image
python -m src.models.detector --image path/to/img.jpg --nms soft_gaussian --sigma 0.5

# Benchmark inference speed
python -m src.models.detector --image path/to/img.jpg --benchmark

# Edge pipeline demo
python -m src.models.edge_pipeline
```

---

## Preliminary Results

### Baseline Methods on Synthetic Data (Count MAE ↓ better)

| Method | 0% Occ | 25% Occ | 50% Occ | 75% Occ | Overall |
|--------|--------|---------|---------|---------|---------|
| Heuristic | ~2.0 | ~5.5 | ~10.0 | ~15.0 | ~8.1 |
| Watershed | ~1.8 | ~4.8 | ~8.5 | ~13.5 | ~7.2 |
| GraphSeg | ~1.5 | ~4.0 | ~7.5 | ~12.0 | ~6.3 |
| RetailPrior | ~1.6 | ~4.2 | ~7.8 | ~12.5 | ~6.5 |

> **Key finding**: All classical methods degrade at high occlusion. Soft-NMS with DL is expected to significantly improve recall at 50%+ overlap.

### NMS Comparison (Expected)

| NMS Method | mAP@0.5 | Count MAE | Overhead |
|-----------|---------|-----------|----------|
| Hard NMS | Baseline | Baseline | 0.5 ms |
| Soft-NMS Linear | +1–2% | −15% | 0.7 ms |
| Soft-NMS Gaussian | +1–3% | −25% | 0.8 ms |

---

## Hardware Requirements

| Configuration | Minimum | Recommended |
|--------------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | — (CPU-only OK) | NVIDIA GPU (CUDA 11.8+) |
| **Storage** | 2 GB | 10 GB (with datasets) |
| **Python** | 3.8 | 3.10+ |

---

## References

1. **Bodla, N. et al.** (2017). *Soft-NMS — Improving Object Detection With One Line of Code.* ICCV 2017.
2. **Bolya, D. et al.** (2019). *YOLACT: Real-time Instance Segmentation.* ICCV 2019.
3. **Goldman, E. et al.** (2019). *Precise Detection in Densely Packed Scenes.* CVPR 2019. (SKU-110K)
4. **He, K. et al.** (2017). *Mask R-CNN.* ICCV 2017.
5. **Liu, S. et al.** (2019). *Adaptive NMS: Refining Pedestrian Detection in a Crowd.* CVPR 2019.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
