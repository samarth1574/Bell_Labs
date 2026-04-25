# Architecture Plan — High-Density Object Segmentation with Soft-NMS

> **Project**: Bell Labs — Dense Object Detection & Segmentation  
> **Target**: 1–50 heavily overlapping objects in static images  
> **Key innovation**: Replace Hard NMS with Soft-NMS (Gaussian/Linear score decay)

---

## 1. Architecture Overview

### 1.1 Detection Pipeline (Faster R-CNN + Soft-NMS)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE (H × W × 3)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BACKBONE (Feature Extraction)                   │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │   ResNet-50-FPN   │ or │ MobileNetV3-Large│                      │
│  │  (accuracy mode)  │    │   (speed mode)   │                      │
│  └────────┬─────────┘    └────────┬─────────┘                      │
│           └──────────┬───────────┘                                  │
│                      ▼                                              │
│          Feature Pyramid Network (FPN)                              │
│         P2 ─── P3 ─── P4 ─── P5 ─── P6                             │
│       (1/4)  (1/8) (1/16) (1/32) (1/64)                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  REGION PROPOSAL NETWORK (RPN)                      │
│    Anchor generation → Classification (obj/bg) → Box regression     │
│    Output: ~2000 candidate proposals                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ROI HEAD (Detection)                           │
│    ROI Align → FC layers → Class scores + Box deltas                │
│    Output: ~500 raw detections (score_thresh=0.01)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
          ┌────────────────┐  ┌────────────────────┐
          │   HARD NMS ✗   │  │  SOFT-NMS ✓ (Ours) │
          │  (suppresses   │  │  Gaussian decay:    │
          │   true pos.)   │  │  s_i *= e^(-IoU²/σ) │
          └────────────────┘  └─────────┬──────────┘
                                        │
                                        ▼
                           ┌──────────────────────┐
                           │  FINAL DETECTIONS     │
                           │  Boxes + Scores +     │
                           │  Labels               │
                           └──────────────────────┘
```

### 1.2 YOLACT-Style Instance Segmentation Extension (Future)

```
                    ┌──────────────┐
                    │   Backbone   │
                    │  + FPN (P3)  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌──────────────────┐     ┌──────────────────────┐
    │     ProtoNet      │     │   Detection Head     │
    │  k prototype masks│     │  boxes + scores +    │
    │  (H/4 × W/4 × k) │     │  k coefficients c_i  │
    └────────┬─────────┘     └──────────┬───────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │   Mask Assembly   │
              │  M_i = σ(P · c_i) │
              │  (matrix multiply │
              │   + sigmoid)      │
              └──────────────────┘
```

---

## 2. Soft-NMS Integration Point

### 2.1 Where Soft-NMS Replaces Standard NMS

In the standard Faster R-CNN pipeline, `roi_heads.postprocess_detections()` applies Hard NMS with `torchvision.ops.nms()`. Our modification:

1. **Set permissive built-in thresholds** to let nearly all proposals through:
   ```python
   model.roi_heads.score_thresh = 0.01    # (default: 0.05)
   model.roi_heads.nms_thresh   = 0.95    # (default: 0.5) — nearly disabled
   model.roi_heads.detections_per_img = 500  # (default: 100)
   ```

2. **Apply our `soft_nms()` on the raw outputs**:
   ```python
   raw_boxes, raw_scores, raw_labels = model(image)
   keep, new_scores = soft_nms(raw_boxes, raw_scores, sigma=0.5, method='gaussian')
   final_boxes = raw_boxes[keep]
   ```

### 2.2 Pseudocode

```
FUNCTION soft_nms(B, S, σ, s_t, method):
    INPUT:  B = {b_1, ..., b_N}  bounding boxes
            S = {s_1, ..., s_N}  confidence scores
            σ = decay width       (e.g. 0.5)
            s_t = score threshold  (e.g. 0.001)
    OUTPUT: D = kept detections with updated scores

    D ← ∅
    WHILE B ≠ ∅:
        m ← arg max(S)                    // highest-scoring box
        M ← B[m]
        D ← D ∪ {(M, S[m])}
        B ← B \ {M}

        FOR each b_i ∈ B:
            iou ← IoU(M, b_i)

            IF method = 'gaussian':
                S[i] ← S[i] · exp(-iou² / σ)        // ← KEY CHANGE
            ELIF method = 'linear':
                IF iou ≥ N_t:
                    S[i] ← S[i] · (1 - iou)
            ELIF method = 'hard':
                IF iou ≥ N_t:
                    S[i] ← 0

        B ← {b_i ∈ B : S[i] ≥ s_t}      // prune low-score

    RETURN D
```

### 2.3 Sigma Parameter Guide

| σ value | Behaviour | Best for |
|---------|-----------|----------|
| 0.1 | Aggressive decay — close to Hard NMS | Sparse scenes |
| 0.3 | Moderate decay | Mixed density |
| **0.5** | **Default — balanced** | **General dense scenes** |
| 0.7 | Gentle decay — more boxes survive | Very dense (30+ objects) |
| 1.0 | Very gentle decay | Extreme overlap (75%+) |

---

## 3. Hybrid Edge Pipeline

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EDGE DEVICE                                │
│  ┌───────────────┐   ┌───────────┐   ┌─────────────┐              │
│  │ MobileNetV3   │──▶│ Soft-NMS  │──▶│ Confidence   │              │
│  │  Small (ONNX) │   │ (Gaussian)│   │ Classifier   │              │
│  │  INT8 quant.  │   │           │   │              │              │
│  └───────────────┘   └───────────┘   └──────┬──────┘              │
│     ~8ms inference       ~1ms               │                      │
│                                    ┌────────┴────────┐             │
│                                    │                  │             │
│                              score ≥ 0.7        score < 0.7        │
│                                    │                  │             │
│                                    ▼                  ▼             │
│                           ┌──────────────┐  ┌──────────────┐       │
│                           │ ACCEPT (fast)│  │ COMPRESS ROI │       │
│                           │ Return local │  │ JPEG Q=80    │       │
│                           └──────────────┘  └──────┬───────┘       │
└────────────────────────────────────────────────────┬────────────────┘
                                                     │ ~50KB/ROI
                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            SERVER                                   │
│  ┌────────────────┐   ┌───────────┐   ┌──────────────────┐         │
│  │ ResNet-50-FPN  │──▶│ Soft-NMS  │──▶│ Refined Results  │         │
│  │  FP32 (full)   │   │ σ=0.5    │   │ High-confidence   │         │
│  └────────────────┘   └───────────┘   └──────────────────┘         │
│     ~45ms inference       ~2ms                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Quantization Strategy

| Component | Edge (INT8) | Server (FP32) |
|-----------|-------------|---------------|
| Model size | ~4 MB | ~160 MB |
| Inference | ~8 ms | ~45 ms |
| Accuracy drop | ~1-2% mAP | Baseline |
| Runtime | ONNX Runtime | PyTorch |

### 3.3 Expected Latency Breakdown

| Stage | Edge (ms) | Server (ms) |
|-------|-----------|-------------|
| Image preprocessing | 1 | 2 |
| Backbone + FPN | 6 | 35 |
| RPN + ROI Head | 2 | 8 |
| Soft-NMS | 1 | 2 |
| **Total (local)** | **~10** | **~47** |
| Network RTT (if needed) | +20–50 | — |

---

## 4. Training Plan (Phase 2)

### 4.1 Dataset

- **Source**: SKU-110K filtered to 1–50 objects per image
- **Expected**: ~1,500–3,000 images after filtering
- **Splits**: 70% train / 15% val / 15% test (no image leakage)
- **Supplement**: 500 synthetic images for controlled evaluation

### 4.2 Training Configuration

```yaml
model:
  backbone: mobilenet_v3_large  # or resnet50
  pretrained: COCO (torchvision default weights)
  freeze_backbone: first 2 stages (first 10 epochs)

optimizer:
  type: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  type: CosineAnnealingLR
  T_max: 24  # total epochs
  eta_min: 0.0001

training:
  epochs: 24
  batch_size: 8
  num_workers: 4
  gradient_clip: 5.0

loss:
  classification: CrossEntropy
  box_regression: SmoothL1 (β=1.0)
  mask (optional): BinaryCrossEntropy (if YOLACT extension)
  total: L_cls + L_box + λ·L_mask (λ=1.0)

augmentation:
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - RandomCrop(min_scale=0.8)
  - RandomAffine(degrees=5, translate=(0.05, 0.05))
```

### 4.3 Milestones

| Epoch | Stage |
|-------|-------|
| 1–10 | Frozen backbone layers, train heads only |
| 11–20 | Unfreeze all, full fine-tuning |
| 21–24 | Low LR annealing, final convergence |

---

## 5. Evaluation Plan

### 5.1 Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **mAP@0.5** | Detection accuracy (IoU=0.5) | > 0.60 |
| **mAP@0.5:0.95** | Strict detection accuracy | > 0.35 |
| **Count MAE** | Object counting error | < 3.0 |
| **FPS** | Inference speed | > 10 (GPU), > 2 (CPU) |

### 5.2 Ablation Studies

**A1 — NMS Method Comparison** (fixed σ=0.5, ResNet50):

| Experiment | NMS | Expected Outcome |
|-----------|-----|------------------|
| Exp-1a | Hard (IoU=0.5) | Baseline — undercounts in dense scenes |
| Exp-1b | Soft-NMS Gaussian (σ=0.5) | +1–3% mAP, better recall |
| Exp-1c | Soft-NMS Linear (N_t=0.5) | Similar to Gaussian, sharper cutoff |

**A2 — Sigma Sweep** (Gaussian Soft-NMS, ResNet50):

| σ | Expected |
|---|----------|
| 0.1 | Near-Hard NMS behaviour |
| 0.3 | Moderate improvement |
| **0.5** | **Sweet spot for 1–50 objects** |
| 0.7 | Good for very dense (30+) |
| 1.0 | May over-retain false positives |

**A3 — Backbone Comparison** (Soft-NMS Gaussian, σ=0.5):

| Backbone | Params | Expected mAP | Expected FPS |
|----------|--------|-------------|-------------|
| ResNet-50-FPN | 41M | Higher | ~5 (GPU) |
| MobileNetV3-Large-FPN | 19M | ~2% lower | ~15 (GPU) |

### 5.3 Per-Occlusion Evaluation

Run all methods on synthetic dataset partitioned by occlusion:

```
For each occlusion ∈ {0%, 25%, 50%, 75%}:
    For each method ∈ {Hard, Soft-Gaussian, Soft-Linear}:
        Compute mAP, Count MAE, FPS
        → Line plot: mAP vs occlusion level per method
        → Bar chart: Count MAE vs occlusion level per method
```

---

## 6. Complexity Analysis

### 6.1 Soft-NMS

```
Time complexity:  O(N²)
  - Outer loop: N iterations (one per selected box)
  - Inner loop: up to N IoU computations + score update
  - Same asymptotic cost as standard hard NMS
  - Extra per-comparison: exp() for Gaussian (~5ns overhead)

Space complexity: O(N)
  - Working copy of scores and indices

Practical cost for N=500 proposals:
  - Hard NMS:     ~0.5 ms
  - Soft-NMS:     ~0.8 ms (+60% due to no early termination)
  - Difference:   negligible vs backbone inference (~40ms)
```

### 6.2 Full Pipeline Breakdown

| Component | FLOPs | Time (GPU) | Time (CPU) |
|-----------|-------|------------|------------|
| **MobileNetV3 backbone** | ~0.6G | 4 ms | 40 ms |
| **FPN** | ~0.3G | 2 ms | 15 ms |
| **RPN** | ~0.1G | 1 ms | 8 ms |
| **ROI Align + Head** | ~0.2G | 2 ms | 12 ms |
| **Soft-NMS (N=500)** | ~250K | 1 ms | 1 ms |
| **Total** | **~1.2G** | **~10 ms** | **~76 ms** |

| Component | FLOPs | Time (GPU) | Time (CPU) |
|-----------|-------|------------|------------|
| **ResNet-50 backbone** | ~4.1G | 25 ms | 200 ms |
| **FPN** | ~0.8G | 5 ms | 30 ms |
| **RPN** | ~0.2G | 2 ms | 15 ms |
| **ROI Align + Head** | ~0.5G | 5 ms | 25 ms |
| **Soft-NMS (N=500)** | ~250K | 1 ms | 1 ms |
| **Total** | **~5.6G** | **~38 ms** | **~271 ms** |

### 6.3 Key Insight

> Soft-NMS adds **< 1 ms** overhead compared to Hard NMS, which is
> **< 3% of total inference time**. The accuracy improvement (1–3% mAP)
> comes at essentially zero computational cost — justifying the paper's
> title: "Improving Object Detection With One Line of Code."

---

*Document version: 1.0 — March 2026*
