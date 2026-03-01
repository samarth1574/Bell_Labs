# 🎯 Phase 1 — Prompt Playbook
## High-Density Object Segmentation with Soft-NMS + YOLACT

> **How to use:** Copy-paste each prompt into Antigravity/AI assistant in order.  
> After each prompt, **review the output**, rewrite in your own words where needed, and commit.

---

## TASK 0 — Project Scaffold & Environment Setup

### Prompt 0.1: Create project structure
```
Create the following project structure for a Python computer vision research project called "High-Density Object Segmentation":

Bell_Labs/
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
├── .gitignore             # Python + data gitignore
├── data/
│   ├── raw/               # Original dataset files
│   ├── processed/         # Cleaned/filtered data
│   └── synthetic/         # Generated test images
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading utilities
│   ├── eda.py             # EDA functions
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── heuristic.py   # Blob/contour baseline
│   │   └── classical_cv.py # Watershed / graph-seg
│   ├── models/
│   │   ├── __init__.py
│   │   ├── soft_nms.py    # Soft-NMS implementation
│   │   ├── detector.py    # DL model wrapper
│   │   └── edge_pipeline.py # Hybrid edge inference
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py     # mAP, count MAE, FPS
│   └── utils/
│       ├── __init__.py
│       └── visualization.py # Plotting helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_classical_cv.ipynb
│   ├── 04_dl_model.ipynb
│   └── 05_experiments.ipynb
├── experiments/
│   └── experiment_log.md  # Running log of experiments
├── reports/
│   ├── figures/           # Exported plots/PNGs
│   └── latex/             # LaTeX report files
└── figures/               # General figures

Include these in requirements.txt:
torch, torchvision, opencv-python, numpy, pandas, matplotlib, seaborn,
scikit-learn, scikit-image, pycocotools, Pillow, tqdm, onnxruntime

Create a proper README.md with: problem statement, dataset info placeholder,
methods overview, how to install & run, and hardware requirements.

Create a .gitignore for Python, Jupyter, data files (*.jpg, *.png in data/),
model weights (*.pt, *.onnx), and IDE files.
```

### Prompt 0.2: Initialize git and experiment log
```
Initialize a git repository in the Bell_Labs/ directory.
Create an initial commit with message "chore: project scaffold and environment setup".

Also create experiments/experiment_log.md with this template:

# Experiment Log

| Date | Experiment | Config | Result | Notes |
|------|-----------|--------|--------|-------|

## Log Entries

### [Date] - Experiment Name
- **Goal:**
- **Setup:**
- **Result:**
- **Conclusion:**
```

---

## TASK 1 — Literature Review & LaTeX Skeleton

### Prompt 1.1: Generate BibTeX entries
```
Generate BibTeX entries for these papers. Use the exact citation keys shown:

1. @article{bodla2017softnms — "Soft-NMS -- Improving Object Detection With One Line of Code", Bodla et al., ICCV 2017, arXiv:1704.04503
2. @inproceedings{bolya2019yolact — "YOLACT: Real-time Instance Segmentation", Bolya et al., ICCV 2019
3. @article{liu2019adaptivenms — "Adaptive NMS: Refining Pedestrian Detection in a Crowd", Liu et al., CVPR 2019
4. @inproceedings{goldman2019sku110k — "Precise Detection in Densely Packed Scenes", Goldman et al., CVPR 2019 (SKU-110K dataset)
5. @article{he2017maskrcnn — "Mask R-CNN", He et al., ICCV 2017

Save to reports/latex/references.bib
```

### Prompt 1.2: Create LaTeX report skeleton
```
Create a LaTeX report skeleton at reports/latex/main.tex using the
IEEEtran document class. Include these sections with placeholder text:

1. Introduction & Problem Statement
   - Dense object detection challenge
   - 1-50 heavily overlapping objects (retail shelves, crowds)
   - Why standard NMS fails

2. Related Work (subsections):
   2.1 Hard NMS and Its Limitations
   2.2 Soft-NMS and Adaptive NMS Variants
   2.3 Real-Time Instance Segmentation (YOLACT, Mask R-CNN)
   2.4 Dense Scene Datasets (SKU-110K)

3. Dataset & Exploratory Data Analysis
4. Methodology
   4.1 Baseline Heuristic
   4.2 Advanced Classical CV
   4.3 Deep Learning Model
   4.4 Hybrid Edge Pipeline
5. Experiments & Results
6. Discussion
7. Conclusion & Future Work

Add \bibliography{references} at the end.
Include placeholder \cite commands for all 5 papers in the Related Work section.
```

### Prompt 1.3: Draft literature review paragraphs
```
For each of these 4 clusters, write ONE paragraph (5-7 sentences each) that I
will then rewrite in my own words. For each cluster, follow this structure:
(a) What the key paper proposes
(b) What assumption it makes
(c) Why that assumption breaks for 1-50 heavily overlapping objects
(d) What numeric gap it leaves

Cluster 1 — Hard NMS:
Standard greedy NMS uses a fixed IoU threshold to suppress overlapping
detections. Cite: general detection pipelines.

Cluster 2 — Soft-NMS & Adaptive NMS:
Bodla et al. 2017 (Soft-NMS) decays scores instead of hard suppression.
Liu et al. 2019 (Adaptive-NMS) adjusts thresholds per instance.
Show the Gaussian decay equation: s_i = s_i * exp(-iou^2 / sigma).

Cluster 3 — Real-Time Instance Segmentation:
YOLACT (Bolya et al. 2019) generates prototype masks + per-instance
coefficients for real-time instance segmentation.

Cluster 4 — Dense Scene Datasets:
SKU-110K (Goldman et al. 2019) — 11,762 images of retail shelves,
average ~150 objects per image, designed for dense detection benchmarks.

End with a concrete gap sentence connecting to our project.

Output as LaTeX-ready text with \cite{} commands.
```

---

## TASK 2 — Dataset & EDA

### Prompt 2.1: Download and prepare SKU-110K subset
```
Write a Python script at src/data_loader.py that:

1. Downloads the SKU-110K dataset annotations (CSV format) from the official
   source or loads them from data/raw/ if already present.
2. Parses the annotation CSV: columns are image_name, x1, y1, x2, y2,
   class, image_width, image_height.
3. Filters images to those containing 1-50 annotated objects (our project scope).
4. Creates a train/val/test split (70/15/15) ensuring no duplicate shelf
   configurations across splits (split by unique image).
5. Saves split metadata as JSON files in data/processed/:
   - train_split.json, val_split.json, test_split.json
   Each entry: {"image": "name.jpg", "num_objects": N, "annotations": [...]}
6. Prints summary statistics: total images per split, object count stats.

Note: For now, if the actual images aren't available, the script should work
with just the annotations CSV and handle missing images gracefully.
```

### Prompt 2.2: Generate synthetic dataset
```
Write a Python script at src/synthetic_generator.py that creates a controlled
synthetic dataset of overlapping geometric shapes for experiments:

1. Generate 500 images (256x256 px, white background) with:
   - Random number of objects: 1-50 per image (uniform distribution)
   - Shapes: circles, rectangles, ellipses (random colors, sizes 15-60px)
   - Controlled overlap: parameter to set occlusion level (0%, 25%, 50%, 75%)
   - Each object gets a unique instance mask (saved as separate PNG or
     combined as a multi-channel mask)
2. Save images to data/synthetic/images/
3. Save annotations in COCO format JSON to data/synthetic/annotations.json
4. Save instance masks to data/synthetic/masks/
5. Generate a metadata CSV with: image_id, num_objects, avg_occlusion_ratio

This gives us a controlled testbed where we know ground truth perfectly.
```

### Prompt 2.3: EDA analysis and plots
```
Write a Python script at src/eda.py and a notebook at notebooks/01_eda.ipynb
that produces these analyses and saves plots to reports/figures/:

1. **Object Count Distribution** (histogram)
   - Plot histogram of objects per image for the filtered SKU-110K subset
   - Use bins of width 5, add mean/median lines
   - Save as: reports/figures/object_count_distribution.png

2. **Object Size Distribution** (log-scale histogram)
   - Plot distribution of bounding box areas (width * height) on log scale
   - Overlay aspect ratio distribution as secondary plot
   - Save as: reports/figures/object_size_distribution.png

3. **Occlusion Analysis** (IoU heatmap)
   - For each image, compute pairwise IoU between all bounding boxes
   - Plot: average IoU with nearest neighbor vs number of objects (scatter)
   - Save as: reports/figures/occlusion_analysis.png

4. **Sample Visualizations** (2x3 grid)
   - Show 3 "easy" images (few objects, low overlap) and 3 "hard" images
     (many objects, high overlap) side by side with bounding boxes drawn
   - Save as: reports/figures/sample_easy_vs_hard.png

5. **Summary Statistics Table**
   - Create a table with: split, num_images, mean/median/max objects,
     mean box area, mean nearest-neighbor IoU
   - Save as CSV and as a LaTeX table snippet

For synthetic data, generate the same plots. Each plot function should accept
a dataset parameter to work with either real or synthetic data.
```

---

## TASK 3 — Baseline Heuristic (Non-ML)

### Prompt 3.1: Implement blob/contour baseline
```
Write a Python module at src/baseline/heuristic.py that implements a simple
image-processing baseline for object separation:

Class: HeuristicDetector
Methods:
  - detect(image_path) -> list of bounding boxes [(x1,y1,x2,y2), ...]
  - detect_with_masks(image_path) -> boxes + binary masks

Pipeline:
1. Load image, convert to grayscale
2. Apply adaptive thresholding (Gaussian, block_size=11, C=2)
3. Morphological operations: opening (3x3 kernel) then closing (5x5 kernel)
4. Find contours using cv2.findContours with RETR_EXTERNAL
5. Filter contours by area (min_area=100, max_area=image_area*0.5)
6. For large merged blobs (area > threshold), apply watershed algorithm:
   a. Distance transform
   b. Threshold distance map
   c. Find markers
   d. Apply cv2.watershed
   e. Extract sub-contours
7. Return bounding boxes from final contours

Also write an evaluation function that compares predicted boxes against
ground truth using: count error (|predicted - true|), precision, recall,
and mean IoU of matched boxes (Hungarian matching).

Include a __main__ block that runs on 10 sample images and prints results.
```

### Prompt 3.2: Evaluate and document baseline failures
```
Write a notebook at notebooks/02_baseline.ipynb that:

1. Runs HeuristicDetector on the synthetic dataset at all 4 occlusion levels
   (0%, 25%, 50%, 75%) — 20 images per level.
2. Computes per-level metrics: count MAE, precision@IoU=0.5, recall@IoU=0.5
3. Creates these plots (save to reports/figures/):
   - Bar chart: count MAE vs occlusion level
   - Bar chart: precision and recall vs occlusion level
   - 2x2 grid: example detections at each occlusion level showing predicted
     boxes (red) overlaid on ground truth boxes (green)
4. Creates a summary table of all metrics
5. Includes markdown cells with analysis:
   - Why does the heuristic fail at high occlusion? (merged blobs)
   - What is the count error trend? (undercounting at high overlap)
   - Sensitivity to lighting/contrast variations
   - Conclusion: "necessary baseline but insufficient for dense scenes"

Save the summary table as reports/figures/baseline_results.csv
```

---

## TASK 4 — Advanced Classical CV (Non-DL)

### Prompt 4.1: Implement watershed + graph segmentation
```
Write a Python module at src/baseline/classical_cv.py with TWO methods:

Class: WatershedSegmenter
- Uses distance-transform-based watershed on preprocessed images
- Steps: grayscale → blur → Otsu threshold → distance transform →
  peak_local_max for markers → watershed → extract regions
- Returns bounding boxes and instance masks

Class: GraphSegmenter
- Uses Felzenszwalb graph-based segmentation (skimage.segmentation.felzenszwalb)
- Parameters: scale=200, sigma=0.5, min_size=50
- Post-process: merge tiny regions, filter by area
- Returns bounding boxes and instance masks

Both classes should implement:
  - detect(image) -> list of boxes
  - detect_with_masks(image) -> boxes + masks
  - a tunable parameter interface (dict of params)

Also add a RetailPriorDetector class that adds domain-specific priors:
- Detect horizontal shelf lines using HoughLinesP
- Use line positions to constrain search regions
- Apply peak detection along vertical strips for grid-like layouts
- This should improve results for structured retail images
```

### Prompt 4.2: Evaluate classical CV methods
```
Write a notebook at notebooks/03_classical_cv.ipynb that:

1. Runs all 3 classical methods (Watershed, GraphSeg, RetailPrior) on:
   - Synthetic dataset at 4 occlusion levels (20 images each)
   - Real SKU-110K subset if available (or synthetic "retail-like" images)

2. Computes: count MAE, precision@0.5, recall@0.5, mean IoU, processing time

3. Creates comparison plots (save to reports/figures/):
   - Grouped bar chart comparing all methods + heuristic baseline across
     occlusion levels for count MAE
   - Same for precision and recall
   - Processing time comparison bar chart
   - Qualitative 3x4 grid: rows = methods, columns = occlusion levels

4. Creates a LaTeX-ready comparison table

5. Analysis markdown cells:
   - Which method works best at low vs high occlusion?
   - Where do all classical methods break down?
   - How does the retail prior help (or not)?
   - Conclusion: "classical CV improves on heuristic, especially for
     structured scenes, but cannot handle extreme occlusion — motivating DL"
```

---

## TASK 5 — Deep Learning Model (Soft-NMS + YOLACT)

### Prompt 5.1: Implement Soft-NMS module
```
Write a PyTorch-compatible module at src/models/soft_nms.py that implements
Soft-NMS with both decay modes:

Function: soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001,
                    method='gaussian')

Parameters:
  - boxes: Tensor (N, 4) in [x1, y1, x2, y2] format
  - scores: Tensor (N,)
  - sigma: float, controls Gaussian decay width
  - score_threshold: float, minimum score to keep
  - method: 'gaussian' | 'linear' | 'hard'

Returns:
  - keep_indices: indices of kept boxes
  - new_scores: decayed scores

Implementation:
  - For each box in descending score order:
    - Compute IoU with all remaining boxes
    - Gaussian: s_i = s_i * exp(-iou^2 / sigma)
    - Linear: s_i = s_i * (1 - iou) if iou >= threshold, else s_i
    - Hard: s_i = 0 if iou >= threshold (standard NMS)
  - Remove boxes below score_threshold

Include:
  - Full docstrings with mathematical formulas in comments
  - A comparison function that runs all 3 methods on the same input
    and returns a dict of results
  - Unit tests at the bottom: test with known overlapping boxes
  - Complexity analysis in comments: O(N^2) same as hard NMS
```

### Prompt 5.2: Detection model wrapper
```
Write a Python module at src/models/detector.py that wraps a pretrained
object detector and swaps in Soft-NMS:

Class: DenseObjectDetector
  __init__(self, backbone='mobilenet_v3', nms_method='soft_gaussian',
           sigma=0.5, score_thresh=0.3, device='cpu')

  Methods:
  - load_model(): Load a pretrained Faster R-CNN or FCOS from torchvision
    with the specified backbone (resnet50 or mobilenet_v3_large)
  - detect(image_tensor) -> boxes, scores, labels:
    Run inference, then apply our Soft-NMS instead of the model's built-in NMS
  - detect_batch(image_list) -> list of (boxes, scores, labels)
  - benchmark(image_tensor, n_runs=10) -> dict with avg time, FPS, memory

The key innovation: intercept the raw proposals BEFORE the model's NMS
and apply our soft_nms function instead. For torchvision models, this
means modifying the postprocessing or using the model in a mode that
returns pre-NMS detections.

Include a CLI interface:
  python -m src.models.detector --image path/to/img.jpg --nms soft_gaussian
  --sigma 0.5 --visualize
```

### Prompt 5.3: Architecture diagram and plan document
```
Create a markdown document at reports/architecture_plan.md that describes
the full DL pipeline with these sections:

1. **Architecture Overview**
   - ASCII or text diagram of the full pipeline:
     Input Image → Backbone (MobileNetV3) → FPN → Detection Head →
     Raw Proposals → Soft-NMS → Final Detections
   - If using YOLACT-style: add Protonet → Prototype Masks →
     Mask Assembly with coefficients

2. **Soft-NMS Integration Point**
   - Where exactly Soft-NMS replaces standard NMS
   - Pseudocode for the modified post-processing
   - Gaussian decay equation with explanation of sigma parameter

3. **Hybrid Edge Pipeline**
   - Diagram: Edge Device (ONNX Runtime + MobileNetV3-Small + Soft-NMS)
     → Compressed ROIs → Server (Full model refinement)
   - Quantization strategy: INT8 for edge, FP32 for server
   - Expected latency breakdown

4. **Training Plan (Phase 2)**
   - Dataset: filtered SKU-110K (1-50 objects)
   - Loss: classification + box regression + (optional mask loss)
   - Optimizer: SGD with momentum, lr=0.01, cosine annealing
   - Epochs: 24, batch_size: 8
   - Augmentation: random horizontal flip, color jitter, random crop

5. **Evaluation Plan**
   - Metrics: mAP@0.5, mAP@0.5:0.95, count MAE, FPS
   - Ablations: Hard NMS vs Soft-NMS (Gaussian) vs Soft-NMS (Linear)
   - Sigma sweep: 0.1, 0.3, 0.5, 0.7, 1.0
   - Compare: ResNet50 vs MobileNetV3 backbone

6. **Complexity Analysis**
   - Soft-NMS: O(N^2) same as Hard NMS, extra exp() per comparison
   - FPN: O(C * H * W) per level
   - Total inference: breakdown by component
```

---

## TASK 6 — Evaluation Framework & Metrics

### Prompt 6.1: Implement evaluation metrics
```
Write a Python module at src/evaluation/metrics.py with these functions:

1. compute_iou(box_a, box_b) -> float
   - Standard IoU for two boxes [x1,y1,x2,y2]

2. match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
   -> (matches, unmatched_preds, unmatched_gts)
   - Hungarian matching to find optimal assignment

3. compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5) -> float
   - Average Precision at given IoU threshold

4. compute_map(predictions, ground_truths, iou_thresholds=[0.5]) -> dict
   - Mean AP across images, supports multiple thresholds
   - Returns: {'mAP@0.5': X, 'mAP@0.5:0.95': Y}

5. count_mae(pred_counts, gt_counts) -> float
   - Mean Absolute Error of object counts

6. compute_fps(model, images, device='cpu', n_warmup=3, n_runs=10) -> float
   - Frames per second benchmark

7. full_evaluation(predictions, ground_truths) -> dict
   - Runs all metrics and returns a comprehensive results dict

Include comprehensive docstrings and unit tests.
```

---

## TASK 7 — Soft-NMS Theoretical Analysis

### Prompt 7.1: Mathematical writeup
```
Create a LaTeX section at reports/latex/soft_nms_theory.tex that contains:

1. **Standard NMS formulation:**
   - Given detections D = {(b_i, s_i)}, IoU threshold N_t
   - Rescoring function: s_i = 0 if IoU(M, b_i) >= N_t
   - Explain why this is greedy and discards valid overlapping detections

2. **Soft-NMS formulation (Bodla et al. 2017):**
   - Linear decay: s_i = s_i(1 - IoU(M, b_i)) if IoU >= N_t
   - Gaussian decay: s_i = s_i * exp(-IoU(M, b_i)^2 / sigma)
   - Show both as special cases of a general rescoring function f(IoU)

3. **Analysis for dense scenes:**
   - Why Gaussian decay is preferred (continuous, no threshold discontinuity)
   - Effect of sigma: small sigma ≈ hard NMS, large sigma ≈ keep everything
   - Expected behavior with 1-50 objects: more detections retained,
     but risk of false positives at very high sigma

4. **Computational complexity:**
   - Both NMS and Soft-NMS are O(N^2) where N = number of proposals
   - Soft-NMS adds O(1) extra work per comparison (exp or multiply)
   - Practical impact on inference time: negligible for N < 1000

Use proper LaTeX: align environments for equations, theorem/proposition
style for key results, and a small comparison table.
```

---

## TASK 8 — Edge Pipeline Prototype

### Prompt 8.1: ONNX export and edge inference
```
Write a Python module at src/models/edge_pipeline.py that implements:

Class: EdgeInferencePipeline
  __init__(self, model_path='model.onnx', nms_method='soft_gaussian',
           input_size=(320, 320))

  Methods:
  - export_to_onnx(pytorch_model, output_path, input_size):
    Export a torchvision detection model to ONNX format
    Apply dynamic axes for batch dimension

  - preprocess(image) -> numpy array:
    Resize, normalize, convert to NCHW format

  - run_inference(image) -> boxes, scores:
    Load ONNX model with onnxruntime
    Run inference + Soft-NMS postprocessing

  - compress_results(boxes, scores, threshold=0.5) -> dict:
    Create a compact JSON payload: object count, top-K ROI coordinates,
    confidence summary — suitable for sending to server

  - benchmark(image, n_runs=50) -> dict:
    Report: avg latency (ms), FPS, peak memory (MB)

Class: ServerRefinementPipeline
  - receive_rois(compressed_data) -> refined_results:
    Placeholder for server-side refinement using full model
  - Accepts ROIs from edge, runs higher-res inference on crops

Include a demo script that:
1. Exports a MobileNetV3 model to ONNX
2. Runs edge inference on a test image
3. Compresses and prints the results payload
4. Benchmarks edge vs full-model inference time
```

---

## TASK 9 — Run All Experiments

### Prompt 9.1: Baseline experiment runner
```
Write a script at experiments/run_baselines.py that:

1. Loads the synthetic dataset (all occlusion levels)
2. Runs ALL methods in sequence:
   - HeuristicDetector
   - WatershedSegmenter
   - GraphSegmenter
   - RetailPriorDetector (if applicable)
3. Computes full_evaluation() for each method at each occlusion level
4. Saves results to experiments/baseline_results.json
5. Generates a comparison table and saves to reports/figures/
6. Appends a formatted entry to experiments/experiment_log.md
7. Prints a summary to console

The script should be idempotent and timestamped so we can re-run it
after parameter changes and track improvements.
```

---

## TASK 10 — Final Report Assembly & README

### Prompt 10.1: Compile publication-quality figures
```
Write a Python script at reports/compile_figures.py that:

1. Reads all result files from experiments/
2. Generates publication-quality figures using matplotlib with:
   - Font size 12, figure size (8, 5), consistent color palette
3. Generates these specific figures:
   - Figure 1: Object count distribution (EDA)
   - Figure 2: Size and aspect ratio distributions (EDA)
   - Figure 3: Occlusion analysis scatter plot (EDA)
   - Figure 4: Baseline comparison bar charts (count MAE, precision, recall)
   - Figure 5: Qualitative detection examples (3x4 grid)
   - Figure 6: NMS vs Soft-NMS comparison on overlapping boxes
   - Figure 7: Architecture diagram (text-based or matplotlib)
4. Saves all figures as both PNG (300 DPI) and PDF to reports/figures/
5. Generates LaTeX \includegraphics commands for each figure
```

### Prompt 10.2: Final README
```
Update the README.md with full project documentation including:
- Problem statement (1-50 overlapping objects, Soft-NMS innovation)
- Methods summary (4 levels: heuristic → classical → DL → edge)
- Dataset description (SKU-110K subset + synthetic)
- Repository structure (tree output)
- Setup instructions (pip install, data generation, run experiments)
- Preliminary results table
- Hardware requirements
- References (5 key papers)
```

---

## 📋 Execution Checklist

| # | Task | Status | Commit Message |
|---|------|--------|----------------|
| 0 | Project scaffold | ⬜ | `chore: project scaffold` |
| 1 | Literature review + LaTeX | ⬜ | `docs: literature review and LaTeX skeleton` |
| 2 | Dataset + EDA | ⬜ | `feat: dataset loading, synthetic gen, and EDA` |
| 3 | Baseline heuristic | ⬜ | `feat: heuristic baseline detector` |
| 4 | Classical CV methods | ⬜ | `feat: watershed and graph segmentation` |
| 5 | DL model + Soft-NMS | ⬜ | `feat: Soft-NMS implementation and detector` |
| 6 | Evaluation metrics | ⬜ | `feat: evaluation framework` |
| 7 | Theory writeup | ⬜ | `docs: Soft-NMS theoretical analysis` |
| 8 | Edge pipeline | ⬜ | `feat: ONNX edge inference pipeline` |
| 9 | Run experiments | ⬜ | `exp: baseline experiment results` |
| 10 | Report + README | ⬜ | `docs: final report assembly` |

---

## ⚠️ Safety Reminders

1. **After every prompt output**: Read the code, check formulas, run it, fix errors
2. **Rewrite text outputs**: Never submit AI-generated text as-is in your report
3. **Commit frequently**: Each task = at least one meaningful commit
4. **Experiment log**: After every experiment, add an entry to `experiment_log.md`
5. **Own your design choices**: In viva, you must explain WHY you chose each method
6. **Check citations**: Verify every BibTeX entry against the actual paper
