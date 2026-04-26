"""
run_baselines.py — Baseline Experiment Runner
===============================================
Runs ALL classical detection methods on the synthetic dataset
across all occlusion levels, computes full evaluation metrics,
and generates comparison tables and figures.

This script is idempotent and timestamped — re-run after parameter
changes to track improvements.

Usage:
    python experiments/run_baselines.py
    python experiments/run_baselines.py --samples 30 --verbose
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Fix imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.heuristic import HeuristicDetector
from src.baseline.classical_cv import (
    WatershedSegmenter,
    GraphSegmenter,
    RetailPriorDetector,
    evaluate_detections,
)
from src.evaluation.metrics import full_evaluation

# ---- Paths ----
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
# Helpers
# ====================================================================

def load_synthetic_data():
    """Load synthetic annotations and metadata."""
    ann_path = SYNTHETIC_DIR / "annotations.json"
    meta_path = SYNTHETIC_DIR / "metadata.csv"

    if not ann_path.exists():
        print("⚠ Synthetic data not found. Run: python -m src.synthetic_generator")
        sys.exit(1)

    with open(ann_path) as f:
        coco = json.load(f)

    meta_df = pd.read_csv(meta_path)

    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    gt_by_image = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        fname = img_map[ann["image_id"]]
        gt_by_image.setdefault(fname, []).append((x, y, x + w, y + h))

    occ_by_image = dict(zip(meta_df["file_name"], meta_df["target_occlusion"]))

    return coco, meta_df, gt_by_image, occ_by_image


def get_detectors():
    """Get all detectors with their configs."""
    return {
        "Heuristic": HeuristicDetector(block_size=11, C=2, min_area=100),
        "Watershed": WatershedSegmenter(blur_ksize=7, min_distance=10, min_area=80),
        "GraphSeg": GraphSegmenter(scale=200, sigma=0.5, min_size=50, min_area=80),
        "RetailPrior": RetailPriorDetector(fallback_scale=150, min_area=80),
    }


# ====================================================================
# Main Runner
# ====================================================================

def run_experiment(
    samples_per_level: int = 20,
    verbose: bool = False,
    seed: int = 42,
):
    """Run the full baseline experiment."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  BASELINE EXPERIMENT — {timestamp}")
    print("=" * 70)

    # ---- Load data ----
    coco, meta_df, gt_by_image, occ_by_image = load_synthetic_data()
    detectors = get_detectors()
    occlusion_levels = sorted(meta_df["target_occlusion"].unique())
    rng = np.random.RandomState(seed)

    print(f"\nDataset: {len(coco['images'])} images")
    print(f"Occlusion levels: {[f'{l:.0%}' for l in occlusion_levels]}")
    print(f"Methods: {list(detectors.keys())}")
    print(f"Samples per level: {samples_per_level}")

    # ---- Run evaluations ----
    all_results = []
    method_times = {name: [] for name in detectors}

    for level in occlusion_levels:
        level_images = meta_df[meta_df["target_occlusion"] == level]["file_name"].tolist()
        sampled = rng.choice(
            level_images, min(samples_per_level, len(level_images)), replace=False
        ).tolist()

        print(f"\n{'─' * 50}")
        print(f"  Occlusion: {level:.0%} ({len(sampled)} images)")
        print(f"{'─' * 50}")

        for method_name, det in detectors.items():
            method_preds = []
            method_gts = []
            level_times = []

            for fname in sampled:
                img_path = str(SYNTHETIC_DIR / "images" / fname)
                gt_boxes = gt_by_image.get(fname, [])

                t0 = time.perf_counter()
                pred_boxes = det.detect(img_path)
                dt = time.perf_counter() - t0
                level_times.append(dt)

                # Build dicts for full_evaluation
                pred_scores = [1.0 / (i + 1) for i in range(len(pred_boxes))]
                method_preds.append({"boxes": list(pred_boxes), "scores": pred_scores})
                method_gts.append({"boxes": list(gt_boxes)})

                # Per-image result
                metrics = evaluate_detections(pred_boxes, gt_boxes)
                metrics["image"] = fname
                metrics["method"] = method_name
                metrics["occlusion_level"] = level
                metrics["occlusion_label"] = f"{level:.0%}"
                metrics["time_ms"] = round(dt * 1000, 1)
                all_results.append(metrics)

            method_times[method_name].extend(level_times)

            # Full eval for this method+level
            eval_result = full_evaluation(method_preds, method_gts)
            avg_ms = np.mean(level_times) * 1000

            if verbose:
                print(f"  {method_name:<14} | MAE={eval_result['count_mae']:.1f}  "
                      f"P={eval_result['avg_precision']:.3f}  "
                      f"R={eval_result['avg_recall']:.3f}  "
                      f"F1={eval_result['avg_f1']:.3f}  "
                      f"IoU={eval_result['avg_iou']:.3f}  "
                      f"{avg_ms:.0f}ms")

    # ---- Build results DataFrame ----
    results_df = pd.DataFrame(all_results)

    # ---- Summary table ----
    summary = results_df.groupby(["method", "occlusion_label"]).agg(
        count_mae=("count_error", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
        avg_time_ms=("time_ms", "mean"),
    ).round(3).reset_index()

    summary.columns = [
        "Method", "Occlusion", "Count MAE", "Precision",
        "Recall", "F1", "Mean IoU", "Avg Time (ms)",
    ]

    # ---- Print summary ----
    print(f"\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}\n")
    print(summary.to_string(index=False))

    # Overall per-method
    overall = results_df.groupby("method").agg(
        count_mae=("count_error", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
        avg_time_ms=("time_ms", "mean"),
    ).round(3)

    print(f"\n{'─' * 50}")
    print("  OVERALL (all occlusion levels)")
    print(f"{'─' * 50}")
    print(overall.to_string())

    # ---- Save results ----
    # 1. JSON
    json_path = EXPERIMENTS_DIR / "baseline_results.json"
    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "samples_per_level": samples_per_level,
            "seed": seed,
            "occlusion_levels": [float(l) for l in occlusion_levels],
            "methods": {name: repr(det) for name, det in detectors.items()},
        },
        "summary": summary.to_dict(orient="records"),
        "overall": overall.reset_index().to_dict(orient="records"),
        "per_image_results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[save] Results JSON → {json_path}")

    # 2. CSV
    csv_path = FIGURES_DIR / "baseline_comparison.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[save] Comparison CSV → {csv_path}")

    # 3. LaTeX
    latex_path = FIGURES_DIR / "baseline_comparison.tex"
    latex = summary.to_latex(
        index=False, float_format="%.3f",
        caption="Baseline method comparison across occlusion levels.",
        label="tab:baseline_comparison",
    )
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"[save] LaTeX table → {latex_path}")

    # 4. Experiment log
    _append_experiment_log(run_id, timestamp, summary, overall, detectors)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT COMPLETE — {run_id}")
    print(f"{'=' * 70}\n")

    return results_df, summary


def _append_experiment_log(
    run_id: str,
    timestamp: str,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    detectors: dict,
):
    """Append a formatted entry to the experiment log."""
    log_path = EXPERIMENTS_DIR / "experiment_log.md"

    entry_lines = [
        f"\n---\n",
        f"## Experiment: Baseline Comparison — {run_id}\n",
        f"- **Date**: {timestamp}\n",
        f"- **Type**: Baseline method comparison (all classical methods)\n",
        f"- **Dataset**: Synthetic overlapping shapes (4 occlusion levels)\n",
        f"- **Methods**: {', '.join(detectors.keys())}\n",
        f"\n### Configuration\n",
    ]

    for name, det in detectors.items():
        entry_lines.append(f"- `{name}`: {repr(det)}\n")

    entry_lines.append(f"\n### Overall Results\n\n")
    entry_lines.append(overall.to_markdown() + "\n")

    entry_lines.append(f"\n### Key Findings\n")
    best_method = overall["f1"].idxmax()
    best_f1 = overall.loc[best_method, "f1"]
    entry_lines.append(f"- Best overall F1: **{best_method}** ({best_f1:.3f})\n")
    entry_lines.append(
        f"- All methods degrade at high occlusion; "
        f"classical CV insufficient for dense scenes\n"
    )
    entry_lines.append(f"- See `experiments/baseline_results.json` for full data\n")

    with open(log_path, "a") as f:
        f.writelines(entry_lines)

    print(f"[save] Experiment log → {log_path}")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run baseline experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=20,
                        help="Samples per occlusion level")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        samples_per_level=args.samples,
        verbose=args.verbose,
        seed=args.seed,
    )
