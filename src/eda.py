"""
eda.py — Exploratory Data Analysis
===================================
Generates statistical analyses and visualizations for:
  - Object count distributions
  - Object size and aspect ratio distributions
  - Occlusion analysis (pairwise IoU)
  - Sample visualizations (easy vs hard images)
  - Summary statistics tables

All plots are saved to reports/figures/.
Each function accepts a ``dataset`` parameter ("sku110k" or "synthetic")
so the same analyses can be run on either dataset.

Usage:
    python -m src.eda                            # run all analyses
    python -m src.eda --dataset synthetic         # synthetic only
    python -m src.eda --dataset sku110k           # SKU-110K only
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# ---- Local imports ----
from src.utils.visualization import (COLORS, FIGURE_DIR, save_figure,
                                     setup_style)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Apply consistent plotting style
setup_style()


# ====================================================================
# Data Loading Helpers
# ====================================================================


def _load_split_json(split_name: str) -> List[Dict]:
    """Load a single split JSON file from data/processed/."""
    path = PROCESSED_DIR / f"{split_name}_split.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_all_splits() -> Dict[str, List[Dict]]:
    """Load train/val/test splits; returns empty dict entries for missing."""
    return {name: _load_split_json(name) for name in ["train", "val", "test"]}


def _load_synthetic_metadata() -> Optional[pd.DataFrame]:
    """Load synthetic metadata CSV."""
    path = SYNTHETIC_DIR / "metadata.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_synthetic_annotations() -> Optional[Dict]:
    """Load synthetic COCO annotations JSON."""
    path = SYNTHETIC_DIR / "annotations.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _annotations_to_dataframe(entries: List[Dict]) -> pd.DataFrame:
    """
    Convert split JSON entries to a flat DataFrame of annotations.

    Each row has: image_name, num_objects, x1, y1, x2, y2, box_w, box_h, area.
    """
    rows = []
    for entry in entries:
        img = entry["image"]
        n = entry["num_objects"]
        for ann in entry["annotations"]:
            x1, y1, x2, y2 = ann
            rows.append(
                {
                    "image_name": img,
                    "num_objects": n,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_w": x2 - x1,
                    "box_h": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1),
                }
            )
    return pd.DataFrame(rows)


def _coco_to_dataframe(coco: Dict) -> pd.DataFrame:
    """Convert COCO-format annotations dict to a flat DataFrame."""
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Count objects per image
    from collections import Counter

    obj_counts = Counter(a["image_id"] for a in coco["annotations"])

    rows = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        rows.append(
            {
                "image_name": img_map[ann["image_id"]],
                "image_id": ann["image_id"],
                "num_objects": obj_counts[ann["image_id"]],
                "category": cat_map.get(ann["category_id"], "unknown"),
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "box_w": w,
                "box_h": h,
                "area": ann.get("area", w * h),
            }
        )
    return pd.DataFrame(rows)


# ====================================================================
# IoU Computation
# ====================================================================


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_pairwise_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute NxN pairwise IoU matrix.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)
        Bounding boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    n = len(boxes)
    iou_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            iou_val = compute_iou(boxes[i], boxes[j])
            iou_mat[i, j] = iou_val
            iou_mat[j, i] = iou_val
    return iou_mat


def nearest_neighbor_iou(boxes: np.ndarray) -> float:
    """Average IoU of each box with its nearest (highest-IoU) neighbor."""
    if len(boxes) < 2:
        return 0.0
    iou_mat = compute_pairwise_iou_matrix(boxes)
    np.fill_diagonal(iou_mat, 0)
    max_ious = iou_mat.max(axis=1)
    return float(np.mean(max_ious))


# ====================================================================
# Plot 1: Object Count Distribution
# ====================================================================


def plot_object_count_distribution(
    df: pd.DataFrame,
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Histogram of objects per image with mean/median lines.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: image_name, num_objects (or we group from annotations).
    dataset : str
        Label for the plot title.
    output_dir : Path, optional
        Directory to save the figure.
    """
    if output_dir is None:
        output_dir = FIGURE_DIR

    # Get per-image object counts
    if "num_objects" in df.columns:
        counts = df.groupby("image_name")["num_objects"].first()
    else:
        counts = df.groupby("image_name").size()

    mean_val = counts.mean()
    median_val = counts.median()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with bins of width 5
    bins = np.arange(0, counts.max() + 6, 5)
    ax.hist(
        counts,
        bins=bins,
        color=COLORS["primary"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.8,
        label="Images",
    )

    # Mean & median lines
    ax.axvline(
        mean_val,
        color=COLORS["danger"],
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.1f}",
    )
    ax.axvline(
        median_val,
        color=COLORS["warning"],
        linestyle="-.",
        linewidth=2,
        label=f"Median = {median_val:.1f}",
    )

    dataset_label = (
        "SKU-110K (1–50 subset)" if dataset == "sku110k" else "Synthetic Shapes"
    )
    ax.set_title(
        f"Object Count Distribution — {dataset_label}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Number of Images")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(0, None)

    sns.despine()
    fig.tight_layout()

    prefix = "synth_" if dataset == "synthetic" else ""
    save_figure(fig, f"{prefix}object_count_distribution.png", output_dir)
    return fig


# ====================================================================
# Plot 2: Object Size Distribution
# ====================================================================


def plot_object_size_distribution(
    df: pd.DataFrame,
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Log-scale histogram of bounding-box areas with aspect-ratio overlay.
    """
    if output_dir is None:
        output_dir = FIGURE_DIR

    areas = df["area"].dropna()
    areas = areas[areas > 0]

    # Aspect ratio (width / height)
    aspect = (df["box_w"] / df["box_h"]).dropna()
    aspect = aspect[(aspect > 0) & (aspect < 20)]  # filter outliers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Box area (log scale) ----
    log_areas = np.log10(areas.clip(lower=1))
    ax1.hist(
        log_areas,
        bins=50,
        color=COLORS["primary"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.axvline(
        np.log10(areas.median()),
        color=COLORS["danger"],
        ls="--",
        lw=2,
        label=f"Median = {areas.median():.0f} px²",
    )
    ax1.set_xlabel("Bounding Box Area (log₁₀ px²)")
    ax1.set_ylabel("Count")
    ax1.set_title("Box Area Distribution (Log Scale)", fontweight="bold")
    ax1.legend(fontsize=10)

    # ---- Aspect ratio ----
    ax2.hist(
        aspect,
        bins=50,
        color=COLORS["secondary"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axvline(
        aspect.median(),
        color=COLORS["danger"],
        ls="--",
        lw=2,
        label=f"Median = {aspect.median():.2f}",
    )
    ax2.set_xlabel("Aspect Ratio (width / height)")
    ax2.set_ylabel("Count")
    ax2.set_title("Aspect Ratio Distribution", fontweight="bold")
    ax2.legend(fontsize=10)

    dataset_label = "SKU-110K" if dataset == "sku110k" else "Synthetic"
    fig.suptitle(
        f"Object Size Distribution — {dataset_label}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    sns.despine()
    fig.tight_layout()

    prefix = "synth_" if dataset == "synthetic" else ""
    save_figure(fig, f"{prefix}object_size_distribution.png", output_dir)
    return fig


# ====================================================================
# Plot 3: Occlusion Analysis
# ====================================================================


def plot_occlusion_analysis(
    df: pd.DataFrame,
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None,
    max_images: int = 500,
) -> plt.Figure:
    """
    Scatter plot: avg nearest-neighbor IoU vs number of objects per image.
    """
    if output_dir is None:
        output_dir = FIGURE_DIR

    image_names = df["image_name"].unique()
    if len(image_names) > max_images:
        rng = np.random.RandomState(42)
        image_names = rng.choice(image_names, max_images, replace=False)

    results = []
    for img_name in image_names:
        img_df = df[df["image_name"] == img_name]
        boxes = img_df[["x1", "y1", "x2", "y2"]].values.astype(float)
        n_obj = len(boxes)
        nn_iou = nearest_neighbor_iou(boxes)
        results.append({"image": img_name, "num_objects": n_obj, "nn_iou": nn_iou})

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        res_df["num_objects"],
        res_df["nn_iou"],
        c=res_df["nn_iou"],
        cmap="YlOrRd",
        alpha=0.6,
        s=30,
        edgecolor="gray",
        linewidth=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Avg NN IoU", shrink=0.8)

    # Trend line
    if len(res_df) > 10:
        z = np.polyfit(res_df["num_objects"], res_df["nn_iou"], 2)
        p = np.poly1d(z)
        x_line = np.linspace(
            res_df["num_objects"].min(), res_df["num_objects"].max(), 100
        )
        ax.plot(
            x_line,
            p(x_line),
            color=COLORS["danger"],
            linewidth=2,
            linestyle="--",
            label="Quadratic trend",
        )

    dataset_label = "SKU-110K" if dataset == "sku110k" else "Synthetic"
    ax.set_title(
        f"Occlusion Analysis — {dataset_label}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Average Nearest-Neighbor IoU")
    ax.legend(fontsize=10)
    sns.despine()
    fig.tight_layout()

    prefix = "synth_" if dataset == "synthetic" else ""
    save_figure(fig, f"{prefix}occlusion_analysis.png", output_dir)
    return fig


# ====================================================================
# Plot 4: Sample Visualizations (Easy vs Hard)
# ====================================================================


def plot_easy_vs_hard_samples(
    df: pd.DataFrame,
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    2×3 grid showing 3 'easy' and 3 'hard' images with bounding boxes.

    Easy = few objects + low overlap; Hard = many objects + high overlap.
    If images are unavailable, generates synthetic placeholder canvases
    with colored rectangles for the bounding boxes.
    """
    if output_dir is None:
        output_dir = FIGURE_DIR

    if image_dir is None:
        if dataset == "synthetic":
            image_dir = SYNTHETIC_DIR / "images"
        else:
            image_dir = RAW_DIR

    # Score each image: more objects + higher NN-IoU = harder
    image_scores = []
    for img_name, group in df.groupby("image_name"):
        boxes = group[["x1", "y1", "x2", "y2"]].values.astype(float)
        n = len(boxes)
        nn_iou = nearest_neighbor_iou(boxes)
        score = n * 0.5 + nn_iou * 50  # weighted difficulty score
        image_scores.append(
            {
                "image": img_name,
                "score": score,
                "n_objects": n,
                "nn_iou": nn_iou,
            }
        )

    score_df = pd.DataFrame(image_scores).sort_values("score")
    easy_imgs = score_df.head(3)["image"].tolist()
    hard_imgs = score_df.tail(3)["image"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, img_name in enumerate(easy_imgs):
        _draw_image_with_boxes(
            axes[0, col], df, img_name, image_dir, title=f"Easy #{col + 1}"
        )

    for col, img_name in enumerate(hard_imgs):
        _draw_image_with_boxes(
            axes[1, col], df, img_name, image_dir, title=f"Hard #{col + 1}"
        )

    axes[0, 0].set_ylabel(
        "EASY\n(few objects, low overlap)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_ylabel(
        "HARD\n(many objects, high overlap)", fontsize=12, fontweight="bold"
    )

    dataset_label = "SKU-110K" if dataset == "sku110k" else "Synthetic"
    fig.suptitle(
        f"Sample Visualizations — {dataset_label}", fontsize=16, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    prefix = "synth_" if dataset == "synthetic" else ""
    save_figure(fig, f"{prefix}sample_easy_vs_hard.png", output_dir)
    return fig


def _draw_image_with_boxes(
    ax: plt.Axes,
    df: pd.DataFrame,
    image_name: str,
    image_dir: Path,
    title: str = "",
):
    """Draw an image with its bounding boxes on a matplotlib axis."""
    img_df = df[df["image_name"] == image_name]
    boxes = img_df[["x1", "y1", "x2", "y2"]].values.astype(float)
    n = len(boxes)
    nn_iou = nearest_neighbor_iou(boxes)

    # Try to load actual image
    img_path = image_dir / image_name
    if img_path.exists():
        import cv2

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    else:
        img = None

    if img is not None:
        ax.imshow(img)
    else:
        # Create a light-gray canvas as placeholder
        if len(boxes) > 0:
            max_x = int(boxes[:, 2].max()) + 20
            max_y = int(boxes[:, 3].max()) + 20
        else:
            max_x, max_y = 256, 256
        canvas = np.full((max(max_y, 50), max(max_x, 50), 3), 240, dtype=np.uint8)
        ax.imshow(canvas)

    # Draw bounding boxes
    cmap = plt.cm.hsv
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = cmap(i / max(n, 1))
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.5,
            edgecolor=color,
            facecolor=(*color[:3], 0.15),
        )
        ax.add_patch(rect)

    ax.set_title(f"{title}\nn={n}, NN-IoU={nn_iou:.3f}", fontsize=10)
    ax.axis("off")


# ====================================================================
# Plot 5: Summary Statistics Table
# ====================================================================


def compute_summary_statistics(
    df: pd.DataFrame,
    splits: Optional[Dict[str, List[Dict]]] = None,
    dataset: str = "sku110k",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute and save summary statistics table.

    Columns: split, num_images, mean_objects, median_objects, max_objects,
             mean_box_area, mean_nn_iou.

    Saves as CSV and LaTeX snippet.
    """
    if output_dir is None:
        output_dir = FIGURE_DIR

    rows = []

    if splits and any(len(v) > 0 for v in splits.values()):
        # Per-split stats (SKU-110K)
        for split_name, entries in splits.items():
            if not entries:
                continue
            split_df = _annotations_to_dataframe(entries)
            row = _compute_split_row(split_name, split_df)
            rows.append(row)
    else:
        # Overall stats (single dataset, e.g. synthetic)
        row = _compute_split_row("all", df)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Save CSV
    prefix = "synth_" if dataset == "synthetic" else ""
    csv_path = output_dir / f"{prefix}summary_statistics.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[eda] Saved summary CSV → {csv_path}")

    # Save LaTeX snippet
    latex_path = output_dir / f"{prefix}summary_statistics.tex"
    latex_str = summary.to_latex(
        index=False,
        float_format="%.2f",
        caption=(
            f"Summary statistics for {'Synthetic' if dataset == 'synthetic' else 'SKU-110K'} dataset."
        ),
        label=f"tab:{prefix}summary",
    )
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"[eda] Saved summary LaTeX → {latex_path}")

    return summary


def _compute_split_row(split_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Compute one row of the summary table for a given split."""
    counts = df.groupby("image_name").size()
    areas = df["area"]

    # Sample up to 200 images for NN-IoU (expensive)
    image_names = df["image_name"].unique()
    sample_size = min(200, len(image_names))
    rng = np.random.RandomState(42)
    sampled = rng.choice(image_names, sample_size, replace=False)

    nn_ious = []
    for img_name in sampled:
        boxes = df[df["image_name"] == img_name][["x1", "y1", "x2", "y2"]].values
        nn_ious.append(nearest_neighbor_iou(boxes.astype(float)))

    return {
        "split": split_name,
        "num_images": len(counts),
        "mean_objects": round(counts.mean(), 1),
        "median_objects": round(counts.median(), 1),
        "max_objects": int(counts.max()),
        "mean_box_area": round(areas.mean(), 1),
        "mean_nn_iou": round(np.mean(nn_ious), 4),
    }


# ====================================================================
# Top-Level Runner
# ====================================================================


def run_eda(dataset: str = "sku110k") -> None:
    """
    Run all EDA analyses for the specified dataset.

    Parameters
    ----------
    dataset : str
        "sku110k" or "synthetic".
    """
    print(f"\n{'=' * 60}")
    print(f"  EDA — {dataset.upper()}")
    print(f"{'=' * 60}\n")

    if dataset == "sku110k":
        splits = _load_all_splits()
        all_entries = []
        for entries in splits.values():
            all_entries.extend(entries)

        if not all_entries:
            print("[eda] ⚠ No SKU-110K split data found in data/processed/.")
            print("      Run `python -m src.data_loader` first.\n")
            return

        df = _annotations_to_dataframe(all_entries)
        print(
            f"[eda] Loaded {len(df):,} annotations across {df['image_name'].nunique():,} images\n"
        )

    elif dataset == "synthetic":
        coco = _load_synthetic_annotations()
        if coco is None:
            print("[eda] ⚠ No synthetic data found in data/synthetic/.")
            print("      Run `python -m src.synthetic_generator` first.\n")
            return

        df = _coco_to_dataframe(coco)
        splits = None
        print(
            f"[eda] Loaded {len(df):,} annotations across {df['image_name'].nunique():,} images\n"
        )

    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Use 'sku110k' or 'synthetic'.")

    # ---- Run analyses ----
    print("─" * 40)
    print("1/5  Object count distribution…")
    plot_object_count_distribution(df, dataset=dataset)
    plt.close("all")

    print("2/5  Object size distribution…")
    plot_object_size_distribution(df, dataset=dataset)
    plt.close("all")

    print("3/5  Occlusion analysis…")
    plot_occlusion_analysis(df, dataset=dataset)
    plt.close("all")

    print("4/5  Sample easy vs hard…")
    plot_easy_vs_hard_samples(df, dataset=dataset)
    plt.close("all")

    print("5/5  Summary statistics…")
    if dataset == "sku110k":
        summary = compute_summary_statistics(df, splits=splits, dataset=dataset)
    else:
        summary = compute_summary_statistics(df, dataset=dataset)
    print("\n" + summary.to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"  EDA complete — figures saved to {FIGURE_DIR}")
    print(f"{'=' * 60}\n")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA analyses")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["sku110k", "synthetic", "all"],
        help="Which dataset to analyze (default: all)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        run_eda("synthetic")
        run_eda("sku110k")
    else:
        run_eda(args.dataset)
