"""
compile_figures.py — Publication-Quality Figure Generator
==========================================================
Reads experiment results and generates all figures for the final report.

Outputs both PNG (300 DPI) and PDF to reports/figures/.
Also generates LaTeX \\includegraphics commands.

Usage:
    python reports/compile_figures.py
    python reports/compile_figures.py --pdf-only
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

# ---- Paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

# ---- Style ----
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
METHOD_NAMES = ["Heuristic", "Watershed", "GraphSeg", "RetailPrior"]


def setup_style():
    """Apply publication-quality style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


setup_style()


# ====================================================================
# Save Helper
# ====================================================================

def save_fig(fig, name, formats=("png", "pdf")):
    """Save figure in PNG and PDF."""
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {name} (.png, .pdf)")


# ====================================================================
# Figure 1: Object Count Distribution
# ====================================================================

def figure_1_object_count_distribution():
    """Histogram of objects per image from synthetic data."""
    meta_path = SYNTHETIC_DIR / "metadata.csv"
    if not meta_path.exists():
        print("  ⚠ Skipping Fig 1: metadata.csv not found")
        return

    meta = pd.read_csv(meta_path)
    counts = meta["num_objects"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, counts.max() + 6, 5)
    ax.hist(counts, bins=bins, color=COLORS[0], alpha=0.8,
            edgecolor="white", linewidth=0.8)
    ax.axvline(counts.mean(), color="#F44336", ls="--", lw=2,
               label=f"Mean = {counts.mean():.1f}")
    ax.axvline(counts.median(), color="#FFC107", ls="-.", lw=2,
               label=f"Median = {counts.median():.1f}")
    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title("Figure 1: Object Count Distribution (Synthetic)", fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig1_object_count_distribution")
    plt.close(fig)


# ====================================================================
# Figure 2: Size & Aspect Ratio Distributions
# ====================================================================

def figure_2_size_distributions():
    """Box area (log) and aspect ratio distributions."""
    ann_path = SYNTHETIC_DIR / "annotations.json"
    if not ann_path.exists():
        print("  ⚠ Skipping Fig 2: annotations.json not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    areas, aspects = [], []
    for ann in coco["annotations"]:
        _, _, w, h = ann["bbox"]
        if w > 0 and h > 0:
            areas.append(w * h)
            aspects.append(w / h)

    areas = np.array(areas)
    aspects = np.array(aspects)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Area (log)
    ax1.hist(np.log10(np.clip(areas, 1, None)), bins=50,
             color=COLORS[0], alpha=0.8, edgecolor="white")
    ax1.axvline(np.log10(np.median(areas)), color="#F44336", ls="--", lw=2,
                label=f"Median = {np.median(areas):.0f} px²")
    ax1.set_xlabel("Bounding Box Area (log₁₀ px²)")
    ax1.set_ylabel("Count")
    ax1.set_title("Box Area Distribution", fontweight="bold")
    ax1.legend()

    # Aspect ratio
    ax2.hist(aspects[aspects < 5], bins=50, color=COLORS[1], alpha=0.8, edgecolor="white")
    ax2.axvline(np.median(aspects), color="#F44336", ls="--", lw=2,
                label=f"Median = {np.median(aspects):.2f}")
    ax2.set_xlabel("Aspect Ratio (width / height)")
    ax2.set_ylabel("Count")
    ax2.set_title("Aspect Ratio Distribution", fontweight="bold")
    ax2.legend()

    fig.suptitle("Figure 2: Object Size & Aspect Ratio Distributions",
                 fontsize=15, fontweight="bold", y=1.02)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig2_size_distributions")
    plt.close(fig)


# ====================================================================
# Figure 3: Occlusion Analysis Scatter
# ====================================================================

def figure_3_occlusion_scatter():
    """Average NN-IoU vs number of objects."""
    ann_path = SYNTHETIC_DIR / "annotations.json"
    if not ann_path.exists():
        print("  ⚠ Skipping Fig 3: annotations.json not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)

    from src.eda import nearest_neighbor_iou

    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    boxes_by_img = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        boxes_by_img.setdefault(ann["image_id"], []).append([x, y, x+w, y+h])

    rng = np.random.RandomState(42)
    img_ids = list(boxes_by_img.keys())
    if len(img_ids) > 300:
        img_ids = rng.choice(img_ids, 300, replace=False).tolist()

    n_objs, nn_ious = [], []
    for iid in img_ids:
        boxes = np.array(boxes_by_img[iid], dtype=float)
        n_objs.append(len(boxes))
        nn_ious.append(nearest_neighbor_iou(boxes))

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(n_objs, nn_ious, c=nn_ious, cmap="YlOrRd",
                    alpha=0.6, s=30, edgecolor="gray", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="Avg NN IoU", shrink=0.8)

    if len(n_objs) > 10:
        z = np.polyfit(n_objs, nn_ious, 2)
        p = np.poly1d(z)
        x_line = np.linspace(min(n_objs), max(n_objs), 100)
        ax.plot(x_line, p(x_line), color="#F44336", lw=2, ls="--", label="Trend")

    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Average Nearest-Neighbor IoU")
    ax.set_title("Figure 3: Occlusion Analysis", fontsize=14, fontweight="bold")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig3_occlusion_analysis")
    plt.close(fig)


# ====================================================================
# Figure 4: Baseline Comparison Bar Charts
# ====================================================================

def figure_4_baseline_comparison():
    """Count MAE, Precision, Recall grouped bars."""
    json_path = EXPERIMENTS_DIR / "baseline_results.json"
    csv_path = FIGURES_DIR / "baseline_comparison.csv"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        summary = pd.DataFrame(data["summary"])
    elif csv_path.exists():
        summary = pd.read_csv(csv_path)
    else:
        print("  ⚠ Skipping Fig 4: no baseline results found")
        return

    methods = summary["Method"].unique()
    occlusion_labels = sorted(summary["Occlusion"].unique())
    n_methods = len(methods)
    x = np.arange(len(occlusion_labels))
    bar_w = 0.8 / n_methods

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

    for metric, ax, title, ylabel in [
        ("Count MAE", ax1, "Count MAE (↓ better)", "Count MAE"),
        ("Precision", ax2, "Precision @ IoU=0.5 (↑)", "Precision"),
        ("Recall", ax3, "Recall @ IoU=0.5 (↑)", "Recall"),
    ]:
        for i, method in enumerate(methods):
            m_data = summary[summary["Method"] == method]
            vals = m_data[metric].values
            offset = (i - n_methods / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w, label=method,
                   color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(occlusion_labels)
        ax.set_xlabel("Occlusion Level")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle("Figure 4: Baseline Method Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig4_baseline_comparison")
    plt.close(fig)


# ====================================================================
# Figure 5: Qualitative Detection Examples (placeholder grid)
# ====================================================================

def figure_5_detection_grid():
    """3x4 grid: methods × occlusion levels."""
    import cv2

    ann_path = SYNTHETIC_DIR / "annotations.json"
    meta_path = SYNTHETIC_DIR / "metadata.csv"
    if not ann_path.exists() or not meta_path.exists():
        print("  ⚠ Skipping Fig 5: synthetic data not found")
        return

    with open(ann_path) as f:
        coco = json.load(f)
    meta = pd.read_csv(meta_path)

    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    gt_by_img = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        fname = img_map[ann["image_id"]]
        gt_by_img.setdefault(fname, []).append((x, y, x+w, y+h))

    from src.baseline.heuristic import HeuristicDetector
    from src.baseline.classical_cv import WatershedSegmenter, GraphSegmenter

    detectors = {
        "Heuristic": HeuristicDetector(),
        "Watershed": WatershedSegmenter(),
        "GraphSeg": GraphSegmenter(),
    }
    occlusion_levels = [0.0, 0.25, 0.50, 0.75]
    level_labels = ["0%", "25%", "50%", "75%"]

    # Pick one image per level
    example_imgs = {}
    for lvl in occlusion_levels:
        fnames = meta[meta["target_occlusion"] == lvl]["file_name"].tolist()
        example_imgs[lvl] = fnames[0] if fnames else None

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for row, (mname, det) in enumerate(detectors.items()):
        for col, (lvl, label) in enumerate(zip(occlusion_levels, level_labels)):
            ax = axes[row, col]
            fname = example_imgs.get(lvl)
            if not fname:
                ax.axis("off")
                continue

            img_path = SYNTHETIC_DIR / "images" / fname
            img = cv2.imread(str(img_path))
            if img is None:
                ax.axis("off")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

            # GT boxes (green)
            for (x1, y1, x2, y2) in gt_by_img.get(fname, []):
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                             lw=1.5, edgecolor="#00E676", facecolor="none"))
            # Pred boxes (red)
            for (x1, y1, x2, y2) in det.detect(str(img_path)):
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                             lw=1.5, edgecolor="#FF1744", facecolor="none", ls="--"))

            if row == 0:
                ax.set_title(f"Occlusion: {label}", fontweight="bold")
            if col == 0:
                ax.set_ylabel(mname, fontsize=12, fontweight="bold")
            ax.axis("off")

    gt_p = mpatches.Patch(edgecolor="#00E676", facecolor="none", lw=2, label="Ground Truth")
    pred_p = mpatches.Patch(edgecolor="#FF1744", facecolor="none", lw=2, label="Predicted")
    fig.legend(handles=[gt_p, pred_p], loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 5: Qualitative Detection Examples",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_fig(fig, "fig5_detection_grid")
    plt.close(fig)


# ====================================================================
# Figure 6: NMS vs Soft-NMS Comparison
# ====================================================================

def figure_6_nms_comparison():
    """Visual comparison of Hard NMS vs Soft-NMS on overlapping boxes."""
    import torch
    from src.models.soft_nms import soft_nms

    # Create overlapping boxes scenario
    boxes = torch.tensor([
        [30, 30, 120, 120],    # Object A (score=0.95)
        [50, 40, 140, 130],    # Object B (score=0.88, overlaps A)
        [160, 30, 250, 120],   # Object C (score=0.82, isolated)
        [170, 45, 255, 125],   # Object D (score=0.75, overlaps C slightly)
        [60, 150, 140, 230],   # Object E (score=0.70, isolated)
    ], dtype=torch.float32)
    scores = torch.tensor([0.95, 0.88, 0.82, 0.75, 0.70])
    labels = ["A", "B", "C", "D", "E"]

    methods = {
        "Hard NMS (IoU=0.5)": ("hard", {}),
        "Soft-NMS Linear": ("linear", {}),
        "Soft-NMS Gaussian (σ=0.5)": ("gaussian", {"sigma": 0.5}),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, (title, (method, kwargs)) in enumerate(methods.items()):
        ax = axes[idx]
        ax.set_xlim(0, 300)
        ax.set_ylim(260, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f9fa")

        keep, new_scores = soft_nms(boxes, scores, method=method,
                                     score_threshold=0.01, iou_threshold=0.5,
                                     **kwargs)
        kept_set = set(keep.tolist())

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            is_kept = i in kept_set
            score_idx = list(keep.tolist()).index(i) if is_kept else -1
            new_s = new_scores[score_idx].item() if is_kept and score_idx >= 0 else 0.0

            color = "#2196F3" if is_kept else "#ccc"
            alpha = 0.7 if is_kept else 0.3
            ls = "-" if is_kept else ":"

            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                              lw=2.5, edgecolor=color, facecolor=(*plt.cm.Blues(0.2)[:3], alpha*0.3),
                              linestyle=ls)
            ax.add_patch(rect)

            s_text = f"{new_s:.2f}" if is_kept else "suppressed"
            font_color = "#1565C0" if is_kept else "#999"
            ax.text(x1+3, y1+15, f"{labels[i]}: {s_text}",
                    fontsize=9, fontweight="bold", color=font_color)

        ax.set_title(f"{title}\n({len(keep)} kept / {len(boxes)} total)",
                     fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Figure 6: NMS vs Soft-NMS on Overlapping Boxes",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_nms_comparison")
    plt.close(fig)


# ====================================================================
# Figure 7: Architecture Diagram
# ====================================================================

def figure_7_architecture():
    """Pipeline architecture diagram using matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 5)
    ax.axis("off")

    # Pipeline stages
    stages = [
        (0.5, 2.5, "Input\nImage", "#E3F2FD"),
        (2.0, 2.5, "Backbone\n(MobileNetV3\nor ResNet50)", "#BBDEFB"),
        (3.8, 2.5, "FPN\n(P2–P6)", "#90CAF9"),
        (5.3, 2.5, "RPN\n+ ROI Head", "#64B5F6"),
        (7.0, 2.5, "Raw\nProposals\n(N≈500)", "#42A5F5"),
        (8.7, 2.5, "Soft-NMS\n(Gaussian\nσ=0.5)", "#FF7043"),
        (10.2, 2.5, "Final\nDetections", "#66BB6A"),
    ]

    for x, y, text, color in stages:
        box = FancyBboxPatch((x-0.6, y-0.7), 1.2, 1.4,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#333", lw=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.6
        x2 = stages[i+1][0] - 0.6
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#555"))

    # Edge pipeline branch
    ax.annotate("Edge Device\n(INT8 ONNX)", xy=(8.7, 1.0),
                fontsize=9, ha="center", fontstyle="italic", color="#E65100",
                bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#E65100", lw=1))
    ax.annotate("", xy=(8.7, 1.5), xytext=(8.7, 1.8),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E65100", ls="--"))

    # Title
    ax.text(5.3, 4.3, "Figure 7: Detection Pipeline Architecture",
            ha="center", fontsize=14, fontweight="bold")

    # Key innovation callout
    ax.annotate("Key Innovation:\nReplace Hard NMS\nwith Soft-NMS",
                xy=(8.7, 3.5), fontsize=8, ha="center", color="#BF360C",
                bbox=dict(boxstyle="round", fc="#FFEBEE", ec="#EF5350", lw=1))
    ax.annotate("", xy=(8.7, 3.2), xytext=(8.7, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1, color="#EF5350"))

    fig.tight_layout()
    save_fig(fig, "fig7_architecture")
    plt.close(fig)


# ====================================================================
# LaTeX Include Commands
# ====================================================================

def generate_latex_includes():
    """Print LaTeX includegraphics commands for all figures."""
    figs = [
        ("fig1_object_count_distribution", "Object count distribution for the synthetic dataset.", "fig:obj_count"),
        ("fig2_size_distributions", "Bounding box area and aspect ratio distributions.", "fig:size_dist"),
        ("fig3_occlusion_analysis", "Average nearest-neighbor IoU vs.\\ number of objects.", "fig:occlusion"),
        ("fig4_baseline_comparison", "Count MAE, Precision, and Recall across classical methods.", "fig:baseline_comp"),
        ("fig5_detection_grid", "Qualitative detection examples: methods vs.\\ occlusion levels.", "fig:det_grid"),
        ("fig6_nms_comparison", "Hard NMS vs.\\ Soft-NMS score decay on overlapping boxes.", "fig:nms_comp"),
        ("fig7_architecture", "Full detection pipeline architecture.", "fig:architecture"),
    ]

    lines = ["\n% ---- Auto-generated LaTeX figure includes ----\n"]
    for fname, caption, label in figs:
        lines.append(f"""\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\linewidth]{{figures/{fname}.pdf}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
""")

    tex_path = FIGURES_DIR / "figure_includes.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  ✓ LaTeX includes → {tex_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 60)
    print("  Compiling Publication-Quality Figures")
    print("=" * 60)
    print()

    figure_1_object_count_distribution()
    figure_2_size_distributions()
    figure_3_occlusion_scatter()
    figure_4_baseline_comparison()
    figure_5_detection_grid()
    figure_6_nms_comparison()
    figure_7_architecture()
    generate_latex_includes()

    print(f"\n{'=' * 60}")
    print(f"  All figures saved to {FIGURES_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
