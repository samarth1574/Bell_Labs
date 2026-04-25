"""
visualization.py — Plotting Helpers
====================================
Provides consistent, publication-quality plotting functions for:
  - Bounding box overlays on images
  - Detection comparison visualizations
  - Metric charts and tables
  - NMS vs Soft-NMS visual comparisons

All plots use a consistent style:
  - Font size: 12pt
  - Figure size: (8, 5) default
  - DPI: 300 for saved figures
  - Seaborn 'whitegrid' style
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---- Style Configuration ----
FIGURE_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette
COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF5722",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "danger": "#F44336",
    "ground_truth": "#00E676",
    "prediction": "#FF1744",
    "soft_nms": "#2979FF",
    "hard_nms": "#FF6D00",
}

# Method colors for comparison plots
METHOD_COLORS = [
    "#2196F3",  # Blue — Heuristic
    "#4CAF50",  # Green — Watershed
    "#FF9800",  # Orange — Graph Seg
    "#9C27B0",  # Purple — Retail Prior
    "#F44336",  # Red — DL + Hard NMS
    "#00BCD4",  # Teal — DL + Soft-NMS
]


def setup_style():
    """Apply consistent matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
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
        }
    )


def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or RGB).
    boxes : list of (x1, y1, x2, y2)
        Bounding box coordinates.
    color : tuple
        BGR color for boxes.
    thickness : int
        Line thickness.
    labels : list of str, optional
        Labels to draw above each box.

    Returns
    -------
    np.ndarray
        Image with boxes drawn.
    """
    img = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if labels and i < len(labels):
            cv2.putText(
                img,
                labels[i],
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return img


def save_figure(fig: plt.Figure, filename: str, output_dir: Optional[str] = None):
    """Save a matplotlib figure to reports/figures/."""
    if output_dir is None:
        output_dir = FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[viz] Saved figure → {filepath}")


# Apply default style on import
setup_style()
