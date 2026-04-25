"""
plots.py — Plotting Utilities for ML Baseline
==============================================
Provides visualizations for ML baseline evaluation:
  - Feature importances from RandomForest
  - Predicted vs. Actual Object Counts
  - Error Distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_importances(importances: np.ndarray, feature_names: List[str], save_name: str = "rf_feature_importances.png"):
    """
    Plot Random Forest feature importances.
    """
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values(by="Importance", ascending=False)
    
    sns.barplot(data=df, x="Importance", y="Feature", palette="viridis")
    plt.title("Random Forest Feature Importances (Region Classifier)")
    plt.tight_layout()
    
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved feature importances to {out_path.name}")


def plot_predicted_vs_actual_counts(pred_counts: List[int], gt_counts: List[int], save_name: str = "pred_vs_actual_counts.png"):
    """
    Scatter plot of Predicted vs Actual object counts across the evaluation set.
    """
    plt.figure(figsize=(6, 6))
    max_val = max(max(pred_counts), max(gt_counts)) + 2
    
    # 1:1 Reference Line
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="Ideal")
    
    sns.scatterplot(x=gt_counts, y=pred_counts, alpha=0.6, color="blue")
    
    plt.xlabel("Actual Object Count")
    plt.ylabel("Predicted Object Count")
    plt.title("ML Baseline: Predicted vs. Actual Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved predicted vs actual counts plot to {out_path.name}")
