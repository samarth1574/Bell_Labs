"""
augmentations_eda.py — Exploratory Data Analysis for Dataset Transformations
============================================================================
Visualizes the effects of normalisation, color jittering,
and spatial transformations on dense imagery bounding boxes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import torch

from src.data_loader import load_sku110k_annotations
from src.dataset import DenseObjectDataset
from src.models.config import DatasetConfig

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def demonstrate_augmentations():
    """Load an image using the PyTorch Dataset with augmentations enabled and plot it."""
    
    try:
        df = load_sku110k_annotations(auto_download=False)
    except FileNotFoundError:
        print("SKU-110K not found locally. Using a synthetic stub.")
        df = pd.DataFrame([{
            "image_name": "synth_00001.png", "x1": 50, "y1": 50, 
            "x2": 100, "y2": 100, "class": 1, "image_width": 300, "image_height": 300
        }])
        image_dir = PROJECT_ROOT / "data" / "synthetic" / "images"
    else:
        image_dir = PROJECT_ROOT / "data" / "raw"

    config = DatasetConfig(
        use_flips=True,
        use_jitter=True,
    )
    
    dataset = DenseObjectDataset(df, image_dir=str(image_dir), config=config, is_train=True)
    
    if len(dataset) == 0:
        print("No images found in dataset.")
        return

    # Fetch one augmented image
    img_t, target = dataset[0]
    
    # Denormalise for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_t = img_t * std + mean
    img_t = torch.clamp(img_t, 0, 1)
    
    img_np = img_t.permute(1, 2, 0).numpy()
    boxes = target["boxes"].numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
    ax.set_title("Augmented Dataset Image (+ Jitter, H/V-Flip)")
    ax.axis('off')
    
    out_path = FIGURES_DIR / "augmentation_demo.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved augmentation visualization to: {out_path.name}")

if __name__ == "__main__":
    demonstrate_augmentations()
