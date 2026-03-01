"""
data_loader.py — Dataset Loading Utilities
==========================================
Handles loading and preprocessing of:
  - SKU-110K dataset (CSV annotations, images)
  - Synthetic overlapping shapes dataset (COCO-format JSON)

Provides train/val/test split generation and filtering for
images containing 1-50 objects (project scope).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---- Constants ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

MIN_OBJECTS = 1
MAX_OBJECTS = 50


# ---- SKU-110K Loader ----

def load_sku110k_annotations(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load SKU-110K annotations from CSV.

    Expected columns: image_name, x1, y1, x2, y2, class, image_width, image_height

    Parameters
    ----------
    csv_path : str, optional
        Path to the annotations CSV. Defaults to data/raw/annotations.csv.

    Returns
    -------
    pd.DataFrame
        Annotations dataframe.
    """
    if csv_path is None:
        csv_path = RAW_DIR / "annotations.csv"

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Annotations CSV not found at {csv_path}. "
            f"Please place the SKU-110K annotations in data/raw/"
        )

    df = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(df)} annotations from {csv_path}")
    return df


def filter_by_object_count(
    df: pd.DataFrame,
    min_objects: int = MIN_OBJECTS,
    max_objects: int = MAX_OBJECTS,
) -> pd.DataFrame:
    """
    Filter to images containing between min_objects and max_objects annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Full annotations dataframe.
    min_objects, max_objects : int
        Inclusive bounds on object count per image.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    counts = df.groupby("image_name").size().reset_index(name="count")
    valid = counts[
        (counts["count"] >= min_objects) & (counts["count"] <= max_objects)
    ]["image_name"]
    filtered = df[df["image_name"].isin(valid)]
    print(
        f"[data_loader] Filtered: {len(valid)} images with "
        f"{min_objects}-{max_objects} objects ({len(filtered)} annotations)"
    )
    return filtered


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits by unique image (no image leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Annotations dataframe.
    train_ratio, val_ratio : float
        Ratios for train and val (test = 1 - train - val).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {"train": df, "val": df, "test": df}
    """
    rng = np.random.RandomState(seed)
    images = df["image_name"].unique()
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_imgs = set(images[:n_train])
    val_imgs = set(images[n_train : n_train + n_val])
    test_imgs = set(images[n_train + n_val :])

    splits = {
        "train": df[df["image_name"].isin(train_imgs)],
        "val": df[df["image_name"].isin(val_imgs)],
        "test": df[df["image_name"].isin(test_imgs)],
    }

    for name, split_df in splits.items():
        n_imgs = split_df["image_name"].nunique()
        print(f"[data_loader] {name}: {n_imgs} images, {len(split_df)} annotations")

    return splits


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Optional[str] = None):
    """Save split metadata as JSON files."""
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        entries = []
        for img_name, group in split_df.groupby("image_name"):
            annotations = group[["x1", "y1", "x2", "y2"]].values.tolist()
            entries.append({
                "image": img_name,
                "num_objects": len(group),
                "annotations": annotations,
            })

        out_path = output_dir / f"{split_name}_split.json"
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"[data_loader] Saved {split_name} split → {out_path}")


# ---- Synthetic Dataset Loader ----

def load_synthetic_annotations(
    annotations_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load synthetic dataset annotations in COCO format.

    Parameters
    ----------
    annotations_path : str, optional
        Path to annotations JSON. Defaults to data/synthetic/annotations.json.

    Returns
    -------
    dict
        COCO-format annotations dictionary.
    """
    if annotations_path is None:
        annotations_path = SYNTHETIC_DIR / "annotations.json"

    if not Path(annotations_path).exists():
        raise FileNotFoundError(
            f"Synthetic annotations not found at {annotations_path}. "
            f"Run: python -m src.synthetic_generator"
        )

    with open(annotations_path) as f:
        data = json.load(f)
    print(f"[data_loader] Loaded synthetic annotations: "
          f"{len(data.get('images', []))} images")
    return data


def load_image(image_path: str) -> np.ndarray:
    """Load an image as a numpy array (RGB)."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


# ---- CLI ----

if __name__ == "__main__":
    print("=" * 60)
    print("High-Density Object Segmentation — Data Loader")
    print("=" * 60)

    # Try loading SKU-110K
    try:
        df = load_sku110k_annotations()
        df = filter_by_object_count(df)
        splits = create_splits(df)
        save_splits(splits)
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Skipping SKU-110K loading. Place annotations in data/raw/\n")

    # Try loading synthetic
    try:
        synth = load_synthetic_annotations()
        print(f"Synthetic dataset: {len(synth.get('images', []))} images loaded.")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
