"""
data_loader.py — Dataset Loading Utilities
==========================================
Handles loading and preprocessing of:
  - SKU-110K dataset (CSV annotations, images)
  - Synthetic overlapping shapes dataset (COCO-format JSON)

Provides:
  - Automatic download of SKU-110K annotations (if missing)
  - Annotation parsing & validation
  - Object-count filtering (1–50 objects per image)
  - Reproducible train/val/test split (70/15/15, no image leakage)
  - JSON metadata export
  - Detailed summary statistics
"""

import json
import os
import sys
import urllib.request
import zipfile
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

# SKU-110K annotation download URLs (mirrors, tried in order)
_SKU110K_URLS = [
    # Official Google-Drive direct-download link (annotations CSV ≈ 85 MB)
    "https://drive.usercontent.google.com/download?id=1iq93lCdhaPUN0fWbLieMtzfB1850pKwd",
    # GitHub mirror (annotations only)
    "https://github.com/eg4000/SKU110K_CVPR19/raw/master/annotations/annotations_train.csv",
]

EXPECTED_COLUMNS = [
    "image_name", "x1", "y1", "x2", "y2",
    "class", "image_width", "image_height",
]


# ====================================================================
# Download
# ====================================================================

def download_sku110k_annotations(
    output_path: Optional[Path] = None,
    timeout: int = 120,
) -> Path:
    """
    Download SKU-110K annotation CSV if it doesn't already exist.

    Tries each URL in ``_SKU110K_URLS`` in order.  If the downloaded file
    is a ZIP archive it will be extracted automatically.

    Parameters
    ----------
    output_path : Path, optional
        Where to save the CSV. Defaults to ``data/raw/annotations.csv``.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path
        Absolute path to the downloaded (or already existing) CSV.

    Raises
    ------
    RuntimeError
        If none of the download URLs succeed.
    """
    if output_path is None:
        output_path = RAW_DIR / "annotations.csv"
    output_path = Path(output_path)

    if output_path.exists():
        print(f"[download] Annotations already present at {output_path}")
        return output_path

    # Ensure target directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for url in _SKU110K_URLS:
        print(f"[download] Trying {url[:80]}…")
        try:
            tmp_path = output_path.with_suffix(".tmp")
            _download_with_progress(url, tmp_path, timeout=timeout)

            # If it's a zip, extract the first CSV inside
            if zipfile.is_zipfile(tmp_path):
                print("[download] Extracting ZIP archive…")
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        raise ValueError("ZIP contains no CSV files")
                    zf.extract(csv_names[0], path=output_path.parent)
                    extracted = output_path.parent / csv_names[0]
                    if extracted != output_path:
                        extracted.rename(output_path)
                tmp_path.unlink(missing_ok=True)
            else:
                tmp_path.rename(output_path)

            print(f"[download] ✓ Saved annotations to {output_path}")
            return output_path

        except Exception as exc:
            print(f"[download] ✗ Failed: {exc}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            continue

    raise RuntimeError(
        "Could not download SKU-110K annotations from any mirror.\n"
        "Please download manually and place the CSV at:\n"
        f"  {output_path}\n\n"
        "Official source: https://github.com/eg4000/SKU110K_CVPR19"
    )


def _download_with_progress(url: str, dest: Path, timeout: int = 120) -> None:
    """Download a file with a simple progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        chunk_size = 1024 * 256  # 256 KB

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded / 1e6
                    print(
                        f"\r[download]   {mb:.1f} MB / {total/1e6:.1f} MB "
                        f"({pct:.0f}%)",
                        end="", flush=True,
                    )
        if total:
            print()  # newline after progress


# ====================================================================
# Load & Validate
# ====================================================================

def load_sku110k_annotations(
    csv_path: Optional[str] = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load SKU-110K annotations from CSV.

    Expected columns:
        image_name, x1, y1, x2, y2, class, image_width, image_height

    Parameters
    ----------
    csv_path : str, optional
        Path to the annotations CSV. Defaults to ``data/raw/annotations.csv``.
    auto_download : bool
        If True and the CSV is missing, attempt to download it automatically.

    Returns
    -------
    pd.DataFrame
        Validated annotations dataframe.
    """
    if csv_path is None:
        csv_path = RAW_DIR / "annotations.csv"
    csv_path = Path(csv_path)

    # Auto-download if missing
    if not csv_path.exists():
        if auto_download:
            print("[data_loader] Annotations CSV not found — attempting download…")
            csv_path = download_sku110k_annotations(csv_path)
        else:
            raise FileNotFoundError(
                f"Annotations CSV not found at {csv_path}. "
                f"Set auto_download=True or place the file in data/raw/"
            )

    # Load
    df = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(df):,} annotations from {csv_path.name}")

    # Validate columns
    _validate_columns(df)

    # Validate coordinate values
    _validate_boxes(df)

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Check that the dataframe contains all expected columns."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        # Try common alternate column names
        rename_map = {}
        alt_names = {
            "image_name": ["filename", "img_name", "image", "file"],
            "class": ["label", "category", "class_id", "cls"],
        }
        for expected, alts in alt_names.items():
            if expected in missing:
                for alt in alts:
                    if alt in df.columns:
                        rename_map[alt] = expected
                        missing.discard(expected)
                        break
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            print(f"[data_loader] Renamed columns: {rename_map}")

        if missing:
            raise ValueError(
                f"Annotations CSV is missing columns: {missing}. "
                f"Expected: {EXPECTED_COLUMNS}"
            )


def _validate_boxes(df: pd.DataFrame) -> None:
    """Warn about invalid bounding boxes (x2 < x1 or y2 < y1)."""
    bad_w = (df["x2"] <= df["x1"]).sum()
    bad_h = (df["y2"] <= df["y1"]).sum()
    if bad_w or bad_h:
        print(
            f"[data_loader] ⚠ Found {bad_w} boxes with x2 ≤ x1, "
            f"{bad_h} boxes with y2 ≤ y1 (will be ignored during evaluation)"
        )


# ====================================================================
# Filter
# ====================================================================

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
        Filtered dataframe (only rows belonging to qualifying images).
    """
    counts = df.groupby("image_name").size().reset_index(name="count")
    valid = counts[
        (counts["count"] >= min_objects) & (counts["count"] <= max_objects)
    ]["image_name"]
    filtered = df[df["image_name"].isin(valid)].copy()

    n_dropped = df["image_name"].nunique() - len(valid)
    print(
        f"[data_loader] Filtered: kept {len(valid):,} images with "
        f"{min_objects}–{max_objects} objects ({len(filtered):,} annotations), "
        f"dropped {n_dropped:,} images outside range"
    )
    return filtered


# ====================================================================
# Train / Val / Test Split
# ====================================================================

def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits by unique image (no image leakage).

    Splitting is done at the image level so that no shelf configuration
    appears in more than one split.

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
    assert 0 < train_ratio + val_ratio < 1.0, (
        f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}"
    )

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
        "train": df[df["image_name"].isin(train_imgs)].copy(),
        "val": df[df["image_name"].isin(val_imgs)].copy(),
        "test": df[df["image_name"].isin(test_imgs)].copy(),
    }

    print(f"\n[data_loader] Split summary (seed={seed}):")
    for name, split_df in splits.items():
        n_imgs = split_df["image_name"].nunique()
        print(f"  {name:>5s}: {n_imgs:>6,} images, {len(split_df):>8,} annotations")

    return splits


# ====================================================================
# Save Splits
# ====================================================================

def save_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Save split metadata as JSON files.

    Each file contains a list of entries:
        {"image": "name.jpg", "num_objects": N, "annotations": [[x1,y1,x2,y2], ...]}

    Parameters
    ----------
    splits : dict
        {"train": df, "val": df, "test": df}
    output_dir : str, optional
        Defaults to ``data/processed/``.

    Returns
    -------
    dict
        Mapping split name → output file Path.
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
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
        saved[split_name] = out_path
        print(f"[data_loader] Saved {split_name} split → {out_path.name}")

    return saved


# ====================================================================
# Summary Statistics
# ====================================================================

def print_summary_statistics(
    splits: Dict[str, pd.DataFrame],
    image_dir: Optional[Path] = None,
) -> None:
    """
    Print detailed summary statistics for each split.

    Includes total images, total annotations, and per-image object count
    statistics (min, max, mean, median, std).  Optionally checks which
    images actually exist on disk.

    Parameters
    ----------
    splits : dict
        {"train": df, "val": df, "test": df}
    image_dir : Path, optional
        Directory to check for image availability.
        Defaults to ``data/raw/``.
    """
    if image_dir is None:
        image_dir = RAW_DIR

    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)

    all_counts = []

    for name, split_df in splits.items():
        counts = split_df.groupby("image_name").size()
        all_counts.append(counts)

        print(f"\n  [{name.upper()}]")
        print(f"    Images:        {len(counts):>8,}")
        print(f"    Annotations:   {len(split_df):>8,}")
        print(f"    ── Objects per image ──")
        print(f"       Min:        {counts.min():>8}")
        print(f"       Max:        {counts.max():>8}")
        print(f"       Mean:       {counts.mean():>8.1f}")
        print(f"       Median:     {counts.median():>8.1f}")
        print(f"       Std:        {counts.std():>8.1f}")

    # Overall totals
    overall = pd.concat(all_counts)
    print(f"\n  [OVERALL]")
    print(f"    Images:        {len(overall):>8,}")
    print(f"    Annotations:   {sum(len(s) for s in splits.values()):>8,}")
    print(f"    ── Objects per image ──")
    print(f"       Min:        {overall.min():>8}")
    print(f"       Max:        {overall.max():>8}")
    print(f"       Mean:       {overall.mean():>8.1f}")
    print(f"       Median:     {overall.median():>8.1f}")
    print(f"       Std:        {overall.std():>8.1f}")

    # Image availability check
    all_images = set()
    for split_df in splits.values():
        all_images.update(split_df["image_name"].unique())

    if image_dir.exists():
        available = {
            img for img in all_images
            if (image_dir / img).exists()
        }
        missing = all_images - available
        print(f"\n  [IMAGE AVAILABILITY]")
        print(f"    Directory:     {image_dir}")
        print(f"    Available:     {len(available):>8,} / {len(all_images):,}")
        print(f"    Missing:       {len(missing):>8,}")
        if missing and len(missing) <= 5:
            for m in sorted(missing):
                print(f"      • {m}")
        elif missing:
            sample = sorted(missing)[:3]
            print(f"      • {sample[0]}")
            print(f"      • {sample[1]}")
            print(f"      • {sample[2]}")
            print(f"      • … and {len(missing) - 3:,} more")
        if missing:
            print(
                f"\n    ℹ  Script works with annotations only — "
                f"missing images are handled gracefully."
            )
    else:
        print(f"\n  [IMAGE AVAILABILITY]")
        print(f"    Directory {image_dir} does not exist yet.")
        print(f"    ℹ  Script works with annotations only — no images required.")

    print("\n" + "=" * 70)


# ====================================================================
# Synthetic Dataset Loader
# ====================================================================

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
    print(
        f"[data_loader] Loaded synthetic annotations: "
        f"{len(data.get('images', []))} images"
    )
    return data


# ====================================================================
# Image Loader
# ====================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image as a numpy array (RGB).

    Returns None with a warning if the file does not exist
    (graceful handling of missing images).
    """
    p = Path(image_path)
    if not p.exists():
        print(f"[data_loader] ⚠ Image not found: {p.name}")
        return None
    img = Image.open(p).convert("RGB")
    return np.array(img)


# ====================================================================
# CLI Entry Point
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  High-Density Object Segmentation — Data Loader")
    print("=" * 70)

    # ---- SKU-110K ----
    try:
        df = load_sku110k_annotations()
        df = filter_by_object_count(df)
        splits = create_splits(df)
        save_splits(splits)
        print_summary_statistics(splits)

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Skipping SKU-110K. Place annotations CSV in data/raw/\n")

    except RuntimeError as e:
        print(f"\n⚠️  {e}\n")

    # ---- Synthetic ----
    try:
        synth = load_synthetic_annotations()
        print(
            f"Synthetic dataset: {len(synth.get('images', []))} images loaded."
        )
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")

    print("\nDone.")
