#!/usr/bin/env python3
"""
Download and prepare SKU-110K for Phase 3 experiments.

The official dataset is distributed for academic, non-commercial use. This
script keeps the download reproducible, converts CSV annotations to YOLO
format, and writes a dataset YAML compatible with Ultralytics training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


OFFICIAL_ARCHIVE_URL = "https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
ANNOTATION_URLS = [
    "https://github.com/eg4000/SKU110K_CVPR19/raw/master/annotations/annotations_train.csv",
    "https://github.com/eg4000/SKU110K_CVPR19/raw/master/annotations/annotations_val.csv",
    "https://github.com/eg4000/SKU110K_CVPR19/raw/master/annotations/annotations_test.csv",
]
EXPECTED_COLUMNS = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PreparedPaths:
    root: Path
    images_dir: Path
    source_images_dir: Path
    labels_dir: Path
    annotations_dir: Path
    data_yaml: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/prepare SKU-110K in YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("data/sku110k"))
    parser.add_argument("--source", type=Path, default=None, help="Existing extracted SKU110K directory.")
    parser.add_argument("--annotations", type=Path, default=None, help="Single CSV or directory of CSV annotations.")
    parser.add_argument("--download", choices=["none", "full", "annotations"], default="none")
    parser.add_argument(
        "--fraction", type=float, default=0.25,
        help="Fraction of unique images to keep (0.0–1.0). Use 0.25 to train on 25%% of SKU-110K.",
    )
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = PreparedPaths(
        root=root,
        images_dir=root / "images",
        source_images_dir=root / "source_images",
        labels_dir=root / "labels",
        annotations_dir=root / "annotations",
        data_yaml=root / "sku110k.yaml",
        manifest=root / "manifest.json",
    )
    root.mkdir(parents=True, exist_ok=True)

    source = args.source.resolve() if args.source else None
    if args.download == "full":
        source = download_and_extract(root / "downloads", root / "extracted", force=args.force)
    elif args.download == "annotations":
        download_annotations_only(paths.annotations_dir, force=args.force)

    if source:
        discover_and_link_source(source, paths)

    annotation_inputs = discover_annotation_files(args.annotations, paths.annotations_dir, source)
    if not annotation_inputs:
        raise FileNotFoundError(
            "No SKU-110K annotation CSV files found. Pass --annotations, --source, or --download full/annotations."
        )

    frames = [read_annotation_csv(path) for path in annotation_inputs]
    annotations = pd.concat(frames, ignore_index=True)
    annotations = validate_and_filter(annotations, args.min_objects, args.max_objects)

    # ---- Subset to requested fraction of images ----
    annotations = subsample_images(annotations, args.fraction, args.seed)

    split_frames = assign_splits(annotations, annotation_inputs, args.seed)
    convert_to_yolo(split_frames, paths, force=args.force)
    write_data_yaml(paths)
    write_manifest(paths, split_frames, annotation_inputs, args)

    print(f"[sku110k] Prepared dataset: {paths.root}")
    print(f"[sku110k] Data YAML: {paths.data_yaml}")


def download_and_extract(download_dir: Path, extract_dir: Path, force: bool = False) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "SKU110K_fixed.tar.gz"

    if force or not archive_path.exists():
        print(f"[sku110k] Downloading official archive to {archive_path}")
        download_file(OFFICIAL_ARCHIVE_URL, archive_path)
    else:
        print(f"[sku110k] Reusing archive {archive_path}")

    marker = extract_dir / ".complete"
    if force or not marker.exists():
        print(f"[sku110k] Extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract_tar(tar, extract_dir)
        marker.write_text(sha256_file(archive_path), encoding="utf-8")

    candidates = [p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() in {"sku110k", "sku-110k"}]
    return candidates[0] if candidates else extract_dir


def download_annotations_only(annotations_dir: Path, force: bool = False) -> None:
    """Download only the annotation CSV files (much smaller than the full archive)."""
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for url in ANNOTATION_URLS:
        filename = url.rsplit("/", 1)[-1]
        dest = annotations_dir / filename
        if dest.exists() and not force:
            print(f"[sku110k] Reusing annotation {dest.name}")
            continue
        print(f"[sku110k] Downloading {filename}")
        try:
            download_file(url, dest)
        except Exception as exc:
            print(f"[sku110k] Warning: could not download {filename}: {exc}")


def subsample_images(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep only `fraction` of unique images (reproducible subset)."""
    if fraction >= 1.0:
        return df
    all_images = df["image_name"].unique()
    n_total = len(all_images)
    n_keep = max(1, int(n_total * fraction))
    rng = pd.Series(all_images).sample(n=n_keep, random_state=seed).tolist()
    out = df[df["image_name"].isin(set(rng))].copy()
    print(f"[sku110k] Subsampled {fraction:.0%} of images: {n_keep:,} / {n_total:,}")
    return out


def download_file(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        with dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[sku110k] {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    if total:
        print()


def safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(dest)


def discover_and_link_source(source: Path, paths: PreparedPaths) -> None:
    image_roots = [p for p in source.rglob("*") if p.is_dir() and p.name.lower() in {"images", "imgs"}]
    annotation_roots = [p for p in source.rglob("*") if p.is_dir() and "annotation" in p.name.lower()]

    if image_roots and not paths.source_images_dir.exists():
        link_or_copy(image_roots[0], paths.source_images_dir)
    if annotation_roots and not paths.annotations_dir.exists():
        link_or_copy(annotation_roots[0], paths.annotations_dir)

    for zip_path in source.rglob("*.zip"):
        if "annotation" in zip_path.name.lower():
            paths.annotations_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(paths.annotations_dir)


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(src, dest)


def discover_annotation_files(explicit: Path | None, annotations_dir: Path, source: Path | None) -> list[Path]:
    candidates: list[Path] = []
    roots: list[Path] = []
    if explicit:
        roots.append(explicit)
    roots.append(annotations_dir)
    if source:
        roots.append(source)

    for root in roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.rglob("*.csv"))

    return sorted({p.resolve() for p in candidates if "meta" not in p.name.lower()})


def read_annotation_csv(path: Path) -> pd.DataFrame:
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)

    df = pd.read_csv(path, header=0 if has_header else None)
    if not has_header:
        if df.shape[1] < 8:
            raise ValueError(f"{path} has {df.shape[1]} columns; expected at least 8")
        df = df.iloc[:, :8]
        df.columns = EXPECTED_COLUMNS

    rename = {
        "filename": "image_name",
        "file": "image_name",
        "image": "image_name",
        "img_name": "image_name",
        "label": "class",
        "category": "class",
        "class_id": "class",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["source_csv"] = path.name
    return df


def validate_and_filter(df: pd.DataFrame, min_objects: int, max_objects: int) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {missing}")

    out = df.copy()
    for col in ["x1", "y1", "x2", "y2", "image_width", "image_height"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["image_name", "x1", "y1", "x2", "y2", "image_width", "image_height"])
    out = out[(out["x2"] > out["x1"]) & (out["y2"] > out["y1"])]

    counts = out.groupby("image_name").size()
    keep = counts[(counts >= min_objects) & (counts <= max_objects)].index
    out = out[out["image_name"].isin(keep)].copy()
    print(f"[sku110k] Kept {out['image_name'].nunique():,} images and {len(out):,} boxes")
    return out


def assign_splits(df: pd.DataFrame, files: Iterable[Path], seed: int) -> dict[str, pd.DataFrame]:
    by_name: dict[str, pd.DataFrame] = {}
    lower_names = {p.name.lower(): p for p in files}

    for split in SPLITS:
        split_file = next((name for name in lower_names if split in name), None)
        if split_file:
            split_sources = df["source_csv"].str.lower() == split_file
            if split_sources.any():
                by_name[split] = df[split_sources].copy()

    assigned = set().union(*(set(s["image_name"].unique()) for s in by_name.values())) if by_name else set()
    remainder = df[~df["image_name"].isin(assigned)].copy()

    if not by_name or len(remainder):
        images = pd.Series(remainder["image_name"].unique()).sample(frac=1.0, random_state=seed).tolist()
        n = len(images)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)
        generated = {
            "train": set(images[:train_end]),
            "val": set(images[train_end:val_end]),
            "test": set(images[val_end:]),
        }
        for split, image_names in generated.items():
            piece = remainder[remainder["image_name"].isin(image_names)].copy()
            by_name[split] = pd.concat([by_name.get(split, pd.DataFrame()), piece], ignore_index=True)

    for split in SPLITS:
        print(f"[sku110k] {split:>5}: {by_name[split]['image_name'].nunique():,} images")
    return by_name


def convert_to_yolo(split_frames: dict[str, pd.DataFrame], paths: PreparedPaths, force: bool) -> None:
    for split, frame in split_frames.items():
        label_dir = paths.labels_dir / split
        image_dir = paths.images_dir / split
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for image_name, group in frame.groupby("image_name"):
            image_src = find_image_file(paths, image_name)
            image_dst = image_dir / Path(image_name).name
            if image_src and not image_dst.exists():
                link_or_copy_file(image_src, image_dst)

            label_path = label_dir / f"{Path(image_name).stem}.txt"
            if label_path.exists() and not force:
                continue
            rows = []
            for row in group.itertuples(index=False):
                width = max(float(row.image_width), 1.0)
                height = max(float(row.image_height), 1.0)
                xc = ((float(row.x1) + float(row.x2)) / 2.0) / width
                yc = ((float(row.y1) + float(row.y2)) / 2.0) / height
                bw = (float(row.x2) - float(row.x1)) / width
                bh = (float(row.y2) - float(row.y1)) / height
                rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def find_image_file(paths: PreparedPaths, image_name: str) ->