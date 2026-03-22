"""
synthetic_generator.py — Controlled Synthetic Dataset Generator
================================================================
Generates a synthetic dataset of overlapping geometric shapes for
benchmarking detection and segmentation methods under controlled
occlusion conditions.

Dataset specification:
  - 500 images, 256×256 px, white background
  - 1–50 random objects per image (uniform)
  - Shapes: circles, rectangles, ellipses
  - Object sizes: 15–60 px
  - Controlled occlusion levels: 0%, 25%, 50%, 75%
  - COCO-format annotations (JSON)
  - Per-instance binary masks (PNG)
  - Metadata CSV with per-image statistics

Usage:
    python -m src.synthetic_generator                     # defaults
    python -m src.synthetic_generator --num_images 100    # fewer images
    python -m src.synthetic_generator --occlusion 0.5     # 50% target
    python -m src.synthetic_generator --seed 123          # reproducibility
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    _missing = e.name or str(e)
    print(
        f"Error: Missing dependency '{_missing}'.\n"
        f"Install project dependencies first:\n\n"
        f"    pip install -r requirements.txt\n"
    )
    sys.exit(1)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
IMAGES_DIR = SYNTHETIC_DIR / "images"
MASKS_DIR = SYNTHETIC_DIR / "masks"

# ---- Defaults ----
DEFAULT_NUM_IMAGES = 500
DEFAULT_IMG_SIZE = 256
DEFAULT_MIN_OBJECTS = 1
DEFAULT_MAX_OBJECTS = 50
DEFAULT_MIN_OBJ_SIZE = 15
DEFAULT_MAX_OBJ_SIZE = 60
DEFAULT_BG_COLOR = (255, 255, 255)  # white
SHAPE_TYPES = ["circle", "rectangle", "ellipse"]


# ====================================================================
# Shape Drawing
# ====================================================================


def _random_color(rng: np.random.RandomState) -> Tuple[int, int, int]:
    """Generate a vivid random BGR color (avoids near-white)."""
    color = rng.randint(20, 220, size=3).tolist()
    return tuple(color)


def _draw_shape_on_mask(
    mask: np.ndarray,
    shape_type: str,
    cx: int,
    cy: int,
    size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Draw a single shape on a binary mask (255 = object, 0 = background).

    Returns the updated mask.
    """
    if shape_type == "circle":
        radius = size // 2
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)

    elif shape_type == "rectangle":
        w = rng.randint(size // 2, size)
        h = rng.randint(size // 2, size)
        x1 = cx - w // 2
        y1 = cy - h // 2
        cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), 255, thickness=-1)

    elif shape_type == "ellipse":
        axis_a = rng.randint(size // 3, size // 2 + 1)
        axis_b = rng.randint(size // 3, size // 2 + 1)
        angle = rng.randint(0, 180)
        cv2.ellipse(mask, (cx, cy), (axis_a, axis_b), angle, 0, 360, 255, -1)

    return mask


def _draw_shape_on_image(
    image: np.ndarray,
    shape_type: str,
    cx: int,
    cy: int,
    size: int,
    color: Tuple[int, int, int],
    rng: np.random.RandomState,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Draw a colored shape on the image using the same geometry as the mask.

    Re-uses the mask to ensure pixel-perfect alignment.
    """
    colored = np.zeros_like(image)
    colored[:] = color
    obj_pixels = mask > 0
    image[obj_pixels] = colored[obj_pixels]
    return image


# ====================================================================
# Placement Engine (Occlusion Control)
# ====================================================================


def _place_objects(
    num_objects: int,
    img_size: int,
    min_size: int,
    max_size: int,
    target_occlusion: float,
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    """
    Generate placement parameters for N objects with controlled occlusion.

    Strategy:
      - target_occlusion = 0.0: spread objects evenly with no overlap
      - target_occlusion > 0.0: cluster centers closer together to force
        overlap; higher values cluster more aggressively

    Returns a list of dicts with keys: shape_type, cx, cy, size, color.
    """
    objects = []
    margin = max_size // 2 + 2
    valid_min = margin
    valid_max = img_size - margin

    if valid_max <= valid_min:
        valid_min, valid_max = 10, img_size - 10

    for i in range(num_objects):
        shape_type = SHAPE_TYPES[rng.randint(0, len(SHAPE_TYPES))]
        size = rng.randint(min_size, max_size + 1)
        color = _random_color(rng)

        if target_occlusion < 0.01 and objects:
            # No-overlap mode: try to find non-overlapping position
            cx, cy = _find_non_overlapping_position(
                objects, img_size, size, valid_min, valid_max, rng, max_tries=50
            )
        elif target_occlusion > 0.01 and objects:
            # Overlap mode: bias towards existing objects
            cx, cy = _find_overlapping_position(
                objects,
                img_size,
                size,
                target_occlusion,
                valid_min,
                valid_max,
                rng,
            )
        else:
            cx = rng.randint(valid_min, valid_max)
            cy = rng.randint(valid_min, valid_max)

        objects.append(
            {
                "shape_type": shape_type,
                "cx": int(cx),
                "cy": int(cy),
                "size": int(size),
                "color": color,
            }
        )

    return objects


def _find_non_overlapping_position(
    existing: List[Dict],
    img_size: int,
    size: int,
    valid_min: int,
    valid_max: int,
    rng: np.random.RandomState,
    max_tries: int = 50,
) -> Tuple[int, int]:
    """Try to place an object without overlapping existing ones."""
    best_cx, best_cy = rng.randint(valid_min, valid_max), rng.randint(
        valid_min, valid_max
    )
    best_min_dist = 0

    for _ in range(max_tries):
        cx = rng.randint(valid_min, valid_max)
        cy = rng.randint(valid_min, valid_max)
        min_dist = min(
            math.hypot(cx - o["cx"], cy - o["cy"]) - (size + o["size"]) / 2
            for o in existing
        )
        if min_dist > 0:
            return cx, cy
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_cx, best_cy = cx, cy

    return best_cx, best_cy


def _find_overlapping_position(
    existing: List[Dict],
    img_size: int,
    size: int,
    target_occlusion: float,
    valid_min: int,
    valid_max: int,
    rng: np.random.RandomState,
) -> Tuple[int, int]:
    """
    Place an object near an existing one to achieve target occlusion.

    Higher target_occlusion → centers placed closer together.
    """
    # Pick a random existing object to overlap with
    anchor = existing[rng.randint(0, len(existing))]
    avg_radius = (size + anchor["size"]) / 2

    # Distance that produces target occlusion
    # occlusion ≈ 1 - (distance / (2 * avg_radius)), clamped
    # So distance ≈ 2 * avg_radius * (1 - target_occlusion)
    target_dist = avg_radius * (1.0 - target_occlusion) * 1.5

    # Add some randomness (± 30% of target distance)
    jitter = target_dist * 0.3
    dist = max(1, target_dist + rng.uniform(-jitter, jitter))

    angle = rng.uniform(0, 2 * math.pi)
    cx = int(anchor["cx"] + dist * math.cos(angle))
    cy = int(anchor["cy"] + dist * math.sin(angle))

    # Clamp to valid range
    cx = int(np.clip(cx, valid_min, valid_max))
    cy = int(np.clip(cy, valid_min, valid_max))

    return cx, cy


# ====================================================================
# Annotation Helpers (COCO Format)
# ====================================================================


def _mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """Extract [x, y, width, height] bounding box from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask as uncompressed COCO RLE.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W), values 0 or 255.

    Returns
    -------
    dict
        {"counts": [int, ...], "size": [H, W]}
    """
    flat = (mask.ravel(order="F") > 0).astype(np.uint8)
    counts = []
    current = 0
    count = 0
    for val in flat:
        if val == current:
            count += 1
        else:
            counts.append(count)
            count = 1
            current = val
    counts.append(count)
    # RLE starts with 0-count
    if flat[0] != 0:
        counts = [0] + counts
    return {"counts": counts, "size": [mask.shape[0], mask.shape[1]]}


def _compute_pairwise_occlusion(masks: List[np.ndarray]) -> float:
    """
    Compute average pairwise occlusion ratio across all object pairs.

    Occlusion for pair (i, j) = intersection(i, j) / min(area_i, area_j).
    """
    n = len(masks)
    if n < 2:
        return 0.0

    areas = [np.sum(m > 0) for m in masks]
    total_occlusion = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if areas[i] == 0 or areas[j] == 0:
                continue
            intersection = np.sum((masks[i] > 0) & (masks[j] > 0))
            min_area = min(areas[i], areas[j])
            total_occlusion += intersection / min_area
            n_pairs += 1

    return total_occlusion / n_pairs if n_pairs > 0 else 0.0


# ====================================================================
# Main Generation Loop
# ====================================================================


def generate_dataset(
    num_images: int = DEFAULT_NUM_IMAGES,
    img_size: int = DEFAULT_IMG_SIZE,
    min_objects: int = DEFAULT_MIN_OBJECTS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    min_obj_size: int = DEFAULT_MIN_OBJ_SIZE,
    max_obj_size: int = DEFAULT_MAX_OBJ_SIZE,
    occlusion_levels: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    num_images : int
        Total number of images to generate.
    img_size : int
        Width and height of each image (square).
    min_objects, max_objects : int
        Range of objects per image (uniform).
    min_obj_size, max_obj_size : int
        Range of shape sizes in pixels.
    occlusion_levels : list of float, optional
        Target occlusion ratios. Images are distributed evenly across
        these levels. Defaults to [0.0, 0.25, 0.50, 0.75].
    output_dir : Path, optional
        Root directory for output. Defaults to data/synthetic/.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        COCO-format annotations dictionary.
    """
    if occlusion_levels is None:
        occlusion_levels = [0.0, 0.25, 0.50, 0.75]

    if output_dir is None:
        output_dir = SYNTHETIC_DIR
    output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # COCO structure
    coco = {
        "info": {
            "description": "Synthetic Overlapping Shapes Dataset",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
            "contributor": "High-Density Object Segmentation Project",
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "circle", "supercategory": "shape"},
            {"id": 2, "name": "rectangle", "supercategory": "shape"},
            {"id": 3, "name": "ellipse", "supercategory": "shape"},
        ],
        "images": [],
        "annotations": [],
    }
    category_map = {"circle": 1, "rectangle": 2, "ellipse": 3}

    # Metadata CSV rows
    metadata_rows = []

    # Distribute images across occlusion levels
    imgs_per_level = num_images // len(occlusion_levels)
    remainder = num_images % len(occlusion_levels)

    annotation_id = 1
    image_id = 1

    print(f"\n{'=' * 60}")
    print("  Synthetic Dataset Generator")
    print(f"{'=' * 60}")
    print(f"  Images:          {num_images}")
    print(f"  Image size:      {img_size}×{img_size}")
    print(f"  Objects/image:   {min_objects}–{max_objects}")
    print(f"  Object size:     {min_obj_size}–{max_obj_size} px")
    print(f"  Occlusion levels: {occlusion_levels}")
    print(f"  Seed:            {seed}")
    print(f"  Output:          {output_dir}")
    print(f"{'=' * 60}\n")

    for level_idx, target_occ in enumerate(occlusion_levels):
        n_imgs = imgs_per_level + (1 if level_idx < remainder else 0)

        desc = f"Occlusion {target_occ:.0%}"
        for _ in tqdm(range(n_imgs), desc=desc, unit="img"):
            num_objects = rng.randint(min_objects, max_objects + 1)
            filename = f"synth_{image_id:05d}.png"

            # Place objects
            obj_params = _place_objects(
                num_objects,
                img_size,
                min_obj_size,
                max_obj_size,
                target_occ,
                rng,
            )

            # Render image and masks
            image = np.full((img_size, img_size, 3), DEFAULT_BG_COLOR, dtype=np.uint8)
            instance_masks = []

            for obj in obj_params:
                # Create individual mask
                mask = np.zeros((img_size, img_size), dtype=np.uint8)
                mask = _draw_shape_on_mask(
                    mask,
                    obj["shape_type"],
                    obj["cx"],
                    obj["cy"],
                    obj["size"],
                    rng,
                )
                instance_masks.append(mask)

                # Draw on composite image
                image = _draw_shape_on_image(
                    image,
                    obj["shape_type"],
                    obj["cx"],
                    obj["cy"],
                    obj["size"],
                    obj["color"],
                    rng,
                    mask,
                )

            # Compute actual occlusion
            avg_occlusion = _compute_pairwise_occlusion(instance_masks)

            # Save image
            cv2.imwrite(str(images_dir / filename), image)

            # Save instance masks (one per object, plus combined)
            img_mask_dir = masks_dir / f"synth_{image_id:05d}"
            img_mask_dir.mkdir(exist_ok=True)

            combined_mask = np.zeros((img_size, img_size), dtype=np.uint16)

            for inst_idx, (mask, obj) in enumerate(
                zip(instance_masks, obj_params), start=1
            ):
                # Individual instance mask
                mask_path = img_mask_dir / f"instance_{inst_idx:03d}.png"
                cv2.imwrite(str(mask_path), mask)

                # Add to combined (instance ID as pixel value)
                combined_mask[mask > 0] = inst_idx

                # COCO annotation
                bbox = _mask_to_bbox(mask)
                if bbox is None:
                    continue
                area = int(np.sum(mask > 0))
                rle = _mask_to_rle(mask)

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map[obj["shape_type"]],
                        "bbox": bbox,
                        "area": area,
                        "segmentation": rle,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            # Save combined mask
            cv2.imwrite(str(img_mask_dir / "combined.png"), combined_mask)

            # COCO image entry
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": img_size,
                    "height": img_size,
                }
            )

            # Metadata row
            metadata_rows.append(
                {
                    "image_id": image_id,
                    "file_name": filename,
                    "num_objects": num_objects,
                    "target_occlusion": target_occ,
                    "actual_avg_occlusion": round(avg_occlusion, 4),
                    "occlusion_level_label": f"{target_occ:.0%}",
                }
            )

            image_id += 1

    # ---- Save COCO annotations ----
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"\n[synth] Saved COCO annotations → {annotations_path.name}")
    print(
        f"        {len(coco['images'])} images, {len(coco['annotations'])} annotations"
    )

    # ---- Save metadata CSV ----
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"[synth] Saved metadata CSV   → {metadata_path.name}")

    # ---- Print summary ----
    _print_summary(metadata_rows, occlusion_levels)

    return coco


def _print_summary(
    metadata: List[Dict[str, Any]],
    occlusion_levels: List[float],
) -> None:
    """Print generation summary statistics."""
    print(f"\n{'=' * 60}")
    print("  GENERATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total images: {len(metadata)}")

    for level in occlusion_levels:
        level_rows = [r for r in metadata if abs(r["target_occlusion"] - level) < 1e-6]
        if not level_rows:
            continue
        n = len(level_rows)
        obj_counts = [r["num_objects"] for r in level_rows]
        actual_occ = [r["actual_avg_occlusion"] for r in level_rows]
        print(f"\n  [{level:.0%} target occlusion] — {n} images")
        print(
            f"    Objects/image:    min={min(obj_counts)}, max={max(obj_counts)}, "
            f"mean={np.mean(obj_counts):.1f}"
        )
        print(
            f"    Actual occlusion: min={min(actual_occ):.3f}, "
            f"max={max(actual_occ):.3f}, mean={np.mean(actual_occ):.3f}"
        )

    print(f"\n{'=' * 60}")


# ====================================================================
# CLI
# ====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic overlapping shapes dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Total number of images to generate",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Image width and height in pixels",
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=DEFAULT_MIN_OBJECTS,
        help="Minimum objects per image",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=DEFAULT_MAX_OBJECTS,
        help="Maximum objects per image",
    )
    parser.add_argument(
        "--min_obj_size",
        type=int,
        default=DEFAULT_MIN_OBJ_SIZE,
        help="Minimum object size in pixels",
    )
    parser.add_argument(
        "--max_obj_size",
        type=int,
        default=DEFAULT_MAX_OBJ_SIZE,
        help="Maximum object size in pixels",
    )
    parser.add_argument(
        "--occlusion",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75],
        help="Target occlusion levels (space-separated)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/synthetic/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    generate_dataset(
        num_images=args.num_images,
        img_size=args.img_size,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        min_obj_size=args.min_obj_size,
        max_obj_size=args.max_obj_size,
        occlusion_levels=args.occlusion,
        output_dir=output_dir,
        seed=args.seed,
    )
