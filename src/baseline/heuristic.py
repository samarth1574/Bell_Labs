"""
heuristic.py — Blob/Contour Baseline Detector
==============================================
Simple image-processing baseline for object separation using:
  - Adaptive thresholding (Gaussian)
  - Morphological operations (opening + closing)
  - Contour detection (RETR_EXTERNAL)
  - Watershed splitting for merged blobs
  - Area-based filtering

This serves as the non-ML baseline to demonstrate limitations
of simple approaches in dense, occluded scenes.

Usage:
    python -m src.baseline.heuristic                         # 10 synthetic samples
    python -m src.baseline.heuristic --image path/to/img.png # single image
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


# ====================================================================
# HeuristicDetector
# ====================================================================

class HeuristicDetector:
    """
    Blob/contour-based baseline detector.

    Parameters
    ----------
    block_size : int
        Block size for adaptive thresholding (must be odd).
    C : int
        Constant subtracted from mean in adaptive threshold.
    min_area : int
        Minimum contour area to keep (px²).
    max_area_ratio : float
        Maximum contour area as fraction of total image area.
    watershed_area_thresh : float
        Contours with area > (image_area * this) are split via watershed.
    morph_open_ksize : int
        Kernel size for morphological opening.
    morph_close_ksize : int
        Kernel size for morphological closing.
    distance_thresh_ratio : float
        Fraction of max distance-transform value used as threshold
        for watershed marker generation.
    """

    def __init__(
        self,
        block_size: int = 11,
        C: int = 2,
        min_area: int = 100,
        max_area_ratio: float = 0.5,
        watershed_area_thresh: float = 0.05,
        morph_open_ksize: int = 3,
        morph_close_ksize: int = 5,
        distance_thresh_ratio: float = 0.4,
    ):
        self.block_size = block_size
        self.C = C
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.watershed_area_thresh = watershed_area_thresh
        self.morph_open_ksize = morph_open_ksize
        self.morph_close_ksize = morph_close_ksize
        self.distance_thresh_ratio = distance_thresh_ratio

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def detect(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects and return bounding boxes.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        list of (x1, y1, x2, y2)
            Bounding boxes for detected objects.
        """
        image = self._load_image(image_path)
        if image is None:
            return []
        binary = self._preprocess(image)
        contours = self._find_contours(binary, image)
        boxes = self._contours_to_boxes(contours)
        return boxes

    def detect_with_masks(
        self, image_path: str
    ) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Detect objects and return bounding boxes + binary masks.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2)
        masks : list of np.ndarray
            Per-instance binary masks (same size as input image).
        """
        image = self._load_image(image_path)
        if image is None:
            return [], []
        binary = self._preprocess(image)
        contours = self._find_contours(binary, image)
        boxes = self._contours_to_boxes(contours)
        masks = self._contours_to_masks(contours, image.shape[:2])
        return boxes, masks

    def detect_from_array(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Run detection on an already-loaded image array (BGR)."""
        binary = self._preprocess(image)
        contours = self._find_contours(binary, image)
        return self._contours_to_boxes(contours)

    # ----------------------------------------------------------------
    # Pipeline Steps
    # ----------------------------------------------------------------

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image as BGR numpy array."""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[heuristic] ⚠ Cannot read image: {image_path}")
        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1–3: Grayscale → adaptive threshold → morphological ops.
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Adaptive thresholding (Gaussian)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.block_size,
            C=self.C,
        )

        # 3. Morphological opening (remove noise) then closing (fill gaps)
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_open_ksize, self.morph_open_ksize),
        )
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_close_ksize, self.morph_close_ksize),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        return binary

    def _find_contours(
        self, binary: np.ndarray, image: np.ndarray
    ) -> List[np.ndarray]:
        """
        Step 4–6: Find contours, filter by area, watershed-split large blobs.
        """
        h, w = binary.shape[:2]
        image_area = h * w
        max_area = image_area * self.max_area_ratio
        watershed_thresh = image_area * self.watershed_area_thresh

        # Step 4: Find external contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        final_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Step 5: Filter by area
            if area < self.min_area or area > max_area:
                continue

            # Step 6: Watershed split for large merged blobs
            if area > watershed_thresh:
                sub_contours = self._watershed_split(cnt, binary, image)
                if sub_contours:
                    final_contours.extend(sub_contours)
                else:
                    final_contours.append(cnt)
            else:
                final_contours.append(cnt)

        return final_contours

    def _watershed_split(
        self, contour: np.ndarray, binary: np.ndarray, image: np.ndarray
    ) -> List[np.ndarray]:
        """
        Apply watershed to split a large merged blob into sub-objects.

        Steps:
          a. Create mask for this contour
          b. Distance transform
          c. Threshold distance map to find seeds
          d. Label connected components as markers
          e. Apply cv2.watershed
          f. Extract sub-contours from watershed regions
        """
        h, w = binary.shape[:2]

        # (a) Mask for this contour only
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # (b) Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # (c) Threshold to find peaks (foreground markers)
        dist_max = dist.max()
        if dist_max < 1:
            return []
        thresh_val = dist_max * self.distance_thresh_ratio
        _, fg = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)

        # (d) Label connected components as markers
        num_labels, markers = cv2.connectedComponents(fg)

        # Only split if we found more than 1 region
        if num_labels <= 2:  # 1 = background + 1 foreground = no split needed
            return []

        # Background = 0, shift markers so background is 1
        markers = markers + 1

        # Mark the unknown region (mask boundary) as 0
        unknown = cv2.subtract(mask, fg)
        markers[unknown == 255] = 0

        # (e) Apply watershed
        img_color = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        # (f) Extract sub-contours from watershed regions
        sub_contours = []
        for label in range(2, num_labels + 1):  # skip 0 (unknown) and 1 (bg)
            region_mask = np.zeros((h, w), dtype=np.uint8)
            region_mask[markers == label] = 255
            region_contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for rc in region_contours:
                if cv2.contourArea(rc) >= self.min_area:
                    sub_contours.append(rc)

        return sub_contours

    # ----------------------------------------------------------------
    # Conversion Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _contours_to_boxes(
        contours: List[np.ndarray],
    ) -> List[Tuple[int, int, int, int]]:
        """Convert contours to (x1, y1, x2, y2) bounding boxes."""
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _contours_to_masks(
        contours: List[np.ndarray], shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Convert contours to per-instance binary masks."""
        masks = []
        for cnt in contours:
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            masks.append(mask)
        return masks

    def __repr__(self) -> str:
        return (
            f"HeuristicDetector(block_size={self.block_size}, C={self.C}, "
            f"min_area={self.min_area}, max_area_ratio={self.max_area_ratio})"
        )


# ====================================================================
# Evaluation Utilities
# ====================================================================

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detections(
    pred_boxes: List[Tuple],
    gt_boxes: List[Tuple],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate predicted boxes against ground truth.

    Uses Hungarian matching (scipy.optimize.linear_sum_assignment) to find
    the optimal one-to-one assignment that maximises total IoU.

    Parameters
    ----------
    pred_boxes : list of (x1, y1, x2, y2)
    gt_boxes : list of (x1, y1, x2, y2)
    iou_threshold : float
        Minimum IoU for a match to count as a true positive.

    Returns
    -------
    dict with keys:
        count_error    : |predicted_count - true_count|
        precision      : TP / (TP + FP)
        recall         : TP / (TP + FN)
        f1             : harmonic mean of precision and recall
        mean_iou       : mean IoU of matched (TP) pairs
        num_predicted  : len(pred_boxes)
        num_gt         : len(gt_boxes)
        tp, fp, fn     : true/false positive/negative counts
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 and n_gt == 0:
        return {
            "count_error": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_iou": 1.0, "num_predicted": 0, "num_gt": 0,
            "tp": 0, "fp": 0, "fn": 0,
        }
    if n_pred == 0:
        return {
            "count_error": n_gt, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mean_iou": 0.0, "num_predicted": 0, "num_gt": n_gt,
            "tp": 0, "fp": 0, "fn": n_gt,
        }
    if n_gt == 0:
        return {
            "count_error": n_pred, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mean_iou": 0.0, "num_predicted": n_pred, "num_gt": 0,
            "tp": 0, "fp": n_pred, "fn": 0,
        }

    # Build IoU cost matrix (negate for minimisation)
    cost = np.zeros((n_pred, n_gt))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            cost[i, j] = compute_iou(pb, gb)

    # Hungarian matching (maximise IoU → minimise negative IoU)
    row_ind, col_ind = linear_sum_assignment(-cost)

    # Count TPs (matched pairs above IoU threshold)
    tp = 0
    matched_ious = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= iou_threshold:
            tp += 1
            matched_ious.append(cost[r, c])

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "count_error": abs(n_pred - n_gt),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "num_predicted": n_pred,
        "num_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_on_synthetic(
    detector: HeuristicDetector,
    num_samples: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run the detector on sample synthetic images and evaluate.

    Returns per-image evaluation results.
    """
    # Load synthetic annotations
    ann_path = SYNTHETIC_DIR / "annotations.json"
    if not ann_path.exists():
        print("[heuristic] ⚠ Synthetic data not found. Run: python -m src.synthetic_generator")
        return []

    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # Build per-image annotation map
    img_anns: Dict[int, List] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        img_anns.setdefault(img_id, []).append((x, y, x + w, y + h))

    # Sample images
    rng = np.random.RandomState(seed)
    if len(images) > num_samples:
        sample_indices = rng.choice(len(images), num_samples, replace=False)
        sampled = [images[i] for i in sample_indices]
    else:
        sampled = images[:num_samples]

    results = []
    for img_info in sampled:
        img_id = img_info["id"]
        filename = img_info["file_name"]
        img_path = SYNTHETIC_DIR / "images" / filename
        gt_boxes = img_anns.get(img_id, [])

        pred_boxes = detector.detect(str(img_path))
        metrics = evaluate_detections(pred_boxes, gt_boxes)
        metrics["image"] = filename
        results.append(metrics)

    return results


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run heuristic baseline detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to run detection on",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of synthetic samples to evaluate on",
    )
    parser.add_argument(
        "--block_size", type=int, default=11,
        help="Adaptive threshold block size",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum contour area",
    )
    args = parser.parse_args()

    detector = HeuristicDetector(
        block_size=args.block_size,
        min_area=args.min_area,
    )
    print(f"\n{detector}\n")

    if args.image:
        # Single-image mode
        boxes = detector.detect(args.image)
        print(f"Detected {len(boxes)} objects in {args.image}")
        for i, b in enumerate(boxes):
            print(f"  [{i+1}] x1={b[0]}, y1={b[1]}, x2={b[2]}, y2={b[3]}")
    else:
        # Evaluate on synthetic samples
        print(f"Evaluating on {args.num_samples} synthetic images…\n")
        results = evaluate_on_synthetic(detector, num_samples=args.num_samples)

        if not results:
            sys.exit(1)

        # Print per-image results
        print(f"{'Image':<22} {'GT':>4} {'Pred':>5} {'CntErr':>7} "
              f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'mIoU':>6}")
        print("─" * 70)

        for r in results:
            print(
                f"{r['image']:<22} {r['num_gt']:>4} {r['num_predicted']:>5} "
                f"{r['count_error']:>7} {r['precision']:>6.3f} {r['recall']:>6.3f} "
                f"{r['f1']:>6.3f} {r['mean_iou']:>6.3f}"
            )

        # Averages
        print("─" * 70)
        avg_keys = ["count_error", "precision", "recall", "f1", "mean_iou"]
        avgs = {k: np.mean([r[k] for r in results]) for k in avg_keys}
        print(
            f"{'AVERAGE':<22} {'':>4} {'':>5} "
            f"{avgs['count_error']:>7.1f} {avgs['precision']:>6.3f} "
            f"{avgs['recall']:>6.3f} {avgs['f1']:>6.3f} {avgs['mean_iou']:>6.3f}"
        )
        print()
