"""
classical_cv.py — Advanced Non-DL Computer Vision Methods
=========================================================
Implements:
  - WatershedSegmenter: distance-transform-based watershed
  - GraphSegmenter: Felzenszwalb graph-based segmentation
  - RetailPriorDetector: domain-specific priors (shelf lines, grid layout)

These methods represent the "advanced classical CV" tier,
bridging the gap between simple heuristics and deep learning.

Usage:
    python -m src.baseline.classical_cv                       # eval on synthetic
    python -m src.baseline.classical_cv --image path/to/img   # single image
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import felzenszwalb

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


# ====================================================================
# Base Class
# ====================================================================


class BaseDetector:
    """Common interface for classical CV detectors."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}

    @property
    def params(self) -> Dict[str, Any]:
        """Return current parameter dict (for experiment logging)."""
        return dict(self._params)

    def set_params(self, **kwargs) -> None:
        """Update parameters."""
        self._params.update(kwargs)

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects → list of (x1, y1, x2, y2) bounding boxes.

        Parameters
        ----------
        image : str or np.ndarray
            File path or BGR image array.
        """
        raise NotImplementedError

    def detect_with_masks(
        self, image
    ) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Detect objects → (boxes, masks)."""
        raise NotImplementedError

    def _load(self, image) -> Optional[np.ndarray]:
        """Load image from path or pass through array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                print(f"[{self.__class__.__name__}] ⚠ Cannot read: {image}")
            return img
        return image

    @staticmethod
    def _contours_to_boxes(contours):
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _contours_to_masks(contours, shape):
        masks = []
        for cnt in contours:
            m = np.zeros(shape[:2], dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
            masks.append(m)
        return masks


# ====================================================================
# WatershedSegmenter
# ====================================================================


class WatershedSegmenter(BaseDetector):
    """
    Distance-transform-based watershed segmentation.

    Pipeline:
      1. Grayscale → Gaussian blur → Otsu threshold
      2. Distance transform
      3. peak_local_max for marker seeds
      4. cv2.watershed
      5. Extract regions as contours

    Parameters
    ----------
    blur_ksize : int
        Gaussian blur kernel size (odd).
    min_distance : int
        Minimum distance between peaks in distance transform.
    min_area : int
        Minimum region area in pixels.
    max_area_ratio : float
        Maximum region area as fraction of image area.
    """

    def __init__(
        self,
        blur_ksize: int = 7,
        min_distance: int = 10,
        min_area: int = 80,
        max_area_ratio: float = 0.5,
    ):
        super().__init__(
            {
                "blur_ksize": blur_ksize,
                "min_distance": min_distance,
                "min_area": min_area,
                "max_area_ratio": max_area_ratio,
            }
        )
        self.blur_ksize = blur_ksize
        self.min_distance = min_distance
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        img = self._load(image)
        if img is None:
            return []
        labels = self._segment(img)
        contours = self._labels_to_contours(labels, img.shape)
        return self._contours_to_boxes(contours)

    def detect_with_masks(self, image):
        img = self._load(image)
        if img is None:
            return [], []
        labels = self._segment(img)
        contours = self._labels_to_contours(labels, img.shape)
        return self._contours_to_boxes(contours), self._contours_to_masks(
            contours, img.shape
        )

    def _segment(self, img: np.ndarray) -> np.ndarray:
        """Run the watershed pipeline, returns label image."""
        h, w = img.shape[:2]

        # 1. Grayscale → blur → Otsu
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 2. Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 3. Find peaks (markers)
        if dist.max() < 1:
            return np.zeros((h, w), dtype=np.int32)

        coords = peak_local_max(
            dist,
            min_distance=self.min_distance,
            labels=binary,
        )

        if len(coords) == 0:
            return np.zeros((h, w), dtype=np.int32)

        # Create marker image
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        # Dilate markers slightly for watershed stability
        markers = cv2.dilate(markers.astype(np.uint8), kernel, iterations=1).astype(
            np.int32
        )
        # Re-label after dilation
        _, markers = cv2.connectedComponents((markers > 0).astype(np.uint8))
        markers = markers + 1  # background = 1
        markers[binary == 0] = 0  # unknown stays 0... no, mark sure bg
        # Actually for watershed: 0 = unknown, positive = known regions
        # background region = 1, foreground regions = 2, 3, ...
        bg_markers = np.zeros_like(markers)
        bg_markers[binary == 0] = 1  # sure background
        # Re-do: create proper markers
        markers_proper = np.zeros((h, w), dtype=np.int32)
        markers_proper[binary == 0] = 1  # sure background
        for i, (r, c) in enumerate(coords, start=2):
            markers_proper[r, c] = i

        # 4. Watershed
        img_3ch = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        labels = cv2.watershed(img_3ch, markers_proper.copy())

        return labels

    def _labels_to_contours(self, labels: np.ndarray, shape: tuple) -> List[np.ndarray]:
        """Extract contours from watershed label image, filtering by area."""
        h, w = shape[:2]
        image_area = h * w
        max_area = image_area * self.max_area_ratio

        contours = []
        unique = np.unique(labels)
        for lbl in unique:
            if lbl <= 1:  # 0 = boundary, 1 = background
                continue
            mask = (labels == lbl).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if self.min_area <= area <= max_area:
                    contours.append(c)
        return contours

    def __repr__(self):
        return (
            f"WatershedSegmenter(blur_ksize={self.blur_ksize}, "
            f"min_distance={self.min_distance}, min_area={self.min_area})"
        )


# ====================================================================
# GraphSegmenter
# ====================================================================


class GraphSegmenter(BaseDetector):
    """
    Felzenszwalb graph-based segmentation.

    Uses skimage.segmentation.felzenszwalb to partition the image into
    regions, then extracts bounding boxes and masks.

    Parameters
    ----------
    scale : float
        Free parameter controlling segment size (higher = larger segments).
    sigma : float
        Width of Gaussian kernel for pre-smoothing.
    min_size : int
        Minimum component size after merging.
    min_area : int
        Minimum region area to keep as a detection.
    max_area_ratio : float
        Maximum region area as fraction of image area.
    """

    def __init__(
        self,
        scale: float = 200,
        sigma: float = 0.5,
        min_size: int = 50,
        min_area: int = 80,
        max_area_ratio: float = 0.5,
    ):
        super().__init__(
            {
                "scale": scale,
                "sigma": sigma,
                "min_size": min_size,
                "min_area": min_area,
                "max_area_ratio": max_area_ratio,
            }
        )
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        img = self._load(image)
        if img is None:
            return []
        labels = self._segment(img)
        contours = self._labels_to_contours(labels, img.shape)
        return self._contours_to_boxes(contours)

    def detect_with_masks(self, image):
        img = self._load(image)
        if img is None:
            return [], []
        labels = self._segment(img)
        contours = self._labels_to_contours(labels, img.shape)
        return self._contours_to_boxes(contours), self._contours_to_masks(
            contours, img.shape
        )

    def _segment(self, img: np.ndarray) -> np.ndarray:
        """Run Felzenszwalb segmentation, returns label image."""
        # Convert BGR → RGB for skimage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        labels = felzenszwalb(
            rgb,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size,
        )
        return labels

    def _labels_to_contours(self, labels: np.ndarray, shape: tuple) -> List[np.ndarray]:
        """Extract valid contours from segment labels."""
        h, w = shape[:2]
        image_area = h * w
        max_area = image_area * self.max_area_ratio

        # Find the background label (largest region, usually touching borders)
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]

        contours = []
        for lbl in unique:
            if lbl == bg_label:
                continue
            mask = (labels == lbl).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            if area < self.min_area or area > max_area:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cnts)
        return contours

    def __repr__(self):
        return (
            f"GraphSegmenter(scale={self.scale}, sigma={self.sigma}, "
            f"min_size={self.min_size})"
        )


# ====================================================================
# RetailPriorDetector
# ====================================================================


class RetailPriorDetector(BaseDetector):
    """
    Domain-specific detector for structured retail shelf images.

    Combines Felzenszwalb segmentation with retail-specific priors:
      1. Detect horizontal shelf lines via HoughLinesP
      2. Partition the image into shelf bands between detected lines
      3. Within each band, detect peaks along vertical strips → grid cells
      4. Refine cells with graph segmentation for final detections

    Parameters
    ----------
    hough_threshold : int
        Accumulator threshold for HoughLinesP.
    hough_min_length : int
        Minimum line length for shelf detection.
    hough_max_gap : int
        Maximum gap between line segments.
    angle_tolerance : float
        Tolerance in degrees for horizontal line detection.
    strip_width : int
        Width of vertical strips for peak detection (pixels).
    peak_prominence : float
        Minimum prominence for peak detection in vertical profiles.
    fallback_scale : float
        Felzenszwalb scale for fallback segmentation.
    min_area : int
        Minimum detection area.
    """

    def __init__(
        self,
        hough_threshold: int = 50,
        hough_min_length: int = 80,
        hough_max_gap: int = 20,
        angle_tolerance: float = 5.0,
        strip_width: int = 30,
        peak_prominence: float = 10.0,
        fallback_scale: float = 150,
        min_area: int = 80,
    ):
        super().__init__(
            {
                "hough_threshold": hough_threshold,
                "hough_min_length": hough_min_length,
                "hough_max_gap": hough_max_gap,
                "angle_tolerance": angle_tolerance,
                "strip_width": strip_width,
                "peak_prominence": peak_prominence,
                "fallback_scale": fallback_scale,
                "min_area": min_area,
            }
        )
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.angle_tolerance = angle_tolerance
        self.strip_width = strip_width
        self.peak_prominence = peak_prominence
        self.fallback_scale = fallback_scale
        self.min_area = min_area

        # Internal fallback segmenter
        self._graph_seg = GraphSegmenter(
            scale=fallback_scale, sigma=0.5, min_size=50, min_area=min_area
        )

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        img = self._load(image)
        if img is None:
            return []
        return self._detect_impl(img)

    def detect_with_masks(self, image):
        img = self._load(image)
        if img is None:
            return [], []
        boxes = self._detect_impl(img)
        masks = []
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in boxes:
            m = np.zeros((h, w), dtype=np.uint8)
            m[y1:y2, x1:x2] = 255
            masks.append(m)
        return boxes, masks

    def _detect_impl(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Main detection pipeline."""
        h, w = img.shape[:2]

        # Step 1: Detect horizontal shelf lines
        shelf_lines = self._detect_shelf_lines(img)

        if len(shelf_lines) >= 2:
            # Step 2: Partition into shelf bands
            bands = self._create_shelf_bands(shelf_lines, h)

            # Step 3: Detect objects within each band using grid peaks
            boxes = []
            for band_top, band_bottom in bands:
                band_h = band_bottom - band_top
                if band_h < 10:
                    continue
                band_img = img[band_top:band_bottom, :, :]
                band_boxes = self._detect_in_band(band_img, w, band_top)
                boxes.extend(band_boxes)

            # If grid-prior produced results, use them
            if boxes:
                return boxes

        # Fallback: use graph segmentation directly
        return self._graph_seg.detect(img)

    def _detect_shelf_lines(self, img: np.ndarray) -> List[int]:
        """Detect horizontal shelf lines using Hough transform."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        if lines is None:
            return []

        # Filter near-horizontal lines
        horizontal_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= self.angle_tolerance or angle >= (180 - self.angle_tolerance):
                horizontal_ys.append((y1 + y2) // 2)

        if not horizontal_ys:
            return []

        # Cluster nearby y-values (merge lines within 15px)
        horizontal_ys = sorted(horizontal_ys)
        clustered = [horizontal_ys[0]]
        for y in horizontal_ys[1:]:
            if y - clustered[-1] > 15:
                clustered.append(y)

        return clustered

    def _create_shelf_bands(
        self, lines: List[int], img_height: int
    ) -> List[Tuple[int, int]]:
        """Create shelf bands between detected horizontal lines."""
        boundaries = [0] + lines + [img_height]
        bands = []
        for i in range(len(boundaries) - 1):
            bands.append((boundaries[i], boundaries[i + 1]))
        return bands

    def _detect_in_band(
        self, band_img: np.ndarray, img_width: int, y_offset: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects in a shelf band using vertical strip peak detection.

        Walks vertical strips across the band, computing the intensity
        profile and finding peaks that indicate object boundaries.
        """
        from scipy.signal import find_peaks

        h, w = band_img.shape[:2]
        if h < 5 or w < 5:
            return []

        gray = (
            cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY)
            if len(band_img.shape) == 3
            else band_img
        )

        # Detect vertical edges → indicates product boundaries
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_profile = np.mean(np.abs(sobel_x), axis=0)  # 1D profile along x

        # Find peaks in the vertical-edge profile
        peaks, properties = find_peaks(
            edge_profile,
            distance=self.strip_width // 2,
            prominence=self.peak_prominence,
        )

        if len(peaks) < 2:
            # Fallback: segment the band with graph segmenter
            band_boxes = self._graph_seg.detect(band_img)
            return [
                (x1, y1 + y_offset, x2, y2 + y_offset) for x1, y1, x2, y2 in band_boxes
            ]

        # Create grid cells from peak positions
        boundaries_x = [0] + peaks.tolist() + [w]
        boxes = []
        for i in range(len(boundaries_x) - 1):
            x1 = boundaries_x[i]
            x2 = boundaries_x[i + 1]
            cell_w = x2 - x1
            if cell_w < 10:
                continue
            cell_area = cell_w * h
            if cell_area >= self.min_area:
                boxes.append((int(x1), y_offset, int(x2), y_offset + h))

        return boxes

    def __repr__(self):
        return (
            f"RetailPriorDetector(hough_thresh={self.hough_threshold}, "
            f"strip_width={self.strip_width}, fallback_scale={self.fallback_scale})"
        )


# ====================================================================
# Evaluation Utility (re-exported from heuristic for convenience)
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
    """Evaluate predictions vs ground truth with Hungarian matching."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 and n_gt == 0:
        return {
            "count_error": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou": 1.0,
            "num_predicted": 0,
            "num_gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    if n_pred == 0:
        return {
            "count_error": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "num_predicted": 0,
            "num_gt": n_gt,
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
        }
    if n_gt == 0:
        return {
            "count_error": n_pred,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou": 0.0,
            "num_predicted": n_pred,
            "num_gt": 0,
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
        }

    cost = np.zeros((n_pred, n_gt))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            cost[i, j] = compute_iou(pb, gb)

    row_ind, col_ind = linear_sum_assignment(-cost)

    tp = 0
    matched_ious = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= iou_threshold:
            tp += 1
            matched_ious.append(cost[r, c])

    fp = n_pred - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "count_error": abs(n_pred - n_gt),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(matched_ious)) if matched_ious else 0.0, 4),
        "num_predicted": n_pred,
        "num_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run classical CV detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["watershed", "graph", "retail", "all"],
    )
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    methods = {
        "watershed": WatershedSegmenter(),
        "graph": GraphSegmenter(),
        "retail": RetailPriorDetector(),
    }

    if args.method != "all":
        methods = {args.method: methods[args.method]}

    if args.image:
        for name, det in methods.items():
            boxes = det.detect(args.image)
            print(f"[{name}] Detected {len(boxes)} objects")
    else:
        # Evaluate on synthetic
        ann_path = SYNTHETIC_DIR / "annotations.json"
        if not ann_path.exists():
            print("⚠ Synthetic data not found. Run: python -m src.synthetic_generator")
            sys.exit(1)

        with open(ann_path) as f:
            coco = json.load(f)

        img_map = {img["id"]: img["file_name"] for img in coco["images"]}
        gt_by_image = {}
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            fname = img_map[ann["image_id"]]
            gt_by_image.setdefault(fname, []).append((x, y, x + w, y + h))

        rng = np.random.RandomState(42)
        images = list(img_map.values())
        if len(images) > args.num_samples:
            images = rng.choice(images, args.num_samples, replace=False).tolist()

        for name, det in methods.items():
            print(f"\n{'=' * 50}")
            print(f"  {name.upper()} — {det}")
            print(f"{'=' * 50}")

            all_metrics = []
            for fname in images:
                img_path = str(SYNTHETIC_DIR / "images" / fname)
                gt = gt_by_image.get(fname, [])
                t0 = time.time()
                pred = det.detect(img_path)
                dt = time.time() - t0
                m = evaluate_detections(pred, gt)
                m["time_ms"] = round(dt * 1000, 1)
                all_metrics.append(m)

            avg = {
                k: np.mean([m[k] for m in all_metrics])
                for k in [
                    "count_error",
                    "precision",
                    "recall",
                    "f1",
                    "mean_iou",
                    "time_ms",
                ]
            }
            print(f"  Count MAE: {avg['count_error']:.1f}")
            print(f"  Precision: {avg['precision']:.3f}")
            print(f"  Recall:    {avg['recall']:.3f}")
            print(f"  F1:        {avg['f1']:.3f}")
            print(f"  Mean IoU:  {avg['mean_iou']:.3f}")
            print(f"  Avg time:  {avg['time_ms']:.1f} ms")
