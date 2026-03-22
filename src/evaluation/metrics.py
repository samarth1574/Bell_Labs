"""
metrics.py — Evaluation Metrics
================================
Implements:
  - compute_iou: Intersection over Union for bounding boxes
  - match_boxes: Hungarian matching of predictions to ground truth
  - compute_ap: Average Precision at a given IoU threshold
  - compute_map: Mean Average Precision (mAP@0.5, mAP@0.5:0.95)
  - count_mae: Mean Absolute Error of object counts
  - compute_fps: Frames per second benchmark
  - full_evaluation: Comprehensive evaluation across all metrics

Usage:
    python -m src.evaluation.metrics              # run unit tests
    python -m src.evaluation.metrics --verbose    # detailed test output
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

# ====================================================================
# 1. IoU Computation
# ====================================================================


def compute_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    box_a, box_b : sequence of 4 floats
        Bounding boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    float
        IoU value in [0, 1].

    Examples
    --------
    >>> compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
    0.142857...
    >>> compute_iou([0, 0, 10, 10], [0, 0, 10, 10])
    1.0
    >>> compute_iou([0, 0, 10, 10], [20, 20, 30, 30])
    0.0
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def compute_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise IoU matrix between two sets of boxes.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (M, 4)
    boxes_b : np.ndarray, shape (N, 4)

    Returns
    -------
    np.ndarray, shape (M, N)
        IoU values.
    """
    M = len(boxes_a)
    N = len(boxes_b)
    iou = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            iou[i, j] = compute_iou(boxes_a[i], boxes_b[j])

    return iou


# ====================================================================
# 2. Box Matching (Hungarian)
# ====================================================================


def match_boxes(
    pred_boxes: Union[List, np.ndarray],
    gt_boxes: Union[List, np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Find the optimal one-to-one assignment between predictions and
    ground truths using the Hungarian algorithm.

    Parameters
    ----------
    pred_boxes : array-like, shape (P, 4)
        Predicted bounding boxes [x1, y1, x2, y2].
    gt_boxes : array-like, shape (G, 4)
        Ground truth bounding boxes.
    iou_threshold : float
        Minimum IoU for a valid match (true positive).

    Returns
    -------
    matches : list of (pred_idx, gt_idx, iou)
        Matched pairs with IoU ≥ threshold (true positives).
    unmatched_preds : list of int
        Indices of unmatched predictions (false positives).
    unmatched_gts : list of int
        Indices of unmatched ground truths (false negatives).
    """
    pred_boxes = (
        np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        if len(pred_boxes)
        else np.empty((0, 4))
    )
    gt_boxes = (
        np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        if len(gt_boxes)
        else np.empty((0, 4))
    )

    P = len(pred_boxes)
    G = len(gt_boxes)

    if P == 0 and G == 0:
        return [], [], []
    if P == 0:
        return [], [], list(range(G))
    if G == 0:
        return [], list(range(P)), []

    # Compute IoU matrix
    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)

    # Hungarian assignment (maximise IoU → minimise negative)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    matches = []
    matched_preds = set()
    matched_gts = set()

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matches.append((int(r), int(c), float(iou_mat[r, c])))
            matched_preds.add(r)
            matched_gts.add(c)

    unmatched_preds = [i for i in range(P) if i not in matched_preds]
    unmatched_gts = [i for i in range(G) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


# ====================================================================
# 3. Average Precision (single image or class)
# ====================================================================


def compute_ap(
    pred_boxes: Union[List, np.ndarray],
    pred_scores: Union[List, np.ndarray],
    gt_boxes: Union[List, np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision (AP) at a single IoU threshold.

    Uses the 11-point interpolation method (PASCAL VOC style).

    Parameters
    ----------
    pred_boxes : array-like, shape (P, 4)
    pred_scores : array-like, shape (P,)
    gt_boxes : array-like, shape (G, 4)
    iou_threshold : float

    Returns
    -------
    float
        AP in [0, 1].
    """
    pred_boxes = (
        np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        if len(pred_boxes)
        else np.empty((0, 4))
    )
    pred_scores = (
        np.asarray(pred_scores, dtype=np.float64).ravel()
        if len(pred_scores)
        else np.empty(0)
    )
    gt_boxes = (
        np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        if len(gt_boxes)
        else np.empty((0, 4))
    )

    P = len(pred_boxes)
    G = len(gt_boxes)

    if G == 0:
        return 1.0 if P == 0 else 0.0
    if P == 0:
        return 0.0

    # Sort predictions by score (descending)
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    # IoU matrix
    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)

    # Greedy matching (following PASCAL VOC protocol)
    gt_matched = np.zeros(G, dtype=bool)
    tp = np.zeros(P, dtype=np.float64)
    fp = np.zeros(P, dtype=np.float64)

    for i in range(P):
        ious = iou_mat[i]
        best_gt = np.argmax(ious)
        best_iou = ious[best_gt]

        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            tp[i] = 1.0
            gt_matched[best_gt] = True
        else:
            fp[i] = 1.0

    # Cumulative sums
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / G
    precisions = cum_tp / (cum_tp + cum_fp)

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_r = precisions[recalls >= t]
        ap += p_at_r.max() if len(p_at_r) > 0 else 0.0
    ap /= 11.0

    return float(ap)


# ====================================================================
# 4. Mean Average Precision (across images)
# ====================================================================


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute mean Average Precision across images.

    Parameters
    ----------
    predictions : list of dict
        Each: {"boxes": [[x1,y1,x2,y2], ...], "scores": [s1, s2, ...]}
    ground_truths : list of dict
        Each: {"boxes": [[x1,y1,x2,y2], ...]}
    iou_thresholds : list of float, optional
        IoU thresholds. Default: [0.5] for mAP@0.5.
        For mAP@0.5:0.95, pass np.arange(0.5, 1.0, 0.05).

    Returns
    -------
    dict
        {"mAP@0.5": float, "mAP@0.5:0.95": float, "AP@<t>": float, ...}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # Also compute mAP@0.5:0.95
    coco_thresholds = np.arange(0.5, 1.0, 0.05)

    assert len(predictions) == len(
        ground_truths
    ), f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} GTs"

    results = {}

    # Per-threshold AP
    for t in iou_thresholds:
        aps = []
        for pred, gt in zip(predictions, ground_truths):
            p_boxes = pred.get("boxes", [])
            p_scores = pred.get("scores", [])
            g_boxes = gt.get("boxes", [])
            ap = compute_ap(p_boxes, p_scores, g_boxes, iou_threshold=t)
            aps.append(ap)
        results[f"AP@{t:.2f}"] = float(np.mean(aps)) if aps else 0.0

    # mAP@0.5
    if 0.5 in iou_thresholds or any(abs(t - 0.5) < 1e-6 for t in iou_thresholds):
        results["mAP@0.5"] = results.get("AP@0.50", 0.0)
    else:
        # Compute mAP@0.5 separately
        aps = []
        for pred, gt in zip(predictions, ground_truths):
            ap = compute_ap(
                pred.get("boxes", []),
                pred.get("scores", []),
                gt.get("boxes", []),
                iou_threshold=0.5,
            )
            aps.append(ap)
        results["mAP@0.5"] = float(np.mean(aps))

    # mAP@0.5:0.95
    coco_aps = []
    for t in coco_thresholds:
        key = f"AP@{t:.2f}"
        if key in results:
            coco_aps.append(results[key])
        else:
            aps = []
            for pred, gt in zip(predictions, ground_truths):
                ap = compute_ap(
                    pred.get("boxes", []),
                    pred.get("scores", []),
                    gt.get("boxes", []),
                    iou_threshold=t,
                )
                aps.append(ap)
            val = float(np.mean(aps))
            results[key] = val
            coco_aps.append(val)
    results["mAP@0.5:0.95"] = float(np.mean(coco_aps))

    return results


# ====================================================================
# 5. Count MAE
# ====================================================================


def count_mae(
    pred_counts: Union[List[int], np.ndarray],
    gt_counts: Union[List[int], np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error of object counts.

    Parameters
    ----------
    pred_counts : array-like
        Predicted number of objects per image.
    gt_counts : array-like
        Ground truth number of objects per image.

    Returns
    -------
    float
        Mean absolute error.

    Examples
    --------
    >>> count_mae([5, 10, 3], [4, 12, 3])
    1.333...
    """
    pred = np.asarray(pred_counts, dtype=np.float64)
    gt = np.asarray(gt_counts, dtype=np.float64)

    assert len(pred) == len(gt), f"Length mismatch: {len(pred)} vs {len(gt)}"

    return float(np.mean(np.abs(pred - gt)))


# ====================================================================
# 6. FPS Benchmark
# ====================================================================


def compute_fps(
    model: Any,
    images: List[Any],
    device: str = "cpu",
    n_warmup: int = 3,
    n_runs: int = 10,
    detect_fn: Optional[str] = "detect",
) -> float:
    """
    Benchmark inference speed in frames per second.

    Parameters
    ----------
    model : object
        Any detector with a detect() or __call__() method.
    images : list
        Sample images for benchmarking.
    device : str
        Device string (for display).
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed runs.
    detect_fn : str
        Name of the detection method on model.

    Returns
    -------
    float
        Average frames per second.
    """
    fn = getattr(model, detect_fn, None)
    if fn is None:
        fn = model  # assume callable

    if not images:
        return 0.0

    # Warmup
    for _ in range(n_warmup):
        for img in images[:1]:
            fn(img)

    # Timed runs
    total_frames = 0
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for img in images:
            fn(img)
            total_frames += 1
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    return round(fps, 2)


# ====================================================================
# 7. Full Evaluation
# ====================================================================


def full_evaluation(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run all metrics and return a comprehensive results dictionary.

    Parameters
    ----------
    predictions : list of dict
        Each: {"boxes": [...], "scores": [...]}
    ground_truths : list of dict
        Each: {"boxes": [...]}

    Returns
    -------
    dict with keys:
        mAP@0.5, mAP@0.5:0.95,
        count_mae, count_error_std,
        avg_precision, avg_recall, avg_f1, avg_iou,
        total_images, total_gt_objects, total_pred_objects,
        per_image: list of per-image metrics
    """
    assert len(predictions) == len(ground_truths)
    n = len(predictions)

    # ---- mAP ----
    map_results = compute_map(predictions, ground_truths, iou_thresholds=[0.5])

    # ---- Per-image metrics ----
    per_image = []
    pred_counts_list = []
    gt_counts_list = []
    total_gt = 0
    total_pred = 0

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        p_boxes = pred.get("boxes", [])
#         p_scores = pred.get("scores", [])
        g_boxes = gt.get("boxes", [])

        n_pred = len(p_boxes)
        n_gt = len(g_boxes)
        pred_counts_list.append(n_pred)
        gt_counts_list.append(n_gt)
        total_gt += n_gt
        total_pred += n_pred

        # Match
        matches, unmatched_p, unmatched_g = match_boxes(p_boxes, g_boxes, 0.5)
        tp = len(matches)
        fp = len(unmatched_p)
        fn = len(unmatched_g)

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if n_gt == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if n_gt == 0 else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        avg_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        per_image.append(
            {
                "index": i,
                "num_gt": n_gt,
                "num_pred": n_pred,
                "count_error": abs(n_pred - n_gt),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mean_iou": round(avg_iou, 4),
            }
        )

    # ---- Aggregate ----
    mae = count_mae(pred_counts_list, gt_counts_list)
    count_errors = np.abs(np.array(pred_counts_list) - np.array(gt_counts_list))

    avg_prec = float(np.mean([r["precision"] for r in per_image]))
    avg_rec = float(np.mean([r["recall"] for r in per_image]))
    avg_f1 = float(np.mean([r["f1"] for r in per_image]))
    avg_iou = float(np.mean([r["mean_iou"] for r in per_image]))

    return {
        # mAP
        "mAP@0.5": round(map_results["mAP@0.5"], 4),
        "mAP@0.5:0.95": round(map_results["mAP@0.5:0.95"], 4),
        # Counting
        "count_mae": round(mae, 4),
        "count_error_std": round(float(np.std(count_errors)), 4),
        # Detection quality
        "avg_precision": round(avg_prec, 4),
        "avg_recall": round(avg_rec, 4),
        "avg_f1": round(avg_f1, 4),
        "avg_iou": round(avg_iou, 4),
        # Totals
        "total_images": n,
        "total_gt_objects": total_gt,
        "total_pred_objects": total_pred,
        # Per-image detail
        "per_image": per_image,
    }


# ====================================================================
# Unit Tests
# ====================================================================


def _run_tests(verbose: bool = False):
    """Comprehensive unit tests for all metric functions."""
    print("=" * 60)
    print("  Evaluation Metrics — Unit Tests")
    print("=" * 60)
    passed = 0
    failed = 0

    def _check(name, val, expected, tol=0.01):
        nonlocal passed, failed
        ok = abs(val - expected) < tol
        status = "✓" if ok else "✗"
        if verbose or not ok:
            print(f"  {status} {name}: {val:.4f} (expected {expected:.4f})")
        if ok:
            passed += 1
        else:
            failed += 1

    # ---- 1. IoU ----
    print("\n[1] compute_iou")
    _check("perfect overlap", compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
    _check("no overlap", compute_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
    _check(
        "partial overlap",
        compute_iou([0, 0, 10, 10], [5, 5, 15, 15]),
        25.0 / (100 + 100 - 25),
    )
    _check("zero-area box", compute_iou([5, 5, 5, 5], [0, 0, 10, 10]), 0.0)

    # ---- 2. match_boxes ----
    print("\n[2] match_boxes")
    pred = [[0, 0, 10, 10], [20, 20, 30, 30], [100, 100, 110, 110]]
    gt = [[1, 1, 11, 11], [21, 21, 31, 31]]
    matches, unm_p, unm_g = match_boxes(pred, gt, iou_threshold=0.5)
    _check("2 matches", len(matches), 2)
    _check("1 unmatched pred", len(unm_p), 1)
    _check("0 unmatched gt", len(unm_g), 0)

    # Empty cases
    m, up, ug = match_boxes([], [[0, 0, 10, 10]], 0.5)
    _check("empty pred → 0 matches", len(m), 0)
    _check("empty pred → 1 unmatched gt", len(ug), 1)

    m, up, ug = match_boxes([[0, 0, 10, 10]], [], 0.5)
    _check("empty gt → 1 unmatched pred", len(up), 1)

    # ---- 3. compute_ap ----
    print("\n[3] compute_ap")
    # Perfect detection
    ap = compute_ap(
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        [0.9, 0.8],
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        iou_threshold=0.5,
    )
    _check("perfect AP", ap, 1.0)

    # No predictions
    ap = compute_ap([], [], [[0, 0, 10, 10]], 0.5)
    _check("no predictions AP", ap, 0.0)

    # No GT
    ap = compute_ap([], [], [], 0.5)
    _check("no GT no pred AP", ap, 1.0)

    # ---- 4. compute_map ----
    print("\n[4] compute_map")
    preds = [{"boxes": [[0, 0, 10, 10]], "scores": [0.9]}]
    gts = [{"boxes": [[0, 0, 10, 10]]}]
    results = compute_map(preds, gts)
    _check("mAP@0.5 perfect", results["mAP@0.5"], 1.0)
    if verbose:
        print(f"    mAP@0.5:0.95 = {results['mAP@0.5:0.95']:.4f}")

    # ---- 5. count_mae ----
    print("\n[5] count_mae")
    _check("exact counts", count_mae([5, 10, 3], [5, 10, 3]), 0.0)
    _check("off by 1", count_mae([5, 10, 3], [4, 11, 2]), 1.0)
    _check("mixed errors", count_mae([5, 10, 3], [4, 12, 3]), 1.0, tol=0.4)

    # ---- 6. full_evaluation ----
    print("\n[6] full_evaluation")
    preds = [
        {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "scores": [0.9, 0.8]},
        {"boxes": [[50, 50, 60, 60]], "scores": [0.7]},
    ]
    gts = [
        {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]]},
        {"boxes": [[50, 50, 60, 60], [70, 70, 80, 80]]},
    ]
    result = full_evaluation(preds, gts)
    _check("full_eval precision", result["avg_precision"], 1.0)
    _check("full_eval count_mae", result["count_mae"], 0.5)
    _check("full_eval total_images", result["total_images"], 2)
    if verbose:
        print(f"    Full result keys: {list(result.keys())}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")
    return failed == 0


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation metrics tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    success = _run_tests(verbose=args.verbose)
    if not success:
        exit(1)
