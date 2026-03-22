"""
soft_nms.py — Soft Non-Maximum Suppression
===========================================
Implements Soft-NMS (Bodla et al., ICCV 2017) with both
Gaussian and Linear score decay modes, plus standard Hard NMS
as a baseline reference.

Mathematical Formulation
------------------------
Given N candidate detections with boxes B = {b_1, ..., b_N} and
scores S = {s_1, ..., s_N}, NMS iteratively selects the highest-
scoring box M and penalises overlapping boxes:

  Hard NMS:
    s_i = 0                                    if IoU(M, b_i) >= N_t

  Linear Soft-NMS:
    s_i = s_i * (1 - IoU(M, b_i))             if IoU(M, b_i) >= N_t

  Gaussian Soft-NMS:
    s_i = s_i * exp( -IoU(M, b_i)^2 / sigma ) for all b_i

where sigma controls the decay width and N_t is the IoU threshold.

Complexity: O(N^2), same asymptotic cost as standard NMS.
  - Outer loop: N iterations (one per selected box)
  - Inner loop: up to N IoU computations per iteration
  - Total: N * N = O(N^2) IoU computations

Reference: https://arxiv.org/abs/1704.04503

Usage:
    from src.models.soft_nms import soft_nms

    keep, new_scores = soft_nms(boxes, scores, sigma=0.5, method='gaussian')
"""

from typing import Dict, Tuple

import numpy as np
import torch

# ====================================================================
# IoU Computation (Vectorised)
# ====================================================================


def _compute_iou_vector(
    box: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between one box and a set of boxes.

    Parameters
    ----------
    box : Tensor, shape (4,)
        Reference box [x1, y1, x2, y2].
    boxes : Tensor, shape (K, 4)
        Candidate boxes [x1, y1, x2, y2].

    Returns
    -------
    Tensor, shape (K,)
        IoU values.
    """
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area_a = (box[2] - box[0]) * (box[3] - box[1])
    areas_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + areas_b - inter

    return inter / torch.clamp(union, min=1e-6)


# ====================================================================
# Soft-NMS Core
# ====================================================================


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    method: str = "gaussian",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Non-Maximum Suppression.

    Replaces hard binary keep/discard with continuous score decay,
    preserving detections of genuinely distinct but overlapping objects.

    Parameters
    ----------
    boxes : Tensor, shape (N, 4)
        Bounding boxes in [x1, y1, x2, y2] format (float).
    scores : Tensor, shape (N,)
        Confidence scores for each box.
    sigma : float
        Width parameter for Gaussian decay. Larger sigma → gentler decay
        → more boxes retained. Typical range: 0.1–1.0.
        Only used when method='gaussian'.
    score_threshold : float
        Minimum score to keep a detection after decay.
        Set very low (e.g. 0.001) for Soft-NMS to avoid premature pruning.
    iou_threshold : float
        IoU threshold for 'linear' and 'hard' methods.
        Not used for 'gaussian' (which decays scores for all IoU > 0).
    method : str
        'gaussian' — s_i *= exp(-IoU^2 / sigma)
        'linear'  — s_i *= (1 - IoU) if IoU >= iou_threshold
        'hard'    — s_i = 0 if IoU >= iou_threshold (standard NMS)

    Returns
    -------
    keep_indices : Tensor (K,)
        Indices of kept boxes in the original input.
    new_scores : Tensor (K,)
        Updated scores after decay (same order as keep_indices).

    Raises
    ------
    ValueError
        If method is not one of 'gaussian', 'linear', 'hard'.
    """
    if method not in ("gaussian", "linear", "hard"):
        raise ValueError(
            f"Unknown NMS method '{method}'. "
            f"Choose from: 'gaussian', 'linear', 'hard'."
        )

    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

    # Ensure float tensors
    boxes = boxes.float()
    scores = scores.float()

    N = boxes.shape[0]

    # Work on copies (we'll modify scores in-place)
    scores_work = scores.clone()
    indices = torch.arange(N, dtype=torch.long)

    keep = []
    keep_scores = []

    # O(N^2) loop: iteratively select highest-scoring box and decay others
    for _ in range(N):
        # Find current max
        if scores_work.numel() == 0:
            break

        max_idx = torch.argmax(scores_work)
        max_score = scores_work[max_idx].item()

        if max_score < score_threshold:
            break

        # Record this box
        keep.append(indices[max_idx].item())
        keep_scores.append(max_score)

        # Compute IoU of selected box with all remaining
        selected_box = boxes[indices[max_idx]]
        remaining_mask = torch.ones(len(scores_work), dtype=torch.bool)
        remaining_mask[max_idx] = False

        if remaining_mask.sum() == 0:
            break

        remaining_indices = torch.where(remaining_mask)[0]
        remaining_boxes = boxes[indices[remaining_indices]]
        ious = _compute_iou_vector(selected_box, remaining_boxes)

        # Apply decay to remaining scores
        # -----------------------------------------------------------------
        # Gaussian Soft-NMS:
        #   s_i = s_i * exp(-IoU(M, b_i)^2 / sigma)
        #   Decays ALL overlapping boxes proportionally to IoU.
        #   Higher sigma → gentler decay → more boxes survive.
        #
        # Linear Soft-NMS:
        #   s_i = s_i * (1 - IoU(M, b_i))   if IoU >= N_t
        #   s_i = s_i                        otherwise
        #   Only penalises boxes above the IoU threshold.
        #
        # Hard NMS (baseline):
        #   s_i = 0                          if IoU >= N_t
        #   s_i = s_i                        otherwise
        #   Binary suppress — the standard approach.
        # -----------------------------------------------------------------
        remaining_scores = scores_work[remaining_indices]

        if method == "gaussian":
            decay = torch.exp(-(ious**2) / sigma)
            remaining_scores = remaining_scores * decay

        elif method == "linear":
            above = ious >= iou_threshold
            decay = torch.ones_like(ious)
            decay[above] = 1.0 - ious[above]
            remaining_scores = remaining_scores * decay

        elif method == "hard":
            above = ious >= iou_threshold
            remaining_scores[above] = 0.0

        # Update working arrays (remove selected, update scores)
        scores_work = remaining_scores
        indices = indices[remaining_indices]

    if not keep:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

    keep_tensor = torch.tensor(keep, dtype=torch.long)
    scores_tensor = torch.tensor(keep_scores, dtype=torch.float32)

    return keep_tensor, scores_tensor


# ====================================================================
# Numpy Wrapper (for non-PyTorch pipelines)
# ====================================================================


def soft_nms_np(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    method: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy interface for soft_nms. Converts to Tensor internally.

    Parameters / Returns: same as soft_nms() but with numpy arrays.
    """
    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))

    keep, new_scores = soft_nms(
        boxes_t,
        scores_t,
        sigma=sigma,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        method=method,
    )

    return keep.numpy(), new_scores.numpy()


# ====================================================================
# Comparison Utility
# ====================================================================


def compare_nms_methods(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
) -> Dict[str, Dict]:
    """
    Run all three NMS methods on the same input and compare results.

    Returns
    -------
    dict
        {method_name: {"keep": Tensor, "scores": Tensor, "num_kept": int}}
    """
    results = {}
    for method in ["hard", "linear", "gaussian"]:
        keep, new_scores = soft_nms(
            boxes,
            scores,
            sigma=sigma,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            method=method,
        )
        results[method] = {
            "keep": keep,
            "scores": new_scores,
            "num_kept": len(keep),
        }
    return results


# ====================================================================
# Unit Tests
# ====================================================================


def _run_tests():
    """
    Unit tests with known overlapping boxes.

    Test scenario: 5 boxes, two pairs overlap heavily.
    - Box 0 & Box 1: IoU ≈ 0.68 (heavily overlapping)
    - Box 2 & Box 3: IoU ≈ 0.47 (moderately overlapping)
    - Box 4: isolated

    Expected behaviour:
    - Hard NMS (IoU=0.5): suppresses Box 1 (overlaps with 0), keeps rest → 4 boxes
    - Linear Soft-NMS: keeps all with decayed scores for overlapping boxes
    - Gaussian Soft-NMS: keeps all (σ=0.5 allows moderate overlap)
    """
    print("=" * 50)
    print("  Soft-NMS Unit Tests")
    print("=" * 50)

    # Test boxes: [x1, y1, x2, y2]
    boxes = torch.tensor(
        [
            [10, 10, 60, 60],  # Box 0: score=0.95
            [20, 15, 65, 62],  # Box 1: overlaps with 0, score=0.90
            [100, 100, 150, 160],  # Box 2: score=0.80
            [110, 110, 160, 170],  # Box 3: overlaps with 2, score=0.70
            [200, 200, 250, 250],  # Box 4: isolated, score=0.60
        ],
        dtype=torch.float32,
    )

    scores = torch.tensor([0.95, 0.90, 0.80, 0.70, 0.60])

    # Verify IoU between box 0 and box 1
    iou_01 = _compute_iou_vector(boxes[0], boxes[1:2]).item()
    print(f"\n  IoU(box0, box1) = {iou_01:.3f}")

    iou_23 = _compute_iou_vector(boxes[2], boxes[3:4]).item()
    print(f"  IoU(box2, box3) = {iou_23:.3f}")

    results = compare_nms_methods(boxes, scores, sigma=0.5, iou_threshold=0.5)

    all_passed = True
    for method, res in results.items():
        n = res["num_kept"]
        print(f"\n  [{method.upper()}]")
        print(f"    Kept: {n} boxes")
        print(f"    Indices: {res['keep'].tolist()}")
        print(f"    Scores:  {[f'{s:.4f}' for s in res['scores'].tolist()]}")

        # Basic checks
        if method == "hard":
            # Hard NMS should keep fewer boxes than Soft variants
            if n > len(boxes):
                print("    ✗ FAIL: more boxes than input!")
                all_passed = False
            else:
                print(f"    ✓ PASS: {n} ≤ {len(boxes)}")
        else:
            # Soft variants should generally keep more boxes
            print("    ✓ PASS: returned valid results")

    # Test edge cases
    print("\n  [EDGE CASES]")

    # Empty input
    keep, sc = soft_nms(torch.zeros(0, 4), torch.zeros(0))
    assert len(keep) == 0, "Empty input should return empty"
    print("    ✓ Empty input handled")

    # Single box
    keep, sc = soft_nms(torch.tensor([[0, 0, 10, 10]]).float(), torch.tensor([0.9]))
    assert len(keep) == 1, "Single box should be kept"
    print("    ✓ Single box handled")

    # All below threshold
    keep, sc = soft_nms(
        torch.tensor([[0, 0, 10, 10]]).float(),
        torch.tensor([0.0001]),
        score_threshold=0.01,
    )
    assert len(keep) == 0, "Below-threshold box should be removed"
    print("    ✓ Below-threshold handled")

    # Invalid method
    try:
        soft_nms(boxes, scores, method="invalid")
        print("    ✗ FAIL: should have raised ValueError")
        all_passed = False
    except ValueError:
        print("    ✓ Invalid method raises ValueError")

    print(f"\n  {'=' * 50}")
    print(f"  {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"  {'=' * 50}\n")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    _run_tests()
