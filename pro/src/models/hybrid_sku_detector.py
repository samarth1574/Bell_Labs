"""Phase 3 hybrid detector for dense retail shelves."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import full_evaluation
from src.models.soft_nms import soft_nms_np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class HybridConfig:
    weights: str = "yolo11l.pt"
    score_threshold: float = 0.08
    iou_threshold: float = 0.5
    sigma: float = 0.5
    score_alpha: float = 0.70
    density_beta: float = 0.20
    count_gamma: float = 0.10
    count_tolerance: float = 0.20
    max_detections: int = 300


class ClassicalDensityPrior:
    """Classical CV prior that estimates object count and local box support."""

    def __init__(self, min_area: int = 18):
        self.min_area = min_area

    def estimate(self, image: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        distance_ready = cv2.dilate(closed, kernel, iterations=1)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(distance_ready, connectivity=8)
        boxes: list[list[float]] = []
        areas: list[int] = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area >= self.min_area and w > 2 and h > 2:
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
                areas.append(int(area))

        density_map = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (9, 9), 0)
        return {
            "count": int(len(boxes)),
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            "scores": self._area_scores(areas),
            "density_map": density_map,
        }

    @staticmethod
    def _area_scores(areas: Iterable[int]) -> np.ndarray:
        areas = np.asarray(list(areas), dtype=np.float32)
        if len(areas) == 0:
            return np.empty(0, dtype=np.float32)
        scaled = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)
        return (0.35 + 0.55 * scaled).astype(np.float32)

    @staticmethod
    def local_support(density_map: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        support = []
        height, width = density_map.shape[:2]
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                support.append(0.0)
                continue
            patch = density_map[y1:y2, x1:x2]
            support.append(float(patch.mean()) if patch.size else 0.0)
        values = np.asarray(support, dtype=np.float32)
        if len(values) == 0:
            return values
        return values / (values.max() + 1e-6)


class HybridSkuDetector:
    """YOLO11 plus classical density/count calibration for SKU-110K."""

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self.prior = ClassicalDensityPrior()
        self._model = None

    def _load_yolo(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for neural inference") from exc
            self._model = YOLO(self.config.weights)
        return self._model

    def predict(self, image_path: str | Path, mode: str = "hybrid") -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        prior = self.prior.estimate(image)
        if mode == "ml_only":
            boxes, scores = prior["boxes"], prior["scores"]
        else:
            boxes, scores = self._predict_dl(image_path)
            if mode in {"dl_soft_nms", "hybrid"}:
                boxes, scores = self._apply_soft_nms(boxes, scores)
            if mode == "hybrid":
                scores = self._hybrid_scores(boxes, scores, prior)
                boxes, scores = self._apply_soft_nms(boxes, scores)
                boxes, scores = self._count_aware_select(boxes, scores, prior["count"])

        return {
            "boxes": boxes.astype(float).tolist(),
            "scores": scores.astype(float).tolist(),
            "count": int(len(boxes)),
            "mode": mode,
            "classical_count": int(prior["count"]),
        }

    def _predict_dl(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        model = self._load_yolo()
        result = model.predict(
            source=str(image_path),
            conf=self.config.score_threshold,
            verbose=False,
            max_det=self.config.max_detections,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        return boxes, scores

    def _hybrid_scores(self, boxes: np.ndarray, scores: np.ndarray, prior: dict[str, Any]) -> np.ndarray:
        if len(scores) == 0:
            return scores
        local = self.prior.local_support(prior["density_map"], boxes)
        dl_count = max(len(scores), 1)
        count_gap = abs(prior["count"] - dl_count) / max(prior["count"], dl_count, 1)
        count_factor = max(0.0, 1.0 - count_gap)
        mixed = (
            self.config.score_alpha * scores
            + self.config.density_beta * local
            + self.config.count_gamma * count_factor
        )
        return np.clip(mixed, 0.0, 1.0).astype(np.float32)

    def _apply_soft_nms(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return boxes, scores
        keep, new_scores = soft_nms_np(
            boxes.copy(),
            scores.copy(),
            sigma=self.config.sigma,
            iou_threshold=self.config.iou_threshold,
            score_threshold=self.config.score_threshold,
            method="gaussian",
        )
        return boxes[keep], new_scores.astype(np.float32)

    def _count_aware_select(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classical_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(scores) == 0 or classical_count <= 0:
            return boxes, scores
        lower = int(classical_count * (1.0 - self.config.count_tolerance))
        upper = int(classical_count * (1.0 + self.config.count_tolerance))
        target_max = min(max(upper, lower, 1), self.config.max_detections)
        order = np.argsort(-scores)
        keep = order[: min(len(order), target_max)]
        keep = keep[scores[keep] >= self.config.score_threshold]
        if len(keep) < lower:
            keep = order[: min(len(order), lower)]
        return boxes[keep], scores[keep]


def evaluate_dataset(
    detector: HybridSkuDetector,
    images_dir: Path,
    labels_dir: Path,
    mode: str,
    limit: int | None = None,
) -> dict[str, float]:
    image_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if limit:
        image_paths = image_paths[:limit]

    predictions = []
    ground_truths = []
    for image_path in image_paths:
        pred = detector.predict(image_path, mode=mode)
        predictions.append({"boxes": pred["boxes"], "scores": pred["scores"]})
        ground_truths.append({"boxes": read_yolo_label(labels_dir / f"{image_path.stem}.txt", image_path)})
    return full_evaluation(predictions, ground_truths)


def read_yolo_label(label_path: Path, image_path: Path) -> list[list[float]]:
    image = cv2.imread(str(image_path))
    if image is None or not label_path.exists():
        return []
    height, width = image.shape[:2]
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 hybrid SKU detector and ablation evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, default="yolo11l.pt")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data/sku110k/images/test")
    parser.add_argument("--labels-dir", type=Path, default=PROJECT_ROOT / "data/sku110k/labels/test")
    parser.add_argument("--mode", choices=["ml_only", "dl_only", "dl_soft_nms", "hybrid", "all"], default="all")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/phase3_ablation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = HybridSkuDetector(HybridConfig(weights=args.weights))

    if args.image:
        modes = ["hybrid"] if args.mode == "all" else [args.mode]
        for mode in modes:
            print(json.dumps(detector.predict(args.image, mode=mode), indent=2))
        return

    modes = ["ml_only", "dl_only", "dl_soft_nms", "hybrid"] if args.mode == "all" else [args.mode]
    rows = []
    for mode in modes:
        metrics = evaluate_dataset(detector, args.images_dir, args.labels_dir, mode=mode, limit=args.limit)
        rows.append({"mode": mode, **metrics, **asdict(detector.config)})
        print(f"[hybrid] {mode}: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[hybrid] Wrote ablation table to {args.output}")


if __name__ == "__main__":
    main()
