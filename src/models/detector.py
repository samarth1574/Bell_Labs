"""
detector.py — Dense Object Detector Wrapper
============================================
Wraps pretrained torchvision detectors (Faster R-CNN, FCOS) and
replaces their built-in NMS with our Soft-NMS implementation.

Supports backbones: ResNet50, MobileNetV3-Large

Key Innovation
--------------
Instead of using the model's built-in Hard NMS post-processing, we
intercept the raw detections and apply Soft-NMS (Gaussian or Linear
score decay). This preserves detections of genuinely distinct but
overlapping objects that Hard NMS would discard — critical for dense
scenes with 1–50 heavily overlapping objects.

Strategy for intercepting pre-NMS detections:
  - Lower the model's built-in score_thresh and nms_thresh to
    permissive values so almost all proposals survive.
  - Apply our Soft-NMS on the raw outputs.

Usage:
    python -m src.models.detector --image path/to/img.jpg --nms soft_gaussian
    python -m src.models.detector --image path/to/img.jpg --nms hard --backbone resnet50
    python -m src.models.detector --image path/to/img.jpg --benchmark
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from src.models.soft_nms import soft_nms
from src.models.config import DetectorConfig
from src.models.density_head import add_density_head_to_backbone

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ====================================================================
# NMS Method Constants
# ====================================================================

NMS_METHODS = {
    "soft_gaussian": "gaussian",
    "soft_linear": "linear",
    "hard": "hard",
}


# ====================================================================
# DenseObjectDetector
# ====================================================================


class DenseObjectDetector:
    """
    Pretrained object detector with configurable NMS post-processing.

    Wraps torchvision Faster R-CNN models and replaces the built-in
    Hard NMS with our Soft-NMS for improved dense-scene detection.

    Parameters
    ----------
    backbone : str
        'mobilenet_v3' or 'resnet50'.
    nms_method : str
        'soft_gaussian', 'soft_linear', or 'hard'.
    sigma : float
        Gaussian decay parameter (only for soft_gaussian).
    iou_threshold : float
        IoU threshold for linear/hard NMS.
    score_thresh : float
        Minimum final score to keep a detection.
    device : str
        'cpu' or 'cuda'.
    """

    # Permissive settings to pass through almost all proposals
    _RAW_SCORE_THRESH = 0.01
    _RAW_NMS_THRESH = 0.95  # nearly disable built-in NMS
    _RAW_DETECTIONS_PER_IMG = 500

    def __init__(
        self,
        config_path: Optional[str] = None,
        backbone: str = "mobilenet_v3",
        nms_method: str = "soft_gaussian",
        sigma: float = 0.5,
        iou_threshold: float = 0.5,
        score_thresh: float = 0.3,
        device: str = "cpu",
    ):
        if config_path:
            self.config = DetectorConfig.from_yaml(config_path)
            self.backbone = self.config.model.backbone
            self.nms_method = self.config.nms.method
            self.sigma = self.config.nms.sigma
            self.iou_threshold = self.config.nms.iou_threshold
            self.score_thresh = self.config.nms.score_thresh
            self.use_density_head = self.config.model.use_density_head
            self.use_custom_anchors = self.config.anchors.use_custom_dense_anchors
        else:
            self.config = None
            self.backbone = backbone
            self.nms_method = nms_method
            self.sigma = sigma
            self.iou_threshold = iou_threshold
            self.score_thresh = score_thresh
            self.use_density_head = False
            self.use_custom_anchors = False

        self.device = torch.device(device)
        self._soft_method = NMS_METHODS.get(self.nms_method, "gaussian")

        self.model = None
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # ----------------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load a pretrained Faster R-CNN from torchvision.

        The model's internal NMS parameters are set to permissive values
        so that Soft-NMS can operate on the nearly-complete set of raw
        proposals.
        """
        print(f"[detector] Loading Faster R-CNN ({self.backbone})…")

        if self.backbone == "resnet50":
            weights = (
                torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights,
            )
        elif self.backbone in ("mobilenet_v3", "mobilenet_v3_large"):
            weights = (
                torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
            )
        else:
            raise ValueError(
                f"Unknown backbone '{self.backbone}'. "
                f"Choose from: 'resnet50', 'mobilenet_v3'"
            )

        if self.use_custom_anchors:
            print("[detector] Applying custom small anchor sizes for dense objects...")
            from torchvision.models.detection.anchor_utils import AnchorGenerator
            # Match standard torchvision anchor generator parameter format
            # 5 features levels -> 5 tuples of sizes
            anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            self.model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        if self.use_density_head:
            print("[detector] Equipping lightweight Density Head...")
            # We wrap the backbone so that during inference/train it internally yields a count count mapping
            # which does not disrupt the usual forward pass to the FPN/RPN.
            self.model.backbone = add_density_head_to_backbone(self.model.backbone)

        # Override built-in NMS to be permissive
        # This lets our Soft-NMS handle the suppression
        self.model.roi_heads.score_thresh = self._RAW_SCORE_THRESH
        self.model.roi_heads.nms_thresh = self._RAW_NMS_THRESH
        self.model.roi_heads.detections_per_img = self._RAW_DETECTIONS_PER_IMG

        self.model.to(self.device)
        self.model.eval()
        print(f"[detector] ✓ Model loaded on {self.device}")

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self,
        image,
        return_raw: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run detection on a single image.

        Parameters
        ----------
        image : str, Path, PIL.Image, or Tensor
            Input image.
        return_raw : bool
            If True, also return pre-NMS detections.

        Returns
        -------
        boxes : Tensor (K, 4)
            Final bounding boxes [x1, y1, x2, y2].
        scores : Tensor (K,)
            Final confidence scores (post Soft-NMS decay).
        labels : Tensor (K,)
            Class labels (COCO class IDs).
        """
        if self.model is None:
            self.load_model()

        img_tensor = self._prepare_image(image)
        img_tensor = img_tensor.to(self.device)

        # Raw inference (with permissive NMS)
        outputs = self.model([img_tensor])[0]
        raw_boxes = outputs["boxes"].cpu()
        raw_scores = outputs["scores"].cpu()
        raw_labels = outputs["labels"].cpu()

        if len(raw_boxes) == 0:
            empty = torch.tensor([])
            return empty.reshape(0, 4), empty, empty.long()

        # Apply our NMS
        keep, new_scores = soft_nms(
            raw_boxes,
            raw_scores,
            sigma=self.sigma,
            score_threshold=self.score_thresh,
            iou_threshold=self.iou_threshold,
            method=self._soft_method,
        )

        if len(keep) == 0:
            empty = torch.tensor([])
            return empty.reshape(0, 4), empty, empty.long()

        final_boxes = raw_boxes[keep]
        final_labels = raw_labels[keep]

        return final_boxes, new_scores, final_labels

    @torch.no_grad()
    def detect_batch(
        self,
        images: List,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run detection on a batch of images.

        Parameters
        ----------
        images : list
            List of images (paths, PIL images, or tensors).

        Returns
        -------
        list of (boxes, scores, labels) tuples.
        """
        if self.model is None:
            self.load_model()

        tensors = [self._prepare_image(img).to(self.device) for img in images]
        raw_outputs = self.model(tensors)

        results = []
        for output in raw_outputs:
            raw_boxes = output["boxes"].cpu()
            raw_scores = output["scores"].cpu()
            raw_labels = output["labels"].cpu()

            if len(raw_boxes) == 0:
                empty = torch.tensor([])
                results.append((empty.reshape(0, 4), empty, empty.long()))
                continue

            keep, new_scores = soft_nms(
                raw_boxes,
                raw_scores,
                sigma=self.sigma,
                score_threshold=self.score_thresh,
                iou_threshold=self.iou_threshold,
                method=self._soft_method,
            )

            if len(keep) == 0:
                empty = torch.tensor([])
                results.append((empty.reshape(0, 4), empty, empty.long()))
            else:
                results.append((raw_boxes[keep], new_scores, raw_labels[keep]))

        return results

    # ----------------------------------------------------------------
    # Benchmarking
    # ----------------------------------------------------------------

    @torch.no_grad()
    def benchmark(
        self,
        image,
        n_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark inference speed.

        Parameters
        ----------
        image : input image
        n_runs : int
            Number of runs for timing.

        Returns
        -------
        dict with: avg_time_ms, fps, num_detections, device, backbone, nms_method
        """
        if self.model is None:
            self.load_model()

        img_tensor = self._prepare_image(image).to(self.device)

        # Warmup
        for _ in range(3):
            self.model([img_tensor])

        # Timed runs
        times = []
        n_det = 0
        for _ in range(n_runs):
            t0 = time.perf_counter()
            boxes, scores, labels = self.detect(img_tensor)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_det = len(boxes)

        avg_ms = np.mean(times) * 1000
        fps = 1000 / avg_ms if avg_ms > 0 else 0

        return {
            "avg_time_ms": round(avg_ms, 2),
            "fps": round(fps, 1),
            "num_detections": n_det,
            "device": str(self.device),
            "backbone": self.backbone,
            "nms_method": self.nms_method,
            "sigma": self.sigma,
            "n_runs": n_runs,
        }

    # ----------------------------------------------------------------
    # Image Preparation
    # ----------------------------------------------------------------

    def _prepare_image(self, image) -> torch.Tensor:
        """Convert various input types to a [C, H, W] float tensor."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                return image.float()
            raise ValueError(f"Expected 3D tensor, got shape {image.shape}")

        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")

        if isinstance(image, Image.Image):
            return self._transform(image)

        if isinstance(image, np.ndarray):
            # Assume BGR (OpenCV) or RGB — convert to tensor
            if image.shape[2] == 3:
                return torch.from_numpy(
                    image.transpose(2, 0, 1).astype(np.float32) / 255.0
                )

        raise TypeError(f"Cannot prepare image of type {type(image)}")

    # ----------------------------------------------------------------
    # Repr
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DenseObjectDetector(backbone='{self.backbone}', "
            f"nms='{self.nms_method}', σ={self.sigma}, "
            f"score_thresh={self.score_thresh}, device='{self.device}')"
        )


# ====================================================================
# Visualisation Helper
# ====================================================================


def visualize_detections(
    image_path: str,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    output_path: Optional[str] = None,
) -> None:
    """Draw detection boxes on an image and display/save."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read {image_path}")
        return

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        score = scores[i].item()
        label = labels[i].item()

        color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"[detector] Saved visualisation → {output_path}")
    else:
        print(f"[detector] Detected {len(boxes)} objects (pass --output to save)")


# ====================================================================
# CLI
# ====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Dense Object Detector with Soft-NMS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3",
        choices=["resnet50", "mobilenet_v3"],
    )
    parser.add_argument(
        "--nms",
        type=str,
        default="soft_gaussian",
        choices=["soft_gaussian", "soft_linear", "hard"],
    )
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualisation to reports/figures/",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config (e.g. configs/dl_default.yaml)"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument(
        "--n_runs", type=int, default=10, help="Number of benchmark runs"
    )
    args = parser.parse_args()

    detector = DenseObjectDetector(
        config_path=args.config,
        backbone=args.backbone,
        nms_method=args.nms,
        sigma=args.sigma,
        score_thresh=args.score_thresh,
        device=args.device,
    )
    print(f"\n{detector}\n")

    boxes, scores, labels = detector.detect(args.image)
    print(f"Detected {len(boxes)} objects")

    if len(boxes) > 0:
        print(f"\n{'#':>3}  {'Score':>6}  {'Label':>5}  {'Box (x1,y1,x2,y2)'}")
        print("─" * 50)
        for i in range(min(len(boxes), 20)):
            b = boxes[i].int().tolist()
            print(f"{i + 1:>3}  {scores[i]:.3f}  {labels[i]:>5}  {b}")
        if len(boxes) > 20:
            print(f"  … and {len(boxes) - 20} more")

    if args.visualize:
        out_path = PROJECT_ROOT / "reports" / "figures" / "detector_output.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_detections(args.image, boxes, scores, labels, str(out_path))

    if args.benchmark:
        print(f"\nBenchmarking ({args.n_runs} runs)…")
        bench = detector.benchmark(args.image, n_runs=args.n_runs)
        print(f"  Avg time: {bench['avg_time_ms']:.1f} ms")
        print(f"  FPS:      {bench['fps']:.1f}")
        print(f"  Device:   {bench['device']}")
        print()


if __name__ == "__main__":
    main()
