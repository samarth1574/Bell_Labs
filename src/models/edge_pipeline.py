"""
edge_pipeline.py — Hybrid Edge Inference Pipeline
===================================================
Implements a two-tier inference pipeline:
  - EdgeInferencePipeline: lightweight ONNX model on edge device
  - ServerRefinementPipeline: full-model refinement on server

The edge pipeline exports a MobileNetV3-based detector to ONNX,
runs inference via ONNX Runtime, applies Soft-NMS post-processing,
and compresses results for server transmission.

Usage:
    python -m src.models.edge_pipeline                       # full demo
    python -m src.models.edge_pipeline --export-only         # export ONNX
    python -m src.models.edge_pipeline --benchmark           # benchmark
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


# ====================================================================
# EdgeInferencePipeline
# ====================================================================

class EdgeInferencePipeline:
    """
    Lightweight edge inference using ONNX Runtime + Soft-NMS.

    Designed for deployment on edge devices (Jetson Nano, RPi, mobile)
    with INT8/FP16 quantisation support.

    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file.
    nms_method : str
        'soft_gaussian', 'soft_linear', or 'hard'.
    sigma : float
        Soft-NMS Gaussian decay parameter.
    score_threshold : float
        Minimum score to keep after NMS.
    input_size : tuple of (int, int)
        Model input size (H, W).
    """

    def __init__(
        self,
        model_path: str = "model.onnx",
        nms_method: str = "soft_gaussian",
        sigma: float = 0.5,
        score_threshold: float = 0.3,
        input_size: Tuple[int, int] = (320, 320),
    ):
        self.model_path = Path(model_path)
        self.nms_method = nms_method
        self.sigma = sigma
        self.score_threshold = score_threshold
        self.input_size = input_size
        self.session = None

        # Map NMS method names to soft_nms function params
        self._method_map = {
            "soft_gaussian": "gaussian",
            "soft_linear": "linear",
            "hard": "hard",
        }

    # ----------------------------------------------------------------
    # ONNX Export
    # ----------------------------------------------------------------

    @staticmethod
    def export_to_onnx(
        pytorch_model,
        output_path: str,
        input_size: Tuple[int, int] = (320, 320),
        opset_version: int = 11,
    ) -> str:
        """
        Export a torchvision detection model to ONNX format.

        Parameters
        ----------
        pytorch_model : torch.nn.Module
            Trained or pretrained detection model.
        output_path : str
            Where to save the .onnx file.
        input_size : tuple
            (H, W) input dimensions.
        opset_version : int
            ONNX opset version.

        Returns
        -------
        str
            Path to the saved ONNX file.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device

        # Dummy input
        dummy = torch.randn(1, 3, *input_size).to(device)

        print(f"[edge] Exporting to ONNX: {output_path}")
        print(f"       Input size: {input_size}, Opset: {opset_version}")

        torch.onnx.export(
            pytorch_model,
            (dummy,),
            str(output_path),
            opset_version=opset_version,
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "images": {0: "batch"},
                "boxes": {0: "num_detections"},
                "labels": {0: "num_detections"},
                "scores": {0: "num_detections"},
            },
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[edge] ✓ Exported ({size_mb:.1f} MB)")
        return str(output_path)

    # ----------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------

    def preprocess(self, image) -> np.ndarray:
        """
        Preprocess image for ONNX inference.

        Steps: load → resize → normalise [0,1] → NCHW → float32.

        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image (path or BGR array).

        Returns
        -------
        np.ndarray, shape (1, 3, H, W)
            Preprocessed image tensor.
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img = image.copy()

        # Store original size for box rescaling
        self._orig_h, self._orig_w = img.shape[:2]

        # Resize
        h, w = self.input_size
        img_resized = cv2.resize(img, (w, h))

        # BGR → RGB, HWC → CHW, normalise to [0, 1]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(img_chw, axis=0)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def _load_session(self) -> None:
        """Load ONNX Runtime inference session."""
        try:
            import onnxruntime as ort
        except ImportError:
            print("[edge] ⚠ onnxruntime not installed. Install with: pip install onnxruntime")
            raise

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        providers = ["CPUExecutionProvider"]
        try:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        print(f"[edge] ✓ Loaded ONNX session: {self.model_path.name}")

    def run_inference(
        self, image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run edge inference: preprocess → ONNX → Soft-NMS.

        Parameters
        ----------
        image : str, Path, or np.ndarray

        Returns
        -------
        boxes : np.ndarray, shape (K, 4) — [x1, y1, x2, y2]
        scores : np.ndarray, shape (K,)
        """
        if self.session is None:
            self._load_session()

        # Preprocess
        input_tensor = self.preprocess(image)

        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        # Parse outputs (torchvision detection model format)
        if len(outputs) >= 3:
            raw_boxes = outputs[0]   # (N, 4)
            raw_labels = outputs[1]  # (N,)
            raw_scores = outputs[2]  # (N,)
        elif len(outputs) == 2:
            raw_boxes = outputs[0]
            raw_scores = outputs[1]
        else:
            raw_boxes = outputs[0]
            raw_scores = np.ones(len(raw_boxes))

        if len(raw_boxes) == 0:
            return np.empty((0, 4)), np.empty(0)

        # Rescale boxes to original image size
        scale_x = self._orig_w / self.input_size[1]
        scale_y = self._orig_h / self.input_size[0]
        raw_boxes[:, [0, 2]] *= scale_x
        raw_boxes[:, [1, 3]] *= scale_y

        # Apply Soft-NMS
        from src.models.soft_nms import soft_nms_np
        method = self._method_map.get(self.nms_method, "gaussian")
        keep, new_scores = soft_nms_np(
            raw_boxes, raw_scores,
            sigma=self.sigma,
            score_threshold=self.score_threshold,
            method=method,
        )

        if len(keep) == 0:
            return np.empty((0, 4)), np.empty(0)

        return raw_boxes[keep], new_scores

    # ----------------------------------------------------------------
    # Result Compression
    # ----------------------------------------------------------------

    @staticmethod
    def compress_results(
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Create a compact JSON payload for server transmission.

        Parameters
        ----------
        boxes : np.ndarray (K, 4)
        scores : np.ndarray (K,)
        threshold : float
            Only include detections above this score.
        top_k : int
            Maximum number of ROIs to send.

        Returns
        -------
        dict
            Compact payload with object count, top-K ROI coordinates,
            and confidence summary statistics.
        """
        # Filter by threshold
        mask = scores >= threshold
        filt_boxes = boxes[mask]
        filt_scores = scores[mask]

        # Sort by score descending, take top-K
        order = np.argsort(-filt_scores)[:top_k]
        top_boxes = filt_boxes[order]
        top_scores = filt_scores[order]

        return {
            "object_count": int(len(top_boxes)),
            "rois": top_boxes.round(1).tolist(),
            "scores": top_scores.round(4).tolist(),
            "confidence_summary": {
                "min": float(top_scores.min()) if len(top_scores) else 0.0,
                "max": float(top_scores.max()) if len(top_scores) else 0.0,
                "mean": float(top_scores.mean()) if len(top_scores) else 0.0,
            },
            "payload_size_bytes": len(json.dumps({
                "r": top_boxes.round(1).tolist(),
                "s": top_scores.round(4).tolist(),
            })),
        }

    # ----------------------------------------------------------------
    # Benchmarking
    # ----------------------------------------------------------------

    def benchmark(
        self, image, n_runs: int = 50,
    ) -> Dict[str, Any]:
        """
        Benchmark edge inference latency.

        Returns
        -------
        dict with: avg_latency_ms, fps, peak_memory_mb, n_runs
        """
        import resource

        if self.session is None:
            try:
                self._load_session()
            except (FileNotFoundError, ImportError):
                return self._benchmark_fallback(image, n_runs)

        # Warmup
        for _ in range(3):
            try:
                self.run_inference(image)
            except Exception:
                return self._benchmark_fallback(image, n_runs)

        # Timed runs
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run_inference(image)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        usage = resource.getrusage(resource.RUSAGE_SELF)
        mem_mb = usage.ru_maxrss / (1024 * 1024)  # macOS reports bytes

        return {
            "avg_latency_ms": round(avg_ms, 2),
            "fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
            "peak_memory_mb": round(mem_mb, 1),
            "n_runs": n_runs,
            "model": str(self.model_path.name),
            "input_size": list(self.input_size),
            "nms_method": self.nms_method,
        }

    def _benchmark_fallback(
        self, image, n_runs: int
    ) -> Dict[str, Any]:
        """Fallback benchmark using preprocess-only (no ONNX model)."""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.preprocess(image) if not isinstance(image, np.ndarray) else image
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        return {
            "avg_latency_ms": round(avg_ms, 2),
            "fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
            "peak_memory_mb": 0.0,
            "n_runs": n_runs,
            "model": "preprocess_only (no ONNX)",
            "input_size": list(self.input_size),
            "nms_method": self.nms_method,
            "note": "ONNX model not available; benchmarked preprocessing only",
        }

    def __repr__(self) -> str:
        return (
            f"EdgeInferencePipeline(model={self.model_path.name}, "
            f"nms={self.nms_method}, σ={self.sigma}, "
            f"input={self.input_size})"
        )


# ====================================================================
# ServerRefinementPipeline
# ====================================================================

class ServerRefinementPipeline:
    """
    Server-side refinement using the full detection model.

    Receives compressed ROIs from edge devices, runs higher-resolution
    inference on cropped regions, and returns refined detections.

    Parameters
    ----------
    model_backbone : str
        'resnet50' or 'mobilenet_v3' for the full server model.
    nms_method : str
        NMS method for server-side post-processing.
    sigma : float
        Soft-NMS sigma for server.
    score_threshold : float
        Score threshold for final detections.
    """

    def __init__(
        self,
        model_backbone: str = "resnet50",
        nms_method: str = "soft_gaussian",
        sigma: float = 0.5,
        score_threshold: float = 0.3,
    ):
        self.model_backbone = model_backbone
        self.nms_method = nms_method
        self.sigma = sigma
        self.score_threshold = score_threshold
        self._detector = None

    def _load_detector(self):
        """Lazy-load the full server detector."""
        from src.models.detector import DenseObjectDetector
        self._detector = DenseObjectDetector(
            backbone=self.model_backbone,
            nms_method=self.nms_method,
            sigma=self.sigma,
            score_thresh=self.score_threshold,
        )
        self._detector.load_model()

    def receive_rois(
        self,
        compressed_data: Dict[str, Any],
        full_image: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Receive compressed ROIs from edge and refine with full model.

        Parameters
        ----------
        compressed_data : dict
            Payload from EdgeInferencePipeline.compress_results().
        full_image : optional
            Full-resolution image for re-detection on ROI crops.

        Returns
        -------
        dict
            Refined results with updated boxes, scores, and counts.
        """
        rois = compressed_data.get("rois", [])
        edge_scores = compressed_data.get("scores", [])
        edge_count = compressed_data.get("object_count", 0)

        if full_image is not None and self._detector is not None:
            # Re-run detection on full image with full model
            boxes, scores, labels = self._detector.detect(full_image)
            return {
                "refined_count": int(len(boxes)),
                "edge_count": edge_count,
                "count_delta": int(len(boxes)) - edge_count,
                "refined_boxes": boxes.tolist() if hasattr(boxes, 'tolist') else list(boxes),
                "refined_scores": scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                "method": f"server_{self.model_backbone}",
            }

        # Placeholder: accept edge results with confidence boost
        return {
            "refined_count": edge_count,
            "edge_count": edge_count,
            "count_delta": 0,
            "refined_boxes": rois,
            "refined_scores": edge_scores,
            "method": "passthrough (no full image provided)",
        }

    def __repr__(self) -> str:
        return (
            f"ServerRefinementPipeline(backbone='{self.model_backbone}', "
            f"nms='{self.nms_method}')"
        )


# ====================================================================
# CLI / Demo
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Edge inference pipeline demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--export-only", action="store_true",
                        help="Only export ONNX model")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")
    parser.add_argument("--image", type=str, default=None,
                        help="Test image path")
    parser.add_argument("--input-size", type=int, nargs=2, default=[320, 320],
                        help="Model input size (H W)")
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    input_size = tuple(args.input_size)

    # ---- Find a test image ----
    if args.image:
        test_image = args.image
    else:
        img_dir = SYNTHETIC_DIR / "images"
        if img_dir.exists():
            imgs = list(img_dir.glob("*.png"))
            test_image = str(imgs[0]) if imgs else None
        else:
            test_image = None

    # ---- Export ----
    onnx_path = MODELS_DIR / "mobilenet_v3_detector.onnx"

    if args.export_only or not onnx_path.exists():
        print("\n=== ONNX Export ===")
        try:
            import torch
            import torchvision
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            )
            model.eval()
            EdgeInferencePipeline.export_to_onnx(model, str(onnx_path), input_size)
        except Exception as e:
            print(f"[edge] ⚠ Export failed: {e}")
            print("       (This is expected if torch/torchvision is not installed)")
            traceback.print_exc()

    if args.export_only:
        return

    # ---- Edge Pipeline Demo ----
    print("\n=== Edge Inference Pipeline ===")
    edge = EdgeInferencePipeline(
        model_path=str(onnx_path),
        nms_method="soft_gaussian",
        sigma=0.5,
        input_size=input_size,
    )
    print(edge)

    if test_image:
        print(f"\nTest image: {test_image}")

        try:
            boxes, scores = edge.run_inference(test_image)
            print(f"Detected: {len(boxes)} objects")

            # Compress
            payload = edge.compress_results(boxes, scores)
            print(f"\nCompressed payload:")
            print(f"  Object count: {payload['object_count']}")
            print(f"  Payload size: {payload['payload_size_bytes']} bytes")
            print(f"  Confidence: {payload['confidence_summary']}")
        except Exception as e:
            print(f"[edge] ⚠ Inference failed: {e}")
            print("       (ONNX model may not be available)")

        # ---- Benchmark ----
        if args.benchmark:
            print(f"\n=== Benchmark ({args.n_runs} runs) ===")
            bench = edge.benchmark(test_image, n_runs=args.n_runs)
            for k, v in bench.items():
                print(f"  {k}: {v}")

    # ---- Server Pipeline ----
    print("\n=== Server Refinement Pipeline ===")
    server = ServerRefinementPipeline(model_backbone="resnet50")
    print(server)
    print("(Server refinement is a placeholder — requires full model loading)")
    print()


if __name__ == "__main__":
    main()
