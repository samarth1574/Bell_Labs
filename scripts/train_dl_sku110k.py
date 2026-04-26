#!/usr/bin/env python3
"""Train, validate, predict, or export a YOLO11 detector on SKU-110K."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 YOLO11 SKU-110K training entry point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_hybrid_yolo11.yaml"))
    parser.add_argument("--data", type=Path, default=None, help="Override dataset YAML.")
    parser.add_argument("--model", type=str, default=None, help="Override YOLO checkpoint/model.")
    parser.add_argument("--mode", choices=["train", "val", "predict", "export"], default="train")
    parser.add_argument("--weights", type=Path, default=None, help="Trained weights for val/predict/export.")
    parser.add_argument("--source", type=Path, default=None, help="Image/video/folder for predict mode.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dl_cfg = cfg.get("dl", {})
    data_yaml = args.data or Path(cfg.get("dataset", {}).get("data_yaml", "data/sku110k/sku110k.yaml"))
    model_name = args.model or str(args.weights or dl_cfg.get("model", "yolo11l.pt"))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required for YOLO11 training. Install with: pip install -r requirements.txt"
        ) from exc

    model = YOLO(model_name)
    device = args.device or dl_cfg.get("device", None)
    if device == "auto":
        device = None

    if args.mode == "train":
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs or int(dl_cfg.get("epochs", 100)),
            imgsz=args.imgsz or int(dl_cfg.get("imgsz", 960)),
            batch=args.batch or int(dl_cfg.get("batch", 8)),
            workers=int(dl_cfg.get("workers", 4)),
            patience=int(dl_cfg.get("patience", 20)),
            optimizer=str(dl_cfg.get("optimizer", "AdamW")),
            lr0=float(dl_cfg.get("lr0", 0.001)),
            weight_decay=float(dl_cfg.get("weight_decay", 0.0005)),
            device=device,
            project="runs/phase3",
            name="yolo11_sku110k",
        )
    elif args.mode == "val":
        results = model.val(data=str(data_yaml), imgsz=args.imgsz or int(dl_cfg.get("imgsz", 960)))
    elif args.mode == "predict":
        if args.source is None:
            raise SystemExit("--source is required for predict mode")
        results = model.predict(
            source=str(args.source),
            imgsz=args.imgsz or int(dl_cfg.get("imgsz", 960)),
            conf=float(cfg.get("hybrid", {}).get("score_threshold", 0.08)),
            save=True,
            project="runs/phase3",
            name="predict",
        )
    else:
        results = model.export(format="onnx", imgsz=args.imgsz or int(dl_cfg.get("imgsz", 960)), dynamic=True)

    print(results)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
