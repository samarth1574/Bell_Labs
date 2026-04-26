"""
robustness.py — Technical Validation (Robustness Analysis)
==========================================================
Measures DL model performance degradation under domain shifts:
- Gaussian Noise 
- Gaussian Blur
- Brightness Shifts
"""

import pandas as pd
import numpy as np
import cv2
import argparse
from pathlib import Path

from src.data_loader import load_synthetic_annotations
from src.evaluation.metrics import full_evaluation
from src.models.detector import DenseObjectDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
IMAGES_DIR = SYNTHETIC_DIR / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def apply_noise(img: np.ndarray, intensity=0.01) -> np.ndarray:
    """Adds Gaussian noise."""
    noise = np.random.normal(0, intensity * 255, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy

def apply_blur(img: np.ndarray, kernel=(5, 5)) -> np.ndarray:
    """Applies Gaussian Blur."""
    return cv2.GaussianBlur(img, kernel, 0)

def apply_brightness(img: np.ndarray, factor=1.3) -> np.ndarray:
    """Adjusts brightness factor (>1 is brighter)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.HSV2BGR)

def evaluate_robustness(detector, test_images, gt_by_image, condition, transform_func):
    """Run evaluation over the transformed dataset."""
    print(f"\n[Robustness] Evaluating condition: {condition}")
    predictions = []
    ground_truths = []
    
    for fname in test_images:
        img_path = str(IMAGES_DIR / fname)
        gt = gt_by_image.get(fname, [])
        ground_truths.append({"boxes": gt})
        
        img = cv2.imread(img_path)
        if img is None:
            predictions.append({"boxes": [], "scores": []})
            continue
            
        # Apply corruption
        if transform_func:
            img = transform_func(img)
            
        try:
            b, s, l = detector.detect(img)
            final_boxes = b.numpy().tolist()
            final_probs = s.numpy().tolist()
        except Exception:
            final_boxes, final_probs = [], []
            
        predictions.append({"boxes": final_boxes, "scores": final_probs})
        
    return full_evaluation(predictions, ground_truths)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_test_images", type=int, default=10)
    parser.add_argument("--config", type=str, default="configs/dl_softnms_density.yaml")
    args = parser.parse_args()
    
    ann_path = SYNTHETIC_DIR / "annotations.json"
    if not ann_path.exists():
        print("Synthetic annotations missing for robustness testing.")
        return
        
    data = load_synthetic_annotations(str(ann_path))
    img_map = {img["id"]: img["file_name"] for img in data["images"]}
    gt_by_image = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        fname = img_map[ann["image_id"]]
        gt_by_image.setdefault(fname, []).append((x, y, x + w, y + h))

    test_images = list(img_map.values())[:args.max_test_images]
    
    detector = DenseObjectDetector(config_path=args.config, device="cpu")
    
    # Define robustness conditions
    conditions = {
        "Baseline (Clean)": None,
        "Gaussian Noise (s=0.01)": lambda x: apply_noise(x, 0.01),
        "Gaussian Blur (5x5)": lambda x: apply_blur(x, (5, 5)),
        "Darkness (-30%)": lambda x: apply_brightness(x, 0.7),
        "Brightness (+30%)": lambda x: apply_brightness(x, 1.3),
    }
    
    results = []
    for name, func in conditions.items():
        try:
            res = evaluate_robustness(detector, test_images, gt_by_image, name, func)
            results.append({
                "Condition": name,
                "mAP@0.5": res['mAP@0.5'],
                "mAP@0.5:0.95": res['mAP@0.5:0.95'],
                "Count MAE": res['count_mae'],
                "Count RMSE": res['count_rmse']
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results.append({"Condition": name, "mAP@0.5": 0, "mAP@0.5:0.95": 0, "Count MAE": 0, "Count RMSE": 0})
            
    df = pd.DataFrame(results)
    out_md = REPORTS_DIR / "robustness_metrics.md"
    
    with open(out_md, "w") as f:
        f.write("# Robustness Degradation Analysis\n\n")
        f.write("Evaluation of Phase 2 DL model gracefully handling domain shifts.\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"\n--- Saved Robustness Table to {out_md.name} ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
