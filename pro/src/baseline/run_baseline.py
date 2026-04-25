"""
run_baseline.py — ML Baseline Entry Point
==========================================
Coordinates classical CV region proposals with the trained
Random Forest model for feature-based classification.

Usage:
    python -m src.baseline.run_baseline --train --max_train_images 100
    python -m src.baseline.run_baseline --evaluate --max_eval_images 50
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

from src.baseline.classical_cv import WatershedSegmenter, compute_iou
from src.baseline.features import extract_features, get_feature_names
from src.baseline.ml_model import ClassicalMLDetector
from src.baseline.plots import plot_feature_importances, plot_predicted_vs_actual_counts
from src.evaluation.metrics import full_evaluation
from src.data_loader import load_synthetic_annotations

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
IMAGES_DIR = SYNTHETIC_DIR / "images"

def _get_dataset(max_images: int = 50, split: str = "train"):
    """
    Fetch image paths and GT boxes from the synthetic dataset.
    Very basic random split by hash.
    """
    ann_path = SYNTHETIC_DIR / "annotations.json"
    if not ann_path.exists():
        print("⚠ Synthetic dataset not found. Run `python -m src.synthetic_generator`.")
        sys.exit(1)
        
    try:
        data = load_synthetic_annotations(str(ann_path))
    except Exception as e:
        print(f"Error loading annotations: {e}")
        sys.exit(1)

    # Build GT map
    img_map = {img["id"]: img["file_name"] for img in data["images"]}
    gt_by_image = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        fname = img_map[ann["image_id"]]
        gt_by_image.setdefault(fname, []).append((x, y, x + w, y + h))

    images = sorted(list(img_map.values()))
    
    # Split: first 70% train, last 30% test
    split_idx = int(len(images) * 0.7)
    if split == "train":
        images = images[:split_idx]
    else:
        images = images[split_idx:]
    
    # Subsample if requested
    rng = np.random.RandomState(42)
    if max_images and len(images) > max_images:
        images = rng.choice(images, max_images, replace=False).tolist()

    return images, gt_by_image

def train_baseline(max_images: int = 100):
    images, gt_by_image = _get_dataset(max_images=max_images, split="train")
    print(f"--- Training ML Baseline on {len(images)} images ---")
    
    # 1. Classical CV Proposer
    segmenter = WatershedSegmenter()
    
    # 2. Extract Features
    proposals_list = []
    features_list = []
    gt_list = []
    
    for fname in images:
        img_path = str(IMAGES_DIR / fname)
        gt = gt_by_image.get(fname, [])
        gt_list.append(gt)
        
        # Region proposals
        try:
            boxes = segmenter.detect(img_path)
        except Exception as e:
            print(f"Error segmenting {fname}: {e}")
            boxes = []
        
        if not boxes:
            proposals_list.append(np.empty((0, 4)))
            features_list.append(np.empty((0, 5)))
            continue
            
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            proposals_list.append(np.empty((0, 4)))
            features_list.append(np.empty((0, 5)))
            continue

        feats = extract_features(img, boxes)
        proposals_list.append(np.array(boxes))
        features_list.append(feats)

    # 3. Prepare data and Train Model
    ml_detector = ClassicalMLDetector()
    X, y = ml_detector.prepare_training_data(
        proposals_list, features_list, gt_list, iou_threshold=0.4
    )
    
    if len(X) == 0:
        print("No valid proposals found to train on.")
        return
        
    ml_detector.train(X, y)
    
    # 4. Plot Feature Importances
    plot_feature_importances(ml_detector.model.feature_importances_, get_feature_names())

def evaluate_baseline(max_images: int = 50):
    images, gt_by_image = _get_dataset(max_images=max_images, split="test")
    print(f"--- Evaluating ML Baseline on {len(images)} images ---")
    
    # Load Models
    segmenter = WatershedSegmenter()
    ml_detector = ClassicalMLDetector()
    
    predictions = []
    ground_truths = []
    
    pred_counts = []
    gt_counts = []

    for fname in images:
        img_path = str(IMAGES_DIR / fname)
        gt = gt_by_image.get(fname, [])
        ground_truths.append({"boxes": gt})
        gt_counts.append(len(gt))
        
        # Region proposals
        try:
            boxes = segmenter.detect(img_path)
        except Exception as e:
            boxes = []
            
        if not boxes:
            predictions.append({"boxes": [], "scores": []})
            pred_counts.append(0)
            continue
            
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            predictions.append({"boxes": [], "scores": []})
            pred_counts.append(0)
            continue
            
        # Feature Extraction
        feats = extract_features(img, boxes)
        
        # ML Prediction (Secondary Filter)
        keep_mask, probs = ml_detector.predict(feats, confidence_threshold=0.5)
        
        # Filter boxes
        final_boxes = np.array(boxes)[keep_mask].tolist()
        final_probs = probs[keep_mask].tolist()
        
        predictions.append({"boxes": final_boxes, "scores": final_probs})
        pred_counts.append(len(final_boxes))

    # Evaluate Metrics
    results = full_evaluation(predictions, ground_truths)
    
    print("\n--- ML Baseline Test Results ---")
    print(f"mAP@0.5:       {results['mAP@0.5']:.3f}")
    print(f"mAP@0.5:0.95:  {results['mAP@0.5:0.95']:.3f}")
    print(f"Count MAE:     {results['count_mae']:.3f}")
    print(f"Count RMSE:    {results['count_rmse']:.3f}")
    print(f"IoU Coverage:  {results['iou_coverage']:.3f}")
    
    plot_predicted_vs_actual_counts(pred_counts, gt_counts, "ml_baseline_counts.png")


def main():
    parser = argparse.ArgumentParser(description="ML Baseline (CV + Random Forest)")
    parser.add_argument("--train", action="store_true", help="Train the ML model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the ML model on test set")
    parser.add_argument("--max_train_images", type=int, default=100)
    parser.add_argument("--max_eval_images", type=int, default=50)
    args = parser.parse_args()
    
    if args.train:
        train_baseline(args.max_train_images)
    if args.evaluate:
        evaluate_baseline(args.max_eval_images)
        
    if not args.train and not args.evaluate:
        print("Please specify --train or --evaluate.")

if __name__ == "__main__":
    main()
