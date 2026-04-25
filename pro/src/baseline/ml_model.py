"""
ml_model.py — Classical ML (Random Forest) Classifier
======================================================
Trains a non-DL classifier on hand-crafted features extracted from
classical computer vision bounding box proposals.

This approach demonstrates a hybrid classical pipeline:
  1. Proposal Generation (e.g. Watershed / GraphSegmenter)
  2. Feature Extraction (from features.py)
  3. Binary Classification (Random Forest) -> Keep or Discard
"""

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Dict, Optional, Any

from src.evaluation.metrics import compute_iou_matrix

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ClassicalMLDetector:
    """
    A machine learning classifier that acts as a secondary filter
    on top of classical computer vision proposals.
    """

    def __init__(self, model_path: Optional[str] = None, n_estimators: int = 100, \
                 max_depth: Optional[int] = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

        if model_path is None:
            self.model_path = MODELS_DIR / "rf_baseline.joblib"
        else:
            self.model_path = Path(model_path)

        if self.model_path.exists():
            self.load_model()
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced"
            )

    def load_model(self) -> None:
        """Load pretrained Random Forest model."""
        print(f"[ClassicalML] Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

    def save_model(self) -> None:
        """Save the Random Forest model."""
        joblib.dump(self.model, self.model_path)
        print(f"[ClassicalML] Model saved to {self.model_path}")

    def prepare_training_data(self, 
                              proposals_list: List[np.ndarray], 
                              features_list: List[np.ndarray], 
                              gt_boxes_list: List[List[Tuple[int, int, int, int]]],
                              iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match proposals against ground truth to generate positive/negative labels.

        Parameters
        ----------
        proposals_list : list of np.ndarray
            List where each element is an array of shape (N, 4) containing proposed boxes.
        features_list : list of np.ndarray
            List where each element is an array of shape (N, 5) containing extracted features.
        gt_boxes_list : list of lists of tuples
            Ground truth boxes per image.
        iou_threshold : float
            Threshold above which a proposal is considered a True Positive.

        Returns
        -------
        X : np.ndarray
            Feature matrix for all images combined.
        y : np.ndarray
            Label matrix (1 for object, 0 for background).
        """
        X_all = []
        y_all = []

        for preds, feats, gts in zip(proposals_list, features_list, gt_boxes_list):
            if len(preds) == 0:
                continue
            
            p_boxes = np.array(preds).reshape(-1, 4)
            g_boxes = np.array(gts).reshape(-1, 4)

            labels = np.zeros(len(p_boxes), dtype=int)
            
            if len(g_boxes) > 0:
                iou_mat = compute_iou_matrix(p_boxes, g_boxes)
                max_ious = np.max(iou_mat, axis=1)
                labels[max_ious >= iou_threshold] = 1

            X_all.append(feats)
            y_all.append(labels)

        if len(X_all) == 0:
            return np.empty((0, 5)), np.empty((0,))

        return np.vstack(X_all), np.concatenate(y_all)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier."""
        print(f"[ClassicalML] Training Random Forest on {len(X)} samples "
              f"({np.sum(y)} positive, {len(y) - np.sum(y)} negative)...")
        self.model.fit(X, y)
        print(f"[ClassicalML] ✓ Training complete. Train accuracy: {self.model.score(X, y):.3f}")
        self.save_model()

    def predict(self, features: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for features, keeping only those above confidence_threshold.

        Returns
        -------
        keep_indices : np.ndarray
            Boolean array or indices of proposals to keep.
        """
        if len(features) == 0:
            return np.array([], dtype=bool)

        # Get probabilities for the positive class
        probs = self.model.predict_proba(features)[:, 1]
        keep = probs >= confidence_threshold
        return keep, probs
