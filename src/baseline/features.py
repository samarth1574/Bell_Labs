"""
features.py — Feature Extraction for Classical CV Region Proposals
==================================================================
Extracts hand-crafted features from proposed bounding boxes.
Features include:
  - Box dimensions (area, aspect_ratio)
  - Edge density (via Canny)
  - Color statistics (mean_intensity, color_variance)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

def extract_features(image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Extract features for a list of bounding box proposals.

    Parameters
    ----------
    image : np.ndarray
        The original BGR image.
    boxes : List[Tuple[int, int, int, int]]
        List of proposed bounding boxes (x1, y1, x2, y2).

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 5) where N is the number of boxes.
        Columns: [area, aspect_ratio, edge_density, mean_intensity, color_variance]
    """
    if len(boxes) == 0:
        return np.empty((0, 5))

    # Convert image heavily used formats once
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    edges = cv2.Canny(gray, 50, 150)

    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    features_list = []

    for x1, y1, x2, y2 in boxes:
        # Constrain coordinates to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # 1. Geometry
        area = (w * h) / img_area  # Normalized area
        aspect_ratio = w / h

        # Crop regions
        edge_crop = edges[y1:y2, x1:x2]
        color_crop = image[y1:y2, x1:x2]

        # 2. Edge Density
        if edge_crop.size > 0:
            edge_density = np.sum(edge_crop > 0) / (w * h)
        else:
            edge_density = 0.0

        # 3. Intensity Statistics
        if color_crop.size > 0:
            mean_intensity = np.mean(color_crop) / 255.0  # Normalize to [0,1]
            color_variance = np.var(color_crop) / (255.0 ** 2)
        else:
            mean_intensity = 0.0
            color_variance = 0.0

        features_list.append([
            float(area),
            float(aspect_ratio),
            float(edge_density),
            float(mean_intensity),
            float(color_variance)
        ])

    return np.array(features_list, dtype=np.float32)

def get_feature_names() -> List[str]:
    """Return the names of the extracted features."""
    return ["area", "aspect_ratio", "edge_density", "mean_intensity", "color_variance"]
