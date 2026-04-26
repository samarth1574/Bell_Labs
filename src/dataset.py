"""
dataset.py — PyTorch Dataset Wrappers & Augmentations
=====================================================
Wraps Phase 1 `data_loader.py` logic into formal PyTorch datasets.
Implements:
- Normalisation
- Flips, Scale/Color Jitter
- MixUp/CutMix/Mosaic modules (conditional)
- Dataloader wrappers (num_workers, batch_size, weighted sampling)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import random
from pathlib import Path

from src.data_loader import load_image, RAW_DIR
from src.models.config import DatasetConfig

class DenseObjectDataset(Dataset):
    """
    A unified PyTorch Dataset for dense object detection.
    """
    def __init__(self, df: pd.DataFrame, image_dir: str = str(RAW_DIR), 
                 config: DatasetConfig = None, is_train: bool = False):
        self.image_dir = Path(image_dir)
        self.config = config or DatasetConfig()
        self.is_train = is_train
        
        # Group annotations by image
        self.images = list(df["image_name"].unique())
        self.df_grouped = df.groupby("image_name")
        
        # Standard ImageNet Norm
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = str(self.image_dir / img_name)
        
        # Default empty return on missing file
        image_np = load_image(img_path)
        if image_np is None:
            # Return a dummy if file doesn't exist to gracefully handle missing local data
            image_np = np.zeros((300, 300, 3), dtype=np.uint8)
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        else:
            group = self.df_grouped.get_group(img_name)
            boxes = group[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
            # Add basic bounds clamping
            h, w = image_np.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            labels = np.ones((len(boxes),), dtype=np.int64) # Class 1 for all objects
            
        # Convert to tensor formats
        image_t = TF.to_tensor(image_np) # Automatically converts [0, 255] -> [0, 1]
        
        # Apply standard bounding-box-aware augmentations (Basic Phase 2 requirements)
        if self.is_train:
            image_t, boxes = self._apply_augmentations(image_t, boxes)

        # Normalise
        image_t = TF.normalize(image_t, mean=self.mean, std=self.std)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return image_t, target

    def _apply_augmentations(self, img: torch.Tensor, boxes: np.ndarray):
        """
        Applies bounding-box safe horizontal/vertical flips and color jitter
        based on configuration toggles.
        """
        _, h, w = img.shape
        
        # Horizontal Flip
        if self.config.use_flips and random.random() > 0.5:
            img = TF.hflip(img)
            if len(boxes) > 0:
                # x1, x2 = w - x2, w - x1
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2

        # Vertical Flip
        if self.config.use_flips and random.random() > 0.5:
            img = TF.vflip(img)
            if len(boxes) > 0:
                y1 = h - boxes[:, 3]
                y2 = h - boxes[:, 1]
                boxes[:, 1] = y1
                boxes[:, 3] = y2

        # Color Jitter
        if self.config.use_jitter and random.random() > 0.5:
            brightness = 0.2
            contrast = 0.2
            saturation = 0.2
            hue = 0.1
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-brightness, brightness))
            img = TF.adjust_contrast(img, 1.0 + random.uniform(-contrast, contrast))
            img = TF.adjust_saturation(img, 1.0 + random.uniform(-saturation, saturation))
            img = TF.adjust_hue(img, random.uniform(-hue, hue))

        # Note: Advanced MixUp/CutMix is typically applied at the batch level during the training loop.
        # Handled in trainer_utils.py.
        return img, boxes

def collate_fn(batch):
    """Custom collate function for object detection batching (returns tuples of tensors/dicts)"""
    return tuple(zip(*batch))

def build_dataloader(df: pd.DataFrame, config: DatasetConfig, is_train: bool = True) -> DataLoader:
    """
    Constructs a PyTorch DataLoader utilizing the configs for num_workers, prefetch, and sampling.
    """
    dataset = DenseObjectDataset(df, config=config, is_train=is_train)
    sampler = None
    
    if is_train and config.oversample_dense:
        print("[dataset] Applying WeightedRandomSampler for dense image regularisation...")
        # Weight by object count (more objects -> higher probability of sampling)
        counts = df.groupby("image_name").size()
        # Create a dict mapping image name to count
        count_map = counts.to_dict()
        weights = [count_map.get(img, 1) for img in dataset.images]
        # Soften weights to prevent extreme oversampling
        weights = np.sqrt(weights)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return loader
