"""
config.py — Model Configuration Loader
=======================================
Parses YAML files for detector configuration.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v3"
    use_density_head: bool = False

@dataclass
class NMSConfig:
    method: str = "soft_gaussian"
    iou_threshold: float = 0.5
    score_thresh: float = 0.1
    sigma: float = 0.5

@dataclass
class AnchorConfig:
    use_custom_dense_anchors: bool = False

@dataclass
class DatasetConfig:
    batch_size: int = 4
    num_workers: int = 2
    prefetch_factor: int = 2
    use_flips: bool = False
    use_jitter: bool = False
    advanced_aug: str = "none"
    oversample_dense: bool = False

@dataclass
class TrainingConfig:
    weight_decay: float = 0.0001
    dropout: float = 0.0
    patience: int = 10
    focal_loss: bool = False

@dataclass
class DetectorConfig:
    model: ModelConfig
    nms: NMSConfig
    anchors: AnchorConfig
    dataset: DatasetConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "DetectorConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        m_data = data.get('model', {})
        n_data = data.get('nms', {})
        a_data = data.get('anchors', {})
        d_data = data.get('dataset', {})
        t_data = data.get('training', {})
        
        return cls(
            model=ModelConfig(**m_data),
            nms=NMSConfig(**n_data),
            anchors=AnchorConfig(**a_data),
            dataset=DatasetConfig(**d_data),
            training=TrainingConfig(**t_data)
        )
