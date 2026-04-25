"""
density_head.py — Coarse Density Estimation Head
=================================================
A lightweight convolutional head attached to the backbone
to predict coarse object counts (or density maps).
This acts as a regularizer during training and can provide
a fast count proxy during inference.
"""

import torch
import torch.nn as nn
from typing import Dict, OrderedDict

class DensityHead(nn.Module):
    """
    Consumes FPN feature maps (typically '0', '1', '2', '3')
    and regress a global object count.
    We use the highest resolution feature map ('0') by default.
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # A few downsampling conv layers
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Final aggregation
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features: OrderedDict) -> torch.Tensor:
        """
        features : dict mapping FPN levels (e.g. '0', '1', '2') to tensors.
        We extract level '0' which has the highest spatial resolution.
        """
        # Usually '0' contains the P2 features (stride 4 or 8)
        if '0' in features:
            x = features['0']
        else:
            # Fallback for non-FPN dicts
            x = list(features.values())[0]

        # Produce a coarse density map (B, 1, H/4, W/4)
        density_map = self.convs(x)
        
        # Global aggregation to a single count scalar (B, 1)
        # We enforce >0 count with softplus or relu. Relu chosen for simplicity.
        count = torch.relu(self.gap(density_map))
        return count.view(x.shape[0], -1)

def add_density_head_to_backbone(backbone: nn.Module) -> nn.Module:
    """
    Wraps the torchvision backbone to intercept its output
    and run it through the DensityHead simultaneously.
    """
    class BackboneWithDensity(nn.Module):
        def __init__(self, base_backbone):
            super().__init__()
            self.base_backbone = base_backbone
            self.out_channels = base_backbone.out_channels
            # Typically out_channels is 256 for FasterRCNN's FPN
            self.density_head = DensityHead(in_channels=self.out_channels)
            
        def forward(self, x):
            features = self.base_backbone(x)
            density_count = self.density_head(features)
            
            # Monkey-patch the return dict or just store it.
            # Faster R-CNN RPN only uses OrderedDict. We can't trivially 
            # push density forward into the standard loss unless we modify 
            # the training loop. We will attach it as an attribute 
            # so the training loop can easily retrieve it: `model.backbone_density_count`
            self.current_density = density_count
            return features

    return BackboneWithDensity(backbone)
