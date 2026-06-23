"""Reduced VGG16 baseline for comparison with MSCA-VGG16.

Shares the same VGG16-BN backbone and FC projection (512→1024) as
MSCA-VGG16, but strips the multi-scale convolutions, channel attention,
and compact embedding head.  This provides a fair, lightweight baseline
for ablation studies (configuration B0 in the paper).
"""

import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class VGG16(nn.Module):
    """VGG16-BN backbone + compact FC head (no MSCA enhancements).

    Architecture:
        VGG16-BN features  →  512 × 7 × 7
        AdaptiveAvgPool2d  →  512
        FC1                →  1024, ReLU, Dropout(0.5)
        Classifier         →  num_classes

    This is equivalent to MSCA-VGG16 with
    ``use_ms=False, use_ca=False, use_eh=False``.

    Args:
        num_classes: number of output classes.
        pretrained:  load ImageNet pre-trained weights.
        dropout:     dropout rate in the FC projection layer (default 0.5).
    """

    def __init__(self, num_classes=10, pretrained=True, dropout=0.5):
        super().__init__()

        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16_bn(weights=weights)

        # Backbone only — discard the original heavy classifier
        self.features = base.features              # → 512 × 7 × 7
        self.avgpool = nn.AdaptiveAvgPool2d(1)     # → 512 × 1 × 1

        # Compact FC projection (same as MSCA-VGG16 fc1)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Direct classifier — no embedding head
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)           # → 512 × 7 × 7
        x = self.avgpool(x)            # → 512 × 1 × 1
        x = torch.flatten(x, 1)        # → 512

        x = self.fc1(x)                # → 1024
        feat = x                       # t-SNE feature (before classifier)

        out = self.classifier(feat)    # → num_classes

        if return_feat:
            return feat, out
        return out


def build_vgg16(num_classes=10, pretrained=True):
    return VGG16(num_classes, pretrained)

