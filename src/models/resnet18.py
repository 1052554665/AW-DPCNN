"""ResNet18 baseline — ubiquitous in fault diagnosis literature.

A standard ResNet18 backbone with a compact FC head matching the VGG16
baseline design (512→1024→num_classes).  This provides a widely-recognized
residual-network comparison point for ablation and backbone studies.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    """ResNet18 backbone + compact FC head.

    Architecture:
        ResNet18 features  →  512 × 7 × 7
        AdaptiveAvgPool2d  →  512
        FC1                →  1024, ReLU, Dropout(0.5)
        Classifier         →  num_classes

    Args:
        num_classes: number of output classes.
        pretrained:  load ImageNet pre-trained weights.
        dropout:     dropout rate in the FC projection layer (default 0.5).
    """

    def __init__(self, num_classes=10, pretrained=True, dropout=0.5):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Strip the original FC layer; keep everything up to avgpool
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # → 512 × 1 × 1

        # Compact FC projection (same design as VGG16 baseline)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)       # → (B, 512)

        x = self.fc1(x)                  # → (B, 1024)
        feat = x                          # t-SNE feature (before classifier)

        out = self.classifier(feat)       # → (B, num_classes)

        if return_feat:
            return feat, out
        return out
