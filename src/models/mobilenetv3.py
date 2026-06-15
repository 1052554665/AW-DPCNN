"""MobileNetV3-Small wrapper for fault diagnosis classification.

Uses torchvision's ``mobilenet_v3_small`` pre-trained on ImageNet.
Adapted for N-class classification by replacing the final classifier head.
Supports ``return_feat`` for t-SNE feature extraction.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small for image classification.

    Args:
        num_classes: number of output classes.
        pretrained: if True, load ImageNet pre-trained weights.
        dropout: dropout rate for the classifier head (default 0.2).
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # Keep reference to the penultimate layer for feature extraction
        self.feature_dim = self.backbone.classifier[0].out_features

        # Replace the classifier head
        # Original classifier: [Linear(576→1024), Hardswish, Dropout, Linear(1024→1000)]
        in_features = self.backbone.classifier[3].in_features
        hidden_features = self.backbone.classifier[0].out_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x, return_feat=False):
        # Extract features before the final classifier
        feat = self.backbone.features(x)
        feat = self.backbone.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        # Pass through classifier layers except the last
        for layer in self.backbone.classifier[:-1]:
            feat = layer(feat)
        out = self.backbone.classifier[-1](feat)
        if return_feat:
            return out, feat
        return out
