import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn

class MSCA_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.conv_dilated = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv3(x) + self.conv5(x) + self.conv_dilated(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out * self.se(out)
        return out


class MSCA_VGG16(nn.Module):
    """Lightweight MSCA-VGG16 with global average pooling.

    Uses VGG16_bn backbone + MSCA block + compact classification head.
    Global average pooling replaces the original 7×7 pool, shrinking the
    classifier from ~103M to ~0.5M parameters — a >200× reduction that
    prevents overfitting on small/medium fault diagnosis datasets.

    Args:
        num_classes: number of output classes (default 4).
        pretrained:  load ImageNet pre-trained weights.
        embed_dim:   embedding dimension (default 256, was 1024).
        dropout:     dropout rate in fc/embed layers (default 0.5).
    """

    def __init__(self, num_classes=4, pretrained=True, embed_dim=256, dropout=0.5):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16_bn(weights=weights)

        self.features = base.features              # → 512 × 7 × 7
        self.msca = MSCA_Block(512)                # → 512 × 7 × 7
        self.avgpool = nn.AdaptiveAvgPool2d(1)     # → 512 × 1 × 1  (was 7×7)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embed = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.msca(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        feat = self.embed(x)
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out
