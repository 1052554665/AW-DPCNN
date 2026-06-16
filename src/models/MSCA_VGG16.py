import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn

class MSCA_Block(nn.Module):
    """Multi-scale convolution + channel attention block.

    Args:
        in_channels: input channels.
        use_ms: enable multi-scale conv branches (3×3, 5×5, dilated 3×3).
                If False, falls back to a single 3×3 conv.
        use_ca: enable SE-style channel attention.
    """

    def __init__(self, in_channels, use_ms=True, use_ca=True):
        super().__init__()
        self.use_ms = use_ms
        self.use_ca = use_ca

        if use_ms:
            self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
            self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
            self.conv_dilated = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        else:
            self.conv_single = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_ca:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if self.use_ms:
            out = self.conv3(x) + self.conv5(x) + self.conv_dilated(x)
        else:
            out = self.conv_single(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.use_ca:
            out = out * self.se(out)
        return out


class MSCA_VGG16(nn.Module):
    """Lightweight MSCA-VGG16 with component ablation support.

    Args:
        num_classes: number of output classes (default 4).
        pretrained:  load ImageNet pre-trained weights.
        embed_dim:   embedding dimension (default 256).
        dropout:     dropout rate in fc/embed layers (default 0.5).
        use_ms:      enable multi-scale convolution (default True).
        use_ca:      enable channel attention (default True).
        use_eh:      enable embedding head; if False, fc1→classifier (default True).
    """

    def __init__(self, num_classes=4, pretrained=True, embed_dim=256, dropout=0.5,
                 use_ms=True, use_ca=True, use_eh=True):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16_bn(weights=weights)

        self.features = base.features              # → 512 × 7 × 7
        self.use_ms = use_ms
        self.use_ca = use_ca
        self.use_eh = use_eh

        # MSCA block — skip entirely if both MS and CA are disabled
        if use_ms or use_ca:
            self.msca = MSCA_Block(512, use_ms=use_ms, use_ca=use_ca)
        else:
            self.msca = None

        self.avgpool = nn.AdaptiveAvgPool2d(1)     # → 512 × 1 × 1

        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if use_eh:
            self.embed = nn.Sequential(
                nn.Linear(1024, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.embed = None
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        if self.msca is not None:
            x = self.msca(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        if self.use_eh:
            feat = self.embed(x)
        else:
            feat = x
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out
