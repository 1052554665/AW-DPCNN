import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class MSSEBlock(nn.Module):
    """
    Multi-Scale Squeeze-and-Excitation Block
    (GAP + GMP)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)

        # 多尺度特征拼接
        feat = torch.cat([avg_feat, max_feat], dim=1)

        weight = self.fc(feat).view(b, c, 1, 1)
        return x * weight

class ResNet50_MSSE(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        base = resnet50(pretrained=pretrained)

        # Stem
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Backbone + MS-SE
        self.layer1 = nn.Sequential(
            base.layer1,
            MSSEBlock(256)
        )
        self.layer2 = nn.Sequential(
            base.layer2,
            MSSEBlock(512)
        )
        self.layer3 = nn.Sequential(
            base.layer3,
            MSSEBlock(1024)
        )
        self.layer4 = nn.Sequential(
            base.layer4,
            MSSEBlock(2048)
        )

        self.avgpool = base.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_feat=False):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out
