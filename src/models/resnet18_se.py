# Channel-wise attention is introduced via
# squeeze-and-excitation blocks to enhance discriminative acoustic feature learning.

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = nn.Sequential(
            base.layer1,
            SEBlock(64)
        )
        self.layer2 = nn.Sequential(
            base.layer2,
            SEBlock(128)
        )
        self.layer3 = nn.Sequential(
            base.layer3,
            SEBlock(256)
        )
        self.layer4 = nn.Sequential(
            base.layer4,
            SEBlock(512)
        )

        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = x.flatten(1)

        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out
