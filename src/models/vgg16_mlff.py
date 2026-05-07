import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn

class VGG16_MultiLevel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16_bn(weights=weights)

        self.block3 = nn.Sequential(*base.features[:23])
        self.block4 = nn.Sequential(*base.features[23:43])
        self.block5 = nn.Sequential(*base.features[43:])

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256 + 512 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, return_feat=False):
        x3 = self.block3(x)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        f3 = self.gap(x3).flatten(1)
        f4 = self.gap(x4).flatten(1)
        f5 = self.gap(x5).flatten(1)

        feat = torch.cat([f3, f4, f5], dim=1)
        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out

