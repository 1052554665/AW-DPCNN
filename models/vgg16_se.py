import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

# ======================
# SE Module
# ======================
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ======================
# VGG16 + SE
# ======================
class VGG16_SE(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        base = vgg16_bn(pretrained=pretrained)

        self.features = nn.Sequential(
            *base.features[:23],   # conv1-3
            SEModule(256),
            *base.features[23:43], # conv4
            SEModule(512),
            *base.features[43:],   # conv5
            SEModule(512)
        )

        self.avgpool = base.avgpool

        self.fc1 = base.classifier[0]
        self.relu1 = base.classifier[1]
        self.drop1 = base.classifier[2]

        self.fc2 = base.classifier[3]
        self.relu2 = base.classifier[4]
        self.drop2 = base.classifier[5]

        self.fc3 = nn.Linear(
            base.classifier[6].in_features,
            num_classes
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        feat = self.fc2(x)
        x = self.relu2(feat)
        x = self.drop2(x)

        out = self.fc3(x)

        if return_feat:
            return feat, out
        return out
