# Classical deep CNN with strong texture modeling capability.
# 对于VGG16：使用 classifier 中最后一层 FC 之前的特征（4096 维）, t-SNE 用的是第二个 FC 输出（进入最后分类层之前）
import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn

class VGG16(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = vgg16_bn(weights=weights)
        # base_model = vgg16_bn(pretrained=False)

        # backbone
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # classifier（拆开）
        self.fc1 = base_model.classifier[0]   # 4096
        self.relu1 = base_model.classifier[1]
        self.drop1 = base_model.classifier[2]

        self.fc2 = base_model.classifier[3]   # 4096
        self.relu2 = base_model.classifier[4]
        self.drop2 = base_model.classifier[5]

        self.fc3 = nn.Linear(
            base_model.classifier[6].in_features,
            num_classes
        )

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        feat = self.fc2(x)     # ← t-SNE 用这个特征
        x = self.relu2(feat)
        x = self.drop2(x)

        out = self.fc3(x)

        if return_feat:
            return feat, out
        return out


def build_vgg16(num_classes=10, pretrained=True):
    return VGG16(num_classes, pretrained)
