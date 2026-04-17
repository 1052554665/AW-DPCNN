# ConvNeXt-Tiny is adopted as a modern CNN baseline that
# bridges the performance gap between classical CNNs and vision transformers.
# 对于ConvNeXt-Tiny：使用 classifier 前的全局特征（Global Feature）。t-SNE 用的就是 flatten 后、进入 classifier 之前的特征

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = convnext_tiny(weights=weights)

        # backbone
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # 分类头
        in_features = base_model.classifier[2].in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_feat=False):
        """
        return_feat = False → 正常训练 / test
        return_feat = True  → t-SNE 特征提取
        """
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)   # ← t-SNE 用这个
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out


def build_convnext_tiny(num_classes=10, pretrained=True):
    return ConvNeXtTiny(num_classes, pretrained)
