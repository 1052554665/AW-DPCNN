import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

# def build_convnext(num_classes=10, pretrained=True):
#     model = convnext_tiny(pretrained=pretrained)
#     model.classifier[2] = nn.Linear(
#         model.classifier[2].in_features,
#         num_classes
#     )
#     return model


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        base_model = convnext_tiny(pretrained=pretrained)

        # backbone
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # 分类头
        in_features = base_model.classifier[2].in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)   # ← t-SNE 用这个特征
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out


def build_convnext(num_classes=10, pretrained=True):
    return ConvNeXt(num_classes, pretrained)
