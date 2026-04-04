# ImageNet 预训练
# 对于ResNet18：Global Average Pooling 之后、FC 之前的 512 维特征

import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)

        # 保留 backbone 的 fc 前特征维度
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)     # 512-d 特征（GAP 后）
        logits = self.classifier(feat)

        if return_feat:
            return logits, feat
        else:
            return logits

