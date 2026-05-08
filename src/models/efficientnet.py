import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.classifier(feat)

        if return_feat:
            return feat, logits
        return logits


def build_efficientnet_b0(num_classes=10, pretrained=True):
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)

