import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn

class VGG16_Embed(nn.Module):
    def __init__(self, num_classes=5, embed_dim=1024, pretrained=True):
        super().__init__()
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16_bn(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU()
        )

        self.embed = nn.Sequential(
            nn.Linear(4096, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        feat = self.embed(x)
        out = self.classifier(feat)
        return (feat, out) if return_feat else out
