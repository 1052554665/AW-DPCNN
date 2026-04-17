import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


class VGG16_Physics(nn.Module):
    def __init__(self, num_classes=5, embed_dim=512, pretrained=True):
        super().__init__()
        base = vgg16_bn(pretrained=pretrained)

        self.features = base.features
        self.avgpool = base.avgpool

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 物理友好特征层（核心）
        self.embed = nn.Sequential(
            nn.Linear(4096, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        feat = self.embed(x)
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out

