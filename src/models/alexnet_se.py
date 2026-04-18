import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class SE_AlexNet(nn.Module):
    """
    AlexNet-style CNN with SE attention.
    Supports return_feat for t-SNE / feature visualization.
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        # ===== Feature extractor =====
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEModule(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEModule(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEModule(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEModule(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEModule(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEModule(512),
            nn.MaxPool2d(2),
        )

        # ===== Global pooling =====
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, return_feat=False):
        feat_map = self.features(x)          # [B, 512, H, W]
        pooled = self.avgpool(feat_map)      # [B, 512, 1, 1]
        feat = torch.flatten(pooled, 1)      # [B, 512]

        logits = self.classifier(feat)

        if return_feat:
            return logits, feat
        else:
            return logits
