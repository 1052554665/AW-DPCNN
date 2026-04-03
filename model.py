# This model has two features:
# 1. SE module and Squeeze-and-Excitation module.
# 2. The convolutional layers use a kernel size of (3, 1) to focus on temporal features while preserving spatial information.
import torch.nn as nn
import torch

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class AlexNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(3, 1), stride=8, padding=(2, 0)),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            SEModule(12),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.Conv2d(12, 32, kernel_size=(3, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEModule(32),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
            nn.Conv2d(32, 48, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SEModule(48),
            nn.Conv2d(48, 48, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            SEModule(48),
            nn.Conv2d(48, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEModule(32),
            nn.MaxPool2d(kernel_size=(3, 1), stride=2),
        )

        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = self.features(dummy_input)
        num_features = dummy_output.numel() // dummy_input.size(0)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

