from typing import Dict

from src.models.alexnet_se import SE_AlexNet
from src.models.baseline_lenet import BaselineCNN
from src.models.convnext import ConvNeXt
from src.models.convnext_tiny import ConvNeXtTiny
from src.models.resnet18 import ResNet18
from src.models.resnet18_se import ResNet18_SE
from src.models.vgg16 import VGG16
from src.models.vit import ViT


def build_model(config: Dict):
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    num_classes = int(model_cfg["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True))

    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if name == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    if name == "resnet18_se":
        return ResNet18_SE(num_classes=num_classes, pretrained=pretrained)
    if name == "vgg16":
        return VGG16(num_classes=num_classes, pretrained=pretrained)
    if name == "alexnet_se":
        return SE_AlexNet(num_classes=num_classes, in_channels=int(model_cfg.get("in_channels", 3)))
    if name == "convnext":
        return ConvNeXt(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return ConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)
    if name == "vit":
        return ViT(
            num_classes=num_classes,
            img_size=int(model_cfg.get("img_size", config["data"].get("img_size", 224))),
            patch_size=int(model_cfg.get("patch_size", 16)),
            in_chans=int(model_cfg.get("in_channels", 3)),
            embed_dim=int(model_cfg.get("embed_dim", 384)),
            depth=int(model_cfg.get("depth", 6)),
            num_heads=int(model_cfg.get("num_heads", 6)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            drop=float(model_cfg.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model name: {name}")

