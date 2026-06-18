from typing import Dict

from src.models.convnext_tiny import ConvNeXtTiny
from src.models.cnn_lstm import CNNLSTM
from src.models.efficientnet import EfficientNetB0
from src.models.harmonic_cnn import HarmonicCNN
from src.models.mobilenetv3 import MobileNetV3Small
from src.models.MSCA_VGG16 import MSCA_VGG16
from src.models.vgg16 import VGG16
from src.models.vit import ViT


def build_model(config: Dict):
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    num_classes = int(model_cfg["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True))

    if name == "vgg16":
        return VGG16(num_classes=num_classes, pretrained=pretrained)

    if name == "convnext_tiny":
        return ConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)

    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnet"}:
        return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)

    if name in {"mobilenetv3", "mobilenet_v3", "mobilenetv3_small"}:
        return MobileNetV3Small(num_classes=num_classes, pretrained=pretrained)

    if name == "harmonic_cnn":
        return HarmonicCNN(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 3)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )

    if name in {"cnn_lstm", "cnn-lstm", "cnnlstm"}:
        return CNNLSTM(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 3)),
            cnn_out_channels=int(model_cfg.get("cnn_out_channels", 128)),
            lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )

    if name == "msca_vgg16":
        return MSCA_VGG16(
            num_classes=num_classes,
            pretrained=pretrained,
            use_ms=bool(model_cfg.get("use_ms", True)),
            use_ca=bool(model_cfg.get("use_ca", True)),
            use_eh=bool(model_cfg.get("use_eh", True)),
        )

    if name == "vit":
        return ViT(
            num_classes=num_classes,
            img_size=int(model_cfg.get("img_size", config["dataset"].get("img_size", 224))),
            patch_size=int(model_cfg.get("patch_size", 16)),
            in_chans=int(model_cfg.get("in_channels", 3)),
            embed_dim=int(model_cfg.get("embed_dim", 384)),
            depth=int(model_cfg.get("depth", 6)),
            num_heads=int(model_cfg.get("num_heads", 6)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            drop=float(model_cfg.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model name: {name}")

