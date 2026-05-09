import argparse

import torch

from src.models import build_model
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight workflow smoke test without dataset access.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp-config", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.exp_config)

    model = build_model(cfg)
    model.eval()

    img_size = int(cfg.get("dataset", {}).get("img_size", 224))
    in_channels = int(cfg.get("model", {}).get("in_channels", 3))
    dummy = torch.randn(args.batch_size, in_channels, img_size, img_size)

    with torch.no_grad():
        output = model(dummy)

    logits = output[0] if isinstance(output, tuple) else output
    print(f"model={cfg['model']['name']}")
    print(f"input_shape={tuple(dummy.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()

