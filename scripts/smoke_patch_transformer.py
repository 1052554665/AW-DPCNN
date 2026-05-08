import torch

from src.models.patch_transformer import PatchTransformer


def main():
    model = PatchTransformer(num_classes=10, img_size=224, patch_size=16, in_chans=3)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()

