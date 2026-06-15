from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_transforms(img_size: int, augment: bool, normalize_mean: List[float], normalize_std: List[float]):
    train_ops = []
    eval_ops = []

    if img_size > 0:
        train_ops.append(transforms.Resize((img_size, img_size)))
        eval_ops.append(transforms.Resize((img_size, img_size)))

    if augment:
        train_ops.extend([
            transforms.RandomResizedCrop(img_size if img_size > 0 else 224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    eval_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_dataloaders(config: Dict, device: Optional[torch.device] = None) -> Tuple[Dict[str, DataLoader], List[str]]:
    data_cfg = config["dataset"]
    root_dir = Path(data_cfg["root_dir"]).expanduser().resolve()

    train_dir = root_dir / data_cfg.get("train_split", "train")
    val_dir = root_dir / data_cfg.get("val_split", "val")
    test_dir = root_dir / data_cfg.get("test_split", "test")

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected train/val/test folders under dataset.root_dir. "
            f"Got: {train_dir}, {val_dir}, {test_dir}"
        )

    train_tf, eval_tf = _build_transforms(
        img_size=int(data_cfg.get("img_size", 224)),
        augment=bool(data_cfg.get("augmentation", True)),
        normalize_mean=data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
    )

    train_set = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_set = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    test_set = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    if train_set.class_to_idx != val_set.class_to_idx or train_set.class_to_idx != test_set.class_to_idx:
        raise ValueError("Class index mapping mismatch across train/val/test splits.")

    pin_memory_requested = bool(data_cfg.get("pin_memory", True))
    pin_memory = pin_memory_requested and bool(device is not None and device.type == "cuda")
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(data_cfg.get("batch_size", 32))

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=bool(num_workers > 0),
        ),
    }

    return loaders, train_set.classes

