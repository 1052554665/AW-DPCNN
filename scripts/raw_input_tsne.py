#!/usr/bin/env python3
"""
Raw Input t-SNE Visualization
==============================
Visualises t-SNE embeddings of **raw input pixels** (before any model
transformation) to check whether the input features are already linearly
separable.

If raw pixel features already form well‑separated clusters, high
classification accuracy may be a trivial consequence of the input
representation rather than meaningful learned patterns.

Usage::
    # Group2_4_harmonic dataset
    python scripts/raw_input_tsne.py --data-dir ./datasets/Group2_4_harmonic/test --output ./experiments/tsne_raw_input/ --max-samples 2000

    # CWRU dataset
    python scripts/raw_input_tsne.py --data-dir ./datasets/cwru_within/test --output ./experiments/tsne_raw_input/ --max-samples 2000

Notes
-----
- Uses the **test** split by default; pass `--data-dir` directly to any
  ImageFolder‑compatible directory.
- Raw 224×224×3 = 150 528 dimensions — t-SNE is expensive on raw pixels.
  The script applies **PCA → 50 dims** before t-SNE by default, which is
  standard practice for raw‑pixel t-SNE.
- NO normalisation is applied (no ImageNet mean/std, no min‑max scaling).
  The only pre‑processing is `Resize` + `ToTensor`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Raw-input t-SNE: check if raw pixels are linearly separable",
    )
    p.add_argument("--data-dir", required=True,
                   help="Path to an ImageFolder directory (e.g. datasets/cwru_within/test)")
    p.add_argument("--output", required=True,
                   help="Output path for the t-SNE PNG")
    p.add_argument("--img-size", type=int, default=224,
                   help="Resize images to this size (default: 224)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for loading images")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers")
    p.add_argument("--pca-dim", type=int, default=50,
                   help="PCA target dimension before t-SNE (default: 50)")
    p.add_argument("--perplexity", type=float, default=30,
                   help="t-SNE perplexity (default: 30)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap total samples (0 = use all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def load_raw_images(data_dir: str, img_size: int, batch_size: int,
                    num_workers: int, max_samples: int):
    """Load images from an ImageFolder directory with MINIMAL transforms.

    Returns
    -------
    features : np.ndarray  shape [N, C*H*W]  — flattened raw pixel values.
    labels   : np.ndarray  shape [N,]        — integer class labels.
    classes  : List[str]                     — class names.
    """
    # Minimal transform: only resize + ToTensor. NO normalisation.
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # scales to [0, 1]
    ])

    ds = datasets.ImageFolder(data_dir, transform=tf)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    features_list = []
    labels_list = []

    print(f"Loading {len(ds)} images from {data_dir}  (classes: {ds.classes})")

    for x, y in tqdm(loader, desc="Loading raw images"):
        # x: [B, C, H, W] in [0, 1]
        features_list.append(x.view(x.size(0), -1).numpy())  # flatten
        labels_list.append(y.numpy())

        if max_samples > 0 and len(labels_list) * batch_size >= max_samples:
            break

    feats = np.concatenate(features_list, axis=0)
    lbls = np.concatenate(labels_list, axis=0)

    if max_samples > 0 and len(feats) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(feats), max_samples, replace=False)
        feats = feats[idx]
        lbls = lbls[idx]

    print(f"Loaded {len(feats)} samples, feature dim = {feats.shape[1]}")
    return feats, lbls, ds.classes


def _resolve_output_path(output_arg: str, data_dir: str) -> Path:
    """Resolve --output to a concrete .png file path.

    If ``output_arg`` is a directory (ends with / or already exists as a dir),
    auto‑generate a filename from the dataset name.
    """
    out = Path(output_arg)
    if output_arg.endswith("/") or (out.exists() and out.is_dir()):
        # Derive a sensible name from the data directory
        data_path = Path(data_dir)
        # e.g. "cwru_within" from "datasets/cwru_within/test"
        # Walk up to find the dataset root (parent of train/val/test)
        dataset_name = data_path.parent.name if data_path.name in ("train", "val", "test") else data_path.name
        out = out / f"tsne_{dataset_name}_raw_input.png"
    elif out.suffix != ".png":
        out = out.with_suffix(".png")

    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    args = parse_args()

    # Resolve output path (handles directory input gracefully)
    out_path = _resolve_output_path(args.output, args.data_dir)

    # ── 1. Load raw images ──
    features, labels, class_names = load_raw_images(
        args.data_dir, args.img_size, args.batch_size,
        args.num_workers, args.max_samples,
    )

    # ── 2. PCA dimensionality reduction ──
    pca_dim = min(args.pca_dim, features.shape[1], features.shape[0])
    print(f"Running PCA: {features.shape[1]} → {pca_dim} ...")
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    features_reduced = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ── 3. t-SNE ──
    print(f"Running t-SNE (perplexity={args.perplexity}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=args.seed,
        n_jobs=1,
    )
    emb = tsne.fit_transform(features_reduced)

    # ── 4. Plot ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        idx = labels == i
        ax.scatter(
            emb[idx, 0], emb[idx, 1],
            s=8, alpha=0.8, label=name,
            edgecolors="none", color=colors[i],
        )

    ax.legend(loc="best", frameon=False, markerscale=1.5)
    ax.grid(False)

    # Add a title indicating these are raw input features
    ax.set_title("t-SNE of Raw Input Pixels (no model)", fontsize=13, pad=10)

    plt.tight_layout()

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE to {out_path}")


if __name__ == "__main__":
    main()
