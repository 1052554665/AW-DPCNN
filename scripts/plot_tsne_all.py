#!/usr/bin/env python3
"""
Unified t-SNE Generator — Models + Raw Input
==============================================
Generates t‑SNE visualisations for all trained models AND raw input pixels
using consistent parameters.  Outputs go to ``paper/tsne_models/``.

Modes
-----
  models   t-SNE of model features for each trained backbone  (default)
  raw      t-SNE of raw input pixels (no model, PCA→50→t-SNE)
  all      both of the above

Usage::

    # Model t-SNE only
    python scripts/plot_tsne_all.py --mode models --trial trial_seed123

    # Raw input t-SNE only
    python scripts/plot_tsne_all.py --mode raw --data-dir ./datasets/cwru_de/test --max-samples 1500

    # Both
    python scripts/plot_tsne_all.py --mode all --data-dir ./datasets/cwru_de/test --trial trial_seed123

Notes
-----
- t-SNE parameters (perplexity=30, lr=200, max_iter=1000, init="pca") are
  shared with ``src/utils/tsne.py`` for reproducibility.
- Model checkpoints with ``_orig_mod.`` prefix (torch.compile) are handled.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.tsne import extract_features, plot_tsne

# ── Shared constants ───────────────────────────────────────────────────
RESULT_ROOT = Path("experiments/experiment_result/exp1")
OUTPUT_DIR  = Path("paper/tsne_models")
DEFAULT_TRIAL = "trial_seed42"

# t-SNE parameters — must match src/utils/tsne.py
TSNE_PARAMS = dict(perplexity=30, learning_rate=200, max_iter=1000, init="pca")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  Model t-SNE
# ═══════════════════════════════════════════════════════════════════════

def find_model_runs(trial: str = DEFAULT_TRIAL) -> list:
    runs = []
    for exp_dir in sorted(RESULT_ROOT.glob("*")):
        if not exp_dir.is_dir():
            continue
        inner = sorted(exp_dir.glob("exp1_*"))
        if not inner:
            continue
        base = inner[0] / trial
        cfg = base / "resolved_config.yaml"
        ckpt = base / "checkpoints/best.pt"
        if cfg.exists() and ckpt.exists():
            runs.append((exp_dir.name, cfg, ckpt))
    return runs


def generate_model_tsne(device: torch.device, max_samples: int, trial: str = DEFAULT_TRIAL):
    runs = find_model_runs(trial)
    if not runs:
        print("[ERROR] No trained models found.")
        return

    print(f"Found {len(runs)} trained models.\n")
    for name, cfg_path, ckpt_path in runs:
        print(f"[{name}] Loading ...")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.suffix == ".json" else None
        if cfg is None:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

        loaders, class_names = build_dataloaders(cfg, device=device)
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()

        features, labels = extract_features(model, loaders["test"], device)
        if max_samples > 0 and len(features) > max_samples:
            idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
            features, labels = features[idx], labels[idx]

        out = OUTPUT_DIR / f"tsne_{name}.png"
        print(f"  t-SNE → {out}")
        plot_tsne(features, labels, class_names,
                  title=f"{name} — Feature t-SNE", save_path=str(out))
        print("  Done.\n")
    print(f"Model t-SNE plots saved to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════
#  Raw-input t-SNE
# ═══════════════════════════════════════════════════════════════════════

def generate_raw_tsne(data_dir: str, max_samples: int, img_size: int = 224,
                      batch_size: int = 64, pca_dim: int = 50, seed: int = 42):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # ── Load raw images (no normalisation) ──
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(str(data_path), transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)

    feats_list, lbls_list = [], []
    print(f"Loading {len(ds)} raw images ...")
    for x, y in tqdm(loader, desc="Raw images"):
        feats_list.append(x.view(x.size(0), -1).numpy())
        lbls_list.append(y.numpy())
        if max_samples > 0 and len(lbls_list) * batch_size >= max_samples:
            break

    features = np.concatenate(feats_list, axis=0)
    labels   = np.concatenate(lbls_list, axis=0)
    if max_samples > 0 and len(features) > max_samples:
        idx = np.random.default_rng(seed).choice(len(features), max_samples, replace=False)
        features, labels = features[idx], labels[idx]
    print(f"Loaded {len(features)} samples, dim = {features.shape[1]}")

    # ── PCA → t-SNE ──
    pca_dim = min(pca_dim, features.shape[1], features.shape[0])
    print(f"PCA: {features.shape[1]} → {pca_dim} ...")
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    print(f"t-SNE (perplexity={TSNE_PARAMS['perplexity']}) ...")
    tsne = TSNE(n_components=2, random_state=seed, n_jobs=1, **TSNE_PARAMS)
    emb = tsne.fit_transform(reduced)

    # ── Derive dataset name ──
    dataset_name = data_path.parent.name if data_path.name in ("train", "val", "test") else data_path.name
    out = OUTPUT_DIR / f"tsne_{dataset_name}_raw_input.png"

    print(f"Plotting → {out}")
    plot_tsne(emb, labels, ds.classes,
              title=f"Raw Input Pixels — {dataset_name}",
              save_path=str(out))
    print(f"Raw-input t-SNE saved to {out}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Unified t-SNE: models + raw input")
    p.add_argument("--mode", default="models", choices=["models", "raw", "all"])
    p.add_argument("--data-dir", default="./datasets/cwru_de/test",
                   help="ImageFolder dir for raw-input t-SNE (--mode raw/all)")
    p.add_argument("--max-samples", type=int, default=1500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trial", default=DEFAULT_TRIAL,
                   help=f"Trial seed dir name (default: {DEFAULT_TRIAL})")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode in ("models", "all"):
        generate_model_tsne(device, args.max_samples, trial=args.trial)

    if args.mode in ("raw", "all"):
        generate_raw_tsne(args.data_dir, args.max_samples)


if __name__ == "__main__":
    main()
