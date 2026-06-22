#!/usr/bin/env python3
"""
Ablation Experiment Runner — Unified B0–B8
===========================================
Runs the complete 9‑experiment component decomposition ablation:
  B0–B3 : Input representation ablation (Mel, GADF, Concat, AW‑DPCNN γ=1)
  B4    : Full AW‑DPCNN + VGG16 baseline
  B5–B8 : Full AW‑DPCNN + MSCA‑VGG16 variants (MS, CA, EH)

All experiments use identical training settings for fair comparison.
Outputs a consolidated CSV summary.

Usage::

    # Full 9‑experiment ablation (default)
    python scripts/run_ablation_experiments.py

    # Fast test run (2 epochs)
    python scripts/run_ablation_experiments.py --epochs 2 --model vgg16

    # Dry‑run (list experiments only)
    python scripts/run_ablation_experiments.py --dry-run
"""

import argparse
import csv
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.trainers.workflow import train_and_evaluate
from src.utils.config import load_yaml, save_yaml
from src.utils.experiment import dump_json, prepare_run_dir, set_seed

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

ABLATION_EXPERIMENTS: List[Dict] = [
    # ── Input representation ablation (all use VGG16 classifier) ──
    {"id": "B0", "dataset": "datasets/ablation/mel_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "Mel-only (single representation, VGG16)"},
    {"id": "B1", "dataset": "datasets/ablation/gadf_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "GADF-only (single representation, VGG16)"},
    {"id": "B2", "dataset": "datasets/ablation/concat",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "Mel+GADF pixel-wise average (naive fusion, VGG16)"},
    {"id": "B3", "dataset": "datasets/ablation/awdpcnn_gamma1",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "AW-DPCNN γ=1 (fixed-weight PCNN, VGG16)"},
    {"id": "B4", "dataset": "datasets/transformer-five",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "Full AW-DPCNN + VGG16 (fusion contribution baseline)"},
    # ── MSCA component ablation (all use full AW‑DPCNN + MSCA‑VGG16 variants) ──
    {"id": "B5", "dataset": "datasets/transformer-five",
     "model": "msca_vgg16", "ms": True,  "ca": False, "eh": False,
     "desc": "Full AW-DPCNN + MSCA-VGG16 (MS only)"},
    {"id": "B6", "dataset": "datasets/transformer-five",
     "model": "msca_vgg16", "ms": False, "ca": True,  "eh": False,
     "desc": "Full AW-DPCNN + MSCA-VGG16 (CA only)"},
    {"id": "B7", "dataset": "datasets/transformer-five",
     "model": "msca_vgg16", "ms": False, "ca": False, "eh": True,
     "desc": "Full AW-DPCNN + MSCA-VGG16 (EH only)"},
    {"id": "B8", "dataset": "datasets/transformer-five",
     "model": "msca_vgg16", "ms": True,  "ca": True,  "eh": True,
     "desc": "Full AW-DPCNN + MSCA-VGG16 (all components)"},
]

_OUTPUT_ROOT_DEFAULT = "experiments/ablation_results"


def build_config(
    base_config: Dict,
    dataset_path: Path,
    model_name: str,
    num_classes: int,
    output_root: Path,
    exp_id: str,
    epochs: int,
    seed: int,
    use_ms: bool,
    use_ca: bool,
    use_eh: bool,
) -> Dict:
    import copy
    cfg = copy.deepcopy(base_config)

    exp_name = f"ablation_{exp_id}"
    cfg["experiment_name"] = exp_name
    cfg["dataset"]["root_dir"] = str(dataset_path)
    cfg["model"]["name"] = model_name
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["pretrained"] = False
    # MSCA component flags
    cfg["model"]["use_ms"] = use_ms
    cfg["model"]["use_ca"] = use_ca
    cfg["model"]["use_eh"] = use_eh
    cfg["train"]["epochs"] = epochs
    cfg["seed"] = seed
    cfg["output"]["root_dir"] = str(output_root)
    if "visualization" not in cfg:
        cfg["visualization"] = {}
    cfg["visualization"]["tsne"] = False
    cfg["visualization"]["roc_curve"] = False
    return cfg


def resolve_device(requested: str) -> torch.device:
    if not requested or requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified B0–B8 ablation experiment runner",
    )
    p.add_argument("--output-root", default=_OUTPUT_ROOT_DEFAULT,
                   help="Root for per‑run outputs and summary CSV")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Base YAML config")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs per run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--exp-ids", default="all",
                   help="Comma‑separated experiment IDs (e.g. B0,B4,B8). Default: all")
    p.add_argument("--dry-run", action="store_true",
                   help="List experiments without training")
    return p


def main():
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()

    # Filter experiments
    if args.exp_ids == "all":
        selected = ABLATION_EXPERIMENTS
    else:
        ids = set(x.strip() for x in args.exp_ids.split(","))
        selected = [e for e in ABLATION_EXPERIMENTS if e["id"] in ids]

    if not selected:
        print("[ERROR] No experiments selected.")
        sys.exit(1)

    print(f"Experiments to run: {len(selected)}")
    for exp in selected:
        print(f"  {exp['id']}: {exp['desc']}")

    if args.dry_run:
        print("\n[Dry run — no training performed]")
        return

    base_config = load_yaml(args.config)
    device = resolve_device(args.device)
    seed = args.seed
    epochs = args.epochs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_table: List[Dict] = []

    for idx, exp in enumerate(selected, 1):
        exp_id = exp["id"]
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(selected)}]  {exp_id}: {exp['desc']}")
        print(f"{'─'*70}")

        dataset_path = Path(exp["dataset"]).resolve()
        if not dataset_path.is_dir():
            print(f"  [SKIP] Dataset not found: {dataset_path}")
            results_table.append({
                "exp_id": exp_id, "desc": exp["desc"],
                "awdpcnn": str(exp["dataset"] != "datasets/ablation/mel_only"),
                "ms": str(exp["ms"]), "ca": str(exp["ca"]), "eh": str(exp["eh"]),
                "acc": float("nan"), "f1": float("nan"),
                "gmean": float("nan"), "kappa": float("nan"),
                "error": "dataset missing",
            })
            continue

        # Infer num_classes
        train_dir = dataset_path / "train"
        class_names = sorted(
            d.name for d in train_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        num_classes = len(class_names)
        if num_classes < 2:
            print(f"  [SKIP] Only {num_classes} class(es).")
            continue

        # Build config
        run_output_dir = output_root / exp_id / f"seed{seed}"
        config = build_config(
            base_config, dataset_path, exp["model"],
            num_classes, run_output_dir, exp_id, epochs, seed,
            exp["ms"], exp["ca"], exp["eh"],
        )

        run_dir = prepare_run_dir(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(str(run_dir / "resolved_config.yaml"), config)

        set_seed(seed)
        try:
            results = train_and_evaluate(config, run_dir, device, seed=seed)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results_table.append({
                "exp_id": exp_id, "desc": exp["desc"],
                "awdpcnn": str(exp["dataset"] != "datasets/ablation/mel_only"),
                "ms": str(exp["ms"]), "ca": str(exp["ca"]), "eh": str(exp["eh"]),
                "acc": float("nan"), "f1": float("nan"),
                "gmean": float("nan"), "kappa": float("nan"),
                "error": str(exc),
            })
            continue

        dump_json(run_dir / "results" / "test_metrics.json", results)

        row = {
            "exp_id": exp_id,
            "desc": exp["desc"],
            "awdpcnn": "✓" if exp["dataset"] not in ("datasets/ablation/mel_only", "datasets/ablation/gadf_only") else "✗",
            "ms": "✓" if exp["ms"] else "✗",
            "ca": "✓" if exp["ca"] else "✗",
            "eh": "✓" if exp["eh"] else "✗",
            "acc": float(results.get("test_acc", float("nan"))),
            "precision": float(results.get("test_precision", float("nan"))),
            "recall": float(results.get("test_recall", float("nan"))),
            "f1": float(results.get("test_f1", float("nan"))),
            "gmean": float(results.get("test_gmean", float("nan"))),
            "bal_acc": float(results.get("test_bal_acc", float("nan"))),
            "kappa": float(results.get("test_kappa", float("nan"))),
            "auc": float(results.get("test_auc", float("nan"))),
            "error": "",
        }
        results_table.append(row)

        print(f"  Acc={row['acc']*100:.2f}%  F1={row['f1']*100:.2f}%  "
              f"G-mean={row['gmean']*100:.2f}%  Kappa={row['kappa']*100:.2f}%")

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print(f"  ABLATION SUMMARY  |  Epochs: {epochs}  |  Seed: {seed}")
    print(f"{'='*80}")

    if results_table:
        csv_path = output_root / f"ablation_summary_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "exp_id", "desc", "awdpcnn", "ms", "ca", "eh",
            "acc", "precision", "recall", "f1", "gmean", "bal_acc", "kappa", "auc",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_table)
        print(f"\n  CSV saved → {csv_path}\n")

        # Print ranked table
        ranked = sorted(results_table, key=lambda r: r.get("acc", 0.0), reverse=True)
        header = (
            f" {'ID':<4} {'AW-DPCNN':>9} {'MS':>4} {'CA':>4} {'EH':>4}  "
            f"{'Acc %':>7} {'F1 %':>7} {'G-mean %':>9} {'Kappa %':>8}"
        )
        sep = "─" * len(header)
        print(header)
        print(sep)
        for row in ranked:
            def _fmt(v, pct=100):
                if isinstance(v, float) and not (v != v):
                    return f"{v*pct:6.2f}" if pct == 100 else f"{v}"
                return "    N/A"
            print(
                f" {row['exp_id']:<4} {row['awdpcnn']:>9} {row['ms']:>4} "
                f"{row['ca']:>4} {row['eh']:>4}  "
                f"{_fmt(row.get('acc', float('nan')))}  "
                f"{_fmt(row.get('f1', float('nan')))}  "
                f"{_fmt(row.get('gmean', float('nan')))}  "
                f"{_fmt(row.get('kappa', float('nan')))}"
            )

    print(f"\n{'='*80}\n  Done.\n{'='*80}")


if __name__ == "__main__":
    main()
