#!/usr/bin/env python3
"""
Representation Comparison — Automated Benchmark
================================================
Automatically loads every representation variant under ``datasets/rep_compare/``,
trains the **same** classifier on each, and produces a side‑by‑side comparison
of test‑set metrics.

Run with a single command::

    python scripts/run_representation_comparison.py

Key features
------------
- Auto‑discovers all 12 (TF × temporal) combinations.
- Trains with the identical training protocol → fair comparison.
- Saves per‑run checkpoints / logs / figures.
- Outputs a consolidated CSV summary and a formatted ranking table.

Supported models (via ``--model``)::

    msca_vgg16  (default)    vgg16       convnext_tiny
    efficientnet_b0           mobilenetv3_small
    harmonic_cnn              cnn_lstm      vit

Usage::

    # Full 12‑way comparison with default MSCA‑VGG16
    python scripts/run_representation_comparison.py

    # Single combination, fast test
    python scripts/run_representation_comparison.py --tf mel --temporal gadf

    # Only Mel‑based variants
    python scripts/run_representation_comparison.py --tf mel

    # Custom model and fewer epochs (fast baseline sweep)
    python scripts/run_representation_comparison.py --model vgg16 --epochs 15
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

# ── Ensure project root is on sys.path ───────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets import build_dataloaders
from src.models import build_model
from src.trainers.workflow import train_and_evaluate
from src.utils.config import load_yaml, save_yaml
from src.utils.experiment import dump_json, prepare_run_dir, set_seed

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_REP_ROOT_DEFAULT = "datasets/rep_compare"
_OUTPUT_ROOT_DEFAULT = "experiments/rep_compare_results"

TF_METHODS = ["mel", "stft", "cwt"]
TEMPORAL_METHODS = ["gadf", "gasf", "mtf", "rp"]

# Pretty display names for the report table
_DISPLAY_TF: Dict[str, str] = {"mel": "Mel", "stft": "STFT", "cwt": "CWT"}
_DISPLAY_TE: Dict[str, str] = {
    "gadf": "GADF", "gasf": "GASF", "mtf": "MTF", "rp": "RP",
}


def discover_combinations(
    rep_root: Path,
    tf_filter: Optional[List[str]] = None,
    te_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, Path]]:
    """Return sorted list of (tf, temporal, dataset_path) for existing dirs."""
    tf_list = tf_filter or TF_METHODS
    te_list = te_filter or TEMPORAL_METHODS
    results: List[Tuple[str, str, Path]] = []
    for tf_name in tf_list:
        for te_name in te_list:
            combo = f"{tf_name}_{te_name}"
            combo_path = rep_root / combo
            if combo_path.is_dir() and (combo_path / "train").is_dir():
                results.append((tf_name, te_name, combo_path))
    return results


def build_config(
    base_config: Dict,
    dataset_path: Path,
    model_name: str,
    num_classes: int,
    output_root: Path,
    combo_name: str,
    epochs: int,
    seed: int,
) -> Dict:
    """Create a run‑specific config by shallow‑updating the base."""
    import copy
    cfg = copy.deepcopy(base_config)

    cfg["experiment_name"] = f"rep_compare_{model_name}_{combo_name}"
    cfg["dataset"]["root_dir"] = str(dataset_path)
    cfg["model"]["name"] = model_name
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["pretrained"] = False
    cfg["train"]["epochs"] = epochs
    cfg["seed"] = seed
    cfg["output"]["root_dir"] = str(output_root)
    # Disable heavy visualisation per run to save disk
    if "visualization" not in cfg:
        cfg["visualization"] = {}
    cfg["visualization"]["tsne"] = False
    cfg["visualization"]["roc_curve"] = False
    return cfg


def resolve_device(requested: str) -> torch.device:
    if not requested or requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Representation comparison benchmark — one command, full table.",
    )
    p.add_argument("--rep-root", default=_REP_ROOT_DEFAULT,
                   help="Root of representation comparison datasets")
    p.add_argument("--output-root", default=_OUTPUT_ROOT_DEFAULT,
                   help="Root for per‑run outputs and summary CSV")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Base YAML config (training hyper‑params)")
    p.add_argument("--model", default="msca_vgg16",
                   help="Classifier model name")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs per run")
    p.add_argument("--tf", default="all",
                   help="TF method(s): mel, stft, cwt, all (comma‑sep)")
    p.add_argument("--temporal", default="all",
                   help="Temporal method(s): gadf, gasf, mtf, rp, all (comma‑sep)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--device", default="auto",
                   help="Device override (cuda, cpu)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print combinations without training")
    return p


def main():
    args = build_parser().parse_args()

    rep_root = Path(args.rep_root).resolve()
    output_root = Path(args.output_root).resolve()

    # Parse filters
    tf_filter = None if args.tf == "all" else [x.strip() for x in args.tf.split(",")]
    te_filter = None if args.temporal == "all" else [x.strip() for x in args.temporal.split(",")]

    # Discover
    combinations = discover_combinations(rep_root, tf_filter, te_filter)
    if not combinations:
        print(f"[ERROR] No dataset directories found under {rep_root}")
        sys.exit(1)

    print(f"Found {len(combinations)} representation variant(s):")
    for tf_name, te_name, _ in combinations:
        print(f"  {_DISPLAY_TF.get(tf_name, tf_name)} × {_DISPLAY_TE.get(te_name, te_name)}")

    if args.dry_run:
        print("\n[Dry run — no training performed]")
        return

    # Load base config
    base_config = load_yaml(args.config)

    device = resolve_device(args.device)
    model_name = args.model
    seed = args.seed
    epochs = args.epochs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nModel: {model_name}  |  Epochs: {epochs}  |  Seed: {seed}  |  Device: {device}")
    print(f"Output root: {output_root}")
    print(f"{'='*80}")

    results_table: List[Dict] = []

    for idx, (tf_name, te_name, dataset_path) in enumerate(combinations, 1):
        combo_name = f"{tf_name}_{te_name}"
        tf_label = _DISPLAY_TF.get(tf_name, tf_name)
        te_label = _DISPLAY_TE.get(te_name, te_name)
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(combinations)}]  {tf_label} × {te_label}  ({combo_name})")
        print(f"  Dataset: {dataset_path}")
        print(f"{'─'*70}")

        # Infer num_classes from train folder
        train_dir = dataset_path / "train"
        class_names = sorted(
            d.name for d in train_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        num_classes = len(class_names)
        if num_classes < 2:
            print(f"  [SKIP] Found only {num_classes} class(es); need ≥2.")
            continue

        # Build run config
        run_output_dir = output_root / model_name / combo_name / f"seed{seed}"
        config = build_config(
            base_config, dataset_path, model_name,
            num_classes, run_output_dir, combo_name, epochs, seed,
        )

        # Prepare run dir
        run_dir = prepare_run_dir(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(str(run_dir / "resolved_config.yaml"), config)

        # ── Train ──
        set_seed(seed)
        try:
            results = train_and_evaluate(config, run_dir, device, seed=seed)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results_table.append({
                "tf": tf_name, "temporal": te_name,
                "combo": combo_name,
                "tf_label": tf_label, "te_label": te_label,
                "acc": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "gmean": float("nan"), "bal_acc": float("nan"),
                "kappa": float("nan"), "auc": float("nan"),
                "params": 0, "flops": 0,
                "error": str(exc),
            })
            continue

        # Save JSON (train_and_evaluate already saves in run_dir)
        dump_json(run_dir / "results" / "test_metrics.json", results)

        row = {
            "tf": tf_name, "temporal": te_name,
            "combo": combo_name,
            "tf_label": tf_label, "te_label": te_label,
            "acc": float(results.get("test_acc", float("nan"))),
            "precision": float(results.get("test_precision", float("nan"))),
            "recall": float(results.get("test_recall", float("nan"))),
            "f1": float(results.get("test_f1", float("nan"))),
            "gmean": float(results.get("test_gmean", float("nan"))),
            "bal_acc": float(results.get("test_bal_acc", float("nan"))),
            "kappa": float(results.get("test_kappa", float("nan"))),
            "auc": float(results.get("test_auc", float("nan"))),
            "params": int(results.get("params", 0)),
            "flops": int(results.get("flops", 0)),
            "error": "",
        }
        results_table.append(row)

        # Per‑run summary
        auc_str = f", AUC={row.get('auc', 0)*100:.2f}%" if not (isinstance(row.get('auc'), float) and np.isnan(row.get('auc'))) else ""
        print(f"  Acc={row['acc']*100:.2f}%  F1={row['f1']*100:.2f}%  G-mean={row['gmean']*100:.2f}%  Kappa={row['kappa']*100:.2f}%{auc_str}")

    # ═══════════════════════════════════════════════════════════════
    #  Summary output
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  REPRESENTATION COMPARISON — SUMMARY")
    print(f"  Model: {model_name}  |  Epochs: {epochs}  |  Seed: {seed}")
    print(f"{'='*80}")

    if results_table:
        # ── CSV ──
        csv_path = output_root / model_name / f"comparison_summary_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tf", "temporal", "combo", "tf_label", "te_label",
            "acc", "precision", "recall", "f1", "gmean", "bal_acc", "kappa",
            "auc", "params", "flops",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_table)
        print(f"\n  CSV saved → {csv_path}\n")

        # ── Ranked Table ──
        # Sort by accuracy descending
        ranked = sorted(results_table, key=lambda r: r.get("acc", 0.0), reverse=True)

        header = (
            f" {'Rank':<4}  {'TF':<5}  {'Temporal':<8}  "
            f"{'Acc %':>7}  {'F1 %':>7}  {'G-mean %':>9}  {'B-Acc %':>8}  {'Kappa %':>8}"
        )
        sep = "─" * len(header)
        print(header)
        print(sep)
        for rank, row in enumerate(ranked, 1):
            acc_str = f"{row['acc']*100:6.2f}" if not (isinstance(row['acc'], float) and np.isnan(row['acc'])) else "    N/A"
            f1_str = f"{row['f1']*100:6.2f}" if not (isinstance(row['f1'], float) and np.isnan(row['f1'])) else "    N/A"
            gm_str = f"{row['gmean']*100:8.2f}" if not (isinstance(row['gmean'], float) and np.isnan(row['gmean'])) else "     N/A"
            ba_str = f"{row['bal_acc']*100:7.2f}" if not (isinstance(row['bal_acc'], float) and np.isnan(row['bal_acc'])) else "    N/A"
            ka_str = f"{row['kappa']*100:7.2f}" if not (isinstance(row['kappa'], float) and np.isnan(row['kappa'])) else "    N/A"

            tf_label = row.get("tf_label", row["tf"])
            te_label = row.get("te_label", row["temporal"])
            print(
                f" {rank:<4}  {tf_label:<5}  {te_label:<8}  "
                f"{acc_str}  {f1_str}  {gm_str}  {ba_str}  {ka_str}"
            )

        # ── Grouped by TF ──
        print(f"\n\n{'─'*70}")
        print("  Grouped by Time‑Frequency method (average Accuracy %)")
        print(f"{'─'*70}")
        for tf_name in TF_METHODS:
            tf_label = _DISPLAY_TF.get(tf_name, tf_name)
            group = [r for r in results_table if r["tf"] == tf_name]
            if group:
                avg_acc = np.mean([r["acc"] for r in group
                                   if not (isinstance(r["acc"], float) and np.isnan(r["acc"]))])
                print(f"  {tf_label:<5}  avg Acc = {avg_acc*100:.2f}%  (n={len(group)})")

        # ── Grouped by Temporal ──
        print(f"\n  Grouped by Temporal encoding method (average Accuracy %)")
        print(f"{'─'*70}")
        for te_name in TEMPORAL_METHODS:
            te_label = _DISPLAY_TE.get(te_name, te_name)
            group = [r for r in results_table if r["temporal"] == te_name]
            if group:
                avg_acc = np.mean([r["acc"] for r in group
                                   if not (isinstance(r["acc"], float) and np.isnan(r["acc"]))])
                print(f"  {te_label:<8}  avg Acc = {avg_acc*100:.2f}%  (n={len(group)})")

        # ── Best combination ──
        best = ranked[0]
        if not (isinstance(best.get("acc"), float) and np.isnan(best.get("acc"))):
            print(f"\n  ★ Best: {best['tf_label']} × {best['te_label']}  "
                  f"→ Acc = {best['acc']*100:.2f}%, F1 = {best['f1']*100:.2f}%")

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")


if __name__ == "__main__":
    import numpy as np  # for summary stats
    main()
