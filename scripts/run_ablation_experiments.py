#!/usr/bin/env python3
"""
Unified Ablation Experiment Runner — B0–B8
===========================================
Runs the complete 9‑experiment ablation study on the CWRU 12k DE dataset.

Design
------
  **Fusion‑level ablation** (B0–B4):  standard VGG16 classifier.
      B0 — Mel‑only                  (single representation)
      B1 — GADF‑only                 (single representation)
      B2 — Concat (pixel average)    (naive fusion)
      B3 — AW‑DPCNN  γ=1             (equal‑weight PCNN, no adaptivity)
      B4 — AW‑DPCNN  γ=10            (full adaptive fusion — baseline)

  **Classifier‑level ablation** (B5–B8):  all use B4 fused images.
      B5 — VGG16 + MS  (multi‑scale convolution only)
      B6 — VGG16 + CA  (channel attention only)
      B7 — VGG16 + EH  (embedding head only)
      B8 — MSCA‑VGG16  (MS + CA + EH — full model)

All experiments share identical training settings for fair comparison.
Outputs a consolidated CSV summary in ``experiments/ablation_results/``.

Usage::

    python scripts/run_ablation_experiments.py --epochs 30

    # Dry‑run
    python scripts/run_ablation_experiments.py --dry-run

    # Single experiment
    python scripts/run_ablation_experiments.py --exp-ids B0,B8
"""

import argparse
import csv
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.trainers.workflow import train_and_evaluate
from src.utils.config import load_yaml
from src.utils.experiment import dump_json, prepare_run_dir, set_seed

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS: List[Dict] = [
    # ── Fusion‑level: VGG16 classifier ──
    {"id": "B0", "dataset": "datasets/ablation/mel_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "Mel-only"},
    {"id": "B1", "dataset": "datasets/ablation/gadf_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "GADF-only"},
    {"id": "B2", "dataset": "datasets/ablation/concat",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "Mel+GADF concat (pixel avg)"},
    {"id": "B3", "dataset": "datasets/ablation/awdpcnn_gamma1",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "AW-DPCNN γ=1 (equal weight)"},
    {"id": "B4", "dataset": "datasets/ablation/awdpcnn_full",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "AW-DPCNN γ=10 — fusion baseline"},

    # ── Classifier‑level: full AW‑DPCNN fusion ──
    {"id": "B5", "dataset": "datasets/ablation/awdpcnn_full",
     "model": "msca_vgg16", "ms": True,  "ca": False, "eh": False,
     "desc": "AW-DPCNN + MS only"},
    {"id": "B6", "dataset": "datasets/ablation/awdpcnn_full",
     "model": "msca_vgg16", "ms": False, "ca": True,  "eh": False,
     "desc": "AW-DPCNN + CA only"},
    {"id": "B7", "dataset": "datasets/ablation/awdpcnn_full",
     "model": "msca_vgg16", "ms": False, "ca": False, "eh": True,
     "desc": "AW-DPCNN + EH only"},
    {"id": "B8", "dataset": "datasets/ablation/awdpcnn_full",
     "model": "msca_vgg16", "ms": True,  "ca": True,  "eh": True,
     "desc": "AW-DPCNN + MSCA-VGG16 (full)"},
]

OUTPUT_ROOT = Path("experiments/ablation_results")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_config(base_cfg: Dict, exp: Dict, num_classes: int,
                 seed: int, epochs: int) -> Dict:
    """Create a resolved config for one experiment."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = f"ablation_{exp['id']}"
    cfg["dataset"]["root_dir"] = exp["dataset"]
    cfg["dataset"]["num_classes"] = num_classes
    cfg["model"]["name"] = exp["model"]
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["pretrained"] = False
    cfg["model"]["use_ms"] = exp["ms"]
    cfg["model"]["use_ca"] = exp["ca"]
    cfg["model"]["use_eh"] = exp["eh"]
    cfg["train"]["epochs"] = epochs
    cfg["seed"] = seed
    cfg["output"]["root_dir"] = str(OUTPUT_ROOT / exp["id"])
    cfg.setdefault("visualization", {})
    cfg["visualization"]["tsne"] = True
    cfg["visualization"]["confusion_matrix"] = True
    cfg["visualization"]["roc_curve"] = False
    return cfg


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified B0–B8 ablation runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exp-ids", default="all",
                        help="Comma‑separated IDs, e.g. B0,B4,B8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Filter
    if args.exp_ids == "all":
        selected = EXPERIMENTS
    else:
        ids = set(x.strip() for x in args.exp_ids.split(","))
        selected = [e for e in EXPERIMENTS if e["id"] in ids]

    if not selected:
        print("[ERROR] No experiments selected.")
        sys.exit(1)

    print(f"Experiments: {len(selected)}")
    for e in selected:
        print(f"  {e['id']}: {e['desc']}  [{e['model']}]")

    if args.dry_run:
        print("\n[Dry run — no training]")
        return

    base_config = load_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and
                          torch.cuda.is_available()) else args.device)
    seed = args.seed
    epochs = args.epochs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_table: List[Dict] = []

    for idx, exp in enumerate(selected, 1):
        eid = exp["id"]
        print(f"\n{'─'*70}\n[{idx}/{len(selected)}]  {eid}: {exp['desc']}\n{'─'*70}")

        ds_path = Path(exp["dataset"])
        if not ds_path.is_dir():
            print(f"  [SKIP] Dataset not found: {ds_path}")
            results_table.append({"exp_id": eid, "desc": exp["desc"],
                                  "error": "dataset missing"})
            continue

        train_dir = ds_path / "train"
        class_names = sorted(d.name for d in train_dir.iterdir()
                             if d.is_dir() and not d.name.startswith("."))
        num_classes = len(class_names)
        if num_classes < 2:
            print(f"  [SKIP] {num_classes} class(es)")
            continue

        config = build_config(base_config, exp, num_classes, seed, epochs)
        run_dir = prepare_run_dir(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)

        try:
            results = train_and_evaluate(config, run_dir, device, seed=seed)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results_table.append({"exp_id": eid, "desc": exp["desc"],
                                  "acc": float("nan"), "error": str(exc)})
            continue

        dump_json(run_dir / "results" / "test_metrics.json", results)

        row = {
            "exp_id": eid, "desc": exp["desc"],
            "model": exp["model"],
            "awdpcnn": ("γ=10" if eid in ("B4","B5","B6","B7","B8")
                        else "γ=1" if eid == "B3"
                        else "none" if eid in ("B0","B1") else "concat"),
            "ms": "✓" if exp["ms"] else "✗",
            "ca": "✓" if exp["ca"] else "✗",
            "eh": "✓" if exp["eh"] else "✗",
            "acc": float(results.get("test_acc", float("nan"))),
            "f1": float(results.get("test_f1", float("nan"))),
            "gmean": float(results.get("test_gmean", float("nan"))),
            "kappa": float(results.get("test_kappa", float("nan"))),
            "auc": float(results.get("test_auc", float("nan"))),
            "error": "",
        }
        results_table.append(row)

        def _p(v): return f"{v*100:.2f}" if v == v else "N/A"
        print(f"  Acc={_p(row['acc'])}%  F1={_p(row['f1'])}%  "
              f"G-mean={_p(row['gmean'])}%  κ={_p(row['kappa'])}%")

    # ── Summary ──
    print(f"\n{'='*80}\n  ABLATION SUMMARY\n{'='*80}")
    if results_table:
        csv_path = OUTPUT_ROOT / f"ablation_summary_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["exp_id", "desc", "model", "awdpcnn", "ms", "ca", "eh",
                "acc", "f1", "gmean", "kappa", "auc"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results_table)
        print(f"CSV: {csv_path}")

        # Ranked table
        ranked = sorted(results_table, key=lambda r: r.get("acc", 0.0),
                        reverse=True)
        hdr = (f"{'ID':<4} {'AW-DPCNN':>10} {'MS':>4} {'CA':>4} {'EH':>4}  "
               f"{'Acc %':>7} {'F1 %':>7} {'G-mean %':>9} {'κ %':>7}")
        print(hdr)
        print("─" * len(hdr))
        for r in ranked:
            def _f(v): return f"{v*100:6.2f}" if isinstance(v, float) and v == v else "   N/A"
            print(f"{r['exp_id']:<4} {r['awdpcnn']:>10} {r['ms']:>4} "
                  f"{r['ca']:>4} {r['eh']:>4}  "
                  f"{_f(r.get('acc',float('nan')))}  "
                  f"{_f(r.get('f1',float('nan')))}  "
                  f"{_f(r.get('gmean',float('nan')))}  "
                  f"{_f(r.get('kappa',float('nan')))}")

    print(f"\n{'='*80}\n  Done.\n{'='*80}")


if __name__ == "__main__":
    main()
