#!/usr/bin/env python3
"""
Unified Ablation Experiment Runner — B0–B8
===========================================
Runs the complete 9‑experiment ablation study on the CWRU 12k DE dataset.

Design
------
  **Fusion‑level ablation** (B0–B4):  standard VGG16 classifier.
      B0 — STFT‑only                  (single representation)
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

    # Single trial (default)
    python scripts/run_ablation_experiments.py --epochs 30

    # Dry‑run
    python scripts/run_ablation_experiments.py --dry-run

    # Single experiment
    python scripts/run_ablation_experiments.py --exp-ids B0,B8

    # Specific trial seed
    python scripts/run_ablation_experiments.py --trial trial_seed456

    # Three independent trials with aggregation (mean ± std)
    !!!!!!!! time-consuming, about 2 hours
    python scripts/run_ablation_experiments.py --num-trials 3

    # Custom trial seeds
    python scripts/run_ablation_experiments.py --trial-seeds 42,123,456,789

    # Continue on error
    python scripts/run_ablation_experiments.py --num-trials 3 --continue-on-error
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
    {"id": "B0", "dataset": "datasets/ablation/stft_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "STFT-only"},
    {"id": "B1", "dataset": "datasets/ablation/gadf_only",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "GADF-only"},
    {"id": "B2", "dataset": "datasets/ablation/concat",
     "model": "vgg16", "ms": False, "ca": False, "eh": False,
     "desc": "STFT+GADF concat (pixel avg)"},
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

OUTPUT_ROOT = Path("experiments/experiment_result/ablation")
DEFAULT_TRIAL = "trial_seed42"
DEFAULT_TRIAL_SEEDS = [42, 123, 456]


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

def _parse_seed_from_trial(trial: str) -> int:
    """Extract seed integer from trial name, e.g. 'trial_seed42' → 42."""
    import re
    m = re.search(r"seed(\d+)", trial)
    if m:
        return int(m.group(1))
    # Fallback: hash the trial string to get a deterministic seed
    return abs(hash(trial)) % (2**31)


def main():
    parser = argparse.ArgumentParser(description="Unified B0–B8 ablation runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for single-trial mode (auto-derived from --trial)")
    parser.add_argument("--trial", default=DEFAULT_TRIAL,
                        help=f"Trial name for single-trial mode (default: {DEFAULT_TRIAL})")
    parser.add_argument("--num-trials", type=int, default=1,
                        help=f"Number of independent trials (default: 1; uses first N of {DEFAULT_TRIAL_SEEDS})")
    parser.add_argument("--trial-seeds", default=None,
                        help="Comma-separated seeds, e.g. 42,123,456 (overrides --num-trials)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exp-ids", default="all",
                        help="Comma‑separated IDs, e.g. B0,B4,B8")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue remaining trials after a failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Resolve trial seeds ──
    if args.trial_seeds:
        seeds = [int(s.strip()) for s in args.trial_seeds.split(",")]
    elif args.num_trials > 1:
        seeds = DEFAULT_TRIAL_SEEDS[:args.num_trials]
    else:
        # Single-trial mode: use --seed or derive from --trial
        single_seed = args.seed if args.seed is not None else _parse_seed_from_trial(args.trial)
        seeds = [single_seed]

    multi_trial = len(seeds) > 1

    # ── Filter experiments ──
    if args.exp_ids == "all":
        selected = EXPERIMENTS
    else:
        ids = set(x.strip() for x in args.exp_ids.split(","))
        selected = [e for e in EXPERIMENTS if e["id"] in ids]

    if not selected:
        print("[ERROR] No experiments selected.")
        sys.exit(1)

    print(f"Trials:   {len(seeds)}  seeds={seeds}")
    print(f"Experiments: {len(selected)}")
    for e in selected:
        print(f"  {e['id']}: {e['desc']}  [{e['model']}]")

    if args.dry_run:
        print("\n[Dry run — no training]")
        return

    base_config = load_yaml(args.config)
    device = torch.device("cuda" if (args.device == "auto" and
                          torch.cuda.is_available()) else args.device)
    epochs = args.epochs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_table: List[Dict] = []

    # ── Run experiments ──
    for idx, exp in enumerate(selected, 1):
        eid = exp["id"]
        print(f"\n{'═'*70}\n  [{idx}/{len(selected)}]  {eid}: {exp['desc']}\n{'═'*70}")

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

        # ── Run each trial ──
        trial_dirs: List[Path] = []
        trial_metrics: List[Dict] = []
        for trial_idx, seed in enumerate(seeds, 1):
            label = f"Trial {trial_idx}/{len(seeds)}" if multi_trial else "Trial"
            print(f"\n  [{label}] seed={seed}")

            config = build_config(base_config, exp, num_classes, seed, epochs)
            run_dir = prepare_run_dir(config)
            run_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)

            try:
                results = train_and_evaluate(config, run_dir, device, seed=seed)
            except Exception as exc:
                print(f"  [FAIL] {exc}")
                if not args.continue_on_error:
                    print(f"  [STOP] Use --continue-on-error to skip failures.")
                    break
                continue

            dump_json(run_dir / "results" / "test_metrics.json", results)
            trial_dirs.append(run_dir)
            trial_metrics.append(results)

            def _p(v): return f"{v*100:.2f}" if v == v else "N/A"
            print(f"    Acc={_p(results.get('test_acc', float('nan')))}%  "
                  f"F1={_p(results.get('test_f1', float('nan')))}%  "
                  f"G-mean={_p(results.get('test_gmean', float('nan')))}%")

        if not trial_dirs:
            results_table.append({"exp_id": eid, "desc": exp["desc"],
                                  "acc": float("nan"), "error": "all trials failed"})
            continue

        # ── Aggregate across trials ──
        if len(trial_dirs) >= 2:
            from src.utils.aggregation import aggregate_and_save
            agg_dir = trial_dirs[0].parent / "aggregated"
            title = f"Ablation {eid} — {exp['desc']} ({len(trial_dirs)} trials)"
            aggregated = aggregate_and_save(trial_dirs, agg_dir, title=title)
            print(f"  Aggregated → {agg_dir}")
        else:
            aggregated = None

        # ── Build summary row ──
        def _agg(key, fallback=float("nan")):
            if aggregated and key in aggregated:
                return aggregated[key]["mean"]
            if trial_metrics:
                return float(trial_metrics[0].get(key, fallback))
            return fallback

        def _agg_std(key, fallback=float("nan")):
            if aggregated and key in aggregated:
                return aggregated[key]["std"]
            return fallback

        awdpcnn_label = ("γ=10" if eid in ("B4","B5","B6","B7","B8")
                         else "γ=1" if eid == "B3"
                         else "none" if eid in ("B0","B1") else "concat")

        row = {
            "exp_id": eid, "desc": exp["desc"],
            "model": exp["model"],
            "awdpcnn": awdpcnn_label,
            "ms": "✓" if exp["ms"] else "✗",
            "ca": "✓" if exp["ca"] else "✗",
            "eh": "✓" if exp["eh"] else "✗",
            "trials": len(trial_dirs),
            "acc": _agg("test_acc"),
            "acc_std": _agg_std("test_acc"),
            "f1": _agg("test_f1"),
            "f1_std": _agg_std("test_f1"),
            "gmean": _agg("test_gmean"),
            "gmean_std": _agg_std("test_gmean"),
            "kappa": _agg("test_kappa"),
            "kappa_std": _agg_std("test_kappa"),
            "auc": _agg("test_auc"),
            "auc_std": _agg_std("test_auc"),
            "error": "",
        }
        results_table.append(row)

    # ═══════════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}\n  ABLATION SUMMARY  ({len(seeds)} trial(s) each)\n{'='*80}")
    if not results_table:
        print("  No results.")
        print(f"\n{'='*80}\n  Done.\n{'='*80}")
        return

    # ── CSV ──
    csv_path = OUTPUT_ROOT / f"ablation_summary_{ts}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["exp_id", "desc", "model", "awdpcnn", "ms", "ca", "eh", "trials",
            "acc", "acc_std", "f1", "f1_std", "gmean", "gmean_std",
            "kappa", "kappa_std", "auc", "auc_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results_table)
    print(f"CSV: {csv_path}")

    # ── Markdown table ──
    md_path = OUTPUT_ROOT / f"ablation_summary_{ts}.md"
    md_lines = [
        "# Ablation Study Results",
        "",
        f"**Trials**: {len(seeds)} independent runs per experiment.",
        f"**Seeds**: {seeds}",
        "",
        "| ID | Description | AW-DPCNN | MS | CA | EH | Acc (%) | F1 (%) | G-Mean (%) | κ (%) |",
        "|----|-------------|----------|----|----|----|---------|--------|------------|-------|",
    ]

    ranked = sorted(results_table,
                    key=lambda r: (r.get("acc", float("nan"))
                                   if r.get("acc", float("nan")) == r.get("acc", float("nan"))
                                   else float("-inf")),
                    reverse=True)

    # ── Console ──
    if multi_trial:
        hdr = (f"{'ID':<4} {'AW-DPCNN':>10} {'MS':>4} {'CA':>4} {'EH':>4}  "
               f"{'Acc %':>14} {'F1 %':>14} {'G-mean %':>14} {'κ %':>14}")
    else:
        hdr = (f"{'ID':<4} {'AW-DPCNN':>10} {'MS':>4} {'CA':>4} {'EH':>4}  "
               f"{'Acc %':>7} {'F1 %':>7} {'G-mean %':>9} {'κ %':>7}")
    print(hdr)
    print("─" * len(hdr))

    for r in ranked:
        acc = r.get("acc", float("nan"))
        f1 = r.get("f1", float("nan"))
        gm = r.get("gmean", float("nan"))
        ka = r.get("kappa", float("nan"))

        if multi_trial:
            acc_s = r.get("acc_std", float("nan"))
            f1_s = r.get("f1_std", float("nan"))
            gm_s = r.get("gmean_std", float("nan"))
            ka_s = r.get("kappa_std", float("nan"))
            def _fs(m, s):
                if isinstance(m, float) and m == m:
                    return f"{m*100:5.2f}±{s*100:.2f}"
                return "        N/A"
            print(f"{r['exp_id']:<4} {r['awdpcnn']:>10} {r['ms']:>4} "
                  f"{r['ca']:>4} {r['eh']:>4}  "
                  f"{_fs(acc, acc_s):>14}  {_fs(f1, f1_s):>14}  "
                  f"{_fs(gm, gm_s):>14}  {_fs(ka, ka_s):>14}")
            # Markdown row
            md_lines.append(
                f"| {r['exp_id']} | {r['desc']} | {r['awdpcnn']} | {r['ms']} | {r['ca']} | {r['eh']} | "
                f"{_fs(acc, acc_s)} | {_fs(f1, f1_s)} | {_fs(gm, gm_s)} | {_fs(ka, ka_s)} |")
        else:
            def _f(v):
                return f"{v*100:6.2f}" if isinstance(v, float) and v == v else "   N/A"
            print(f"{r['exp_id']:<4} {r['awdpcnn']:>10} {r['ms']:>4} "
                  f"{r['ca']:>4} {r['eh']:>4}  "
                  f"{_f(acc)}  {_f(f1)}  {_f(gm)}  {_f(ka)}")
            md_lines.append(
                f"| {r['exp_id']} | {r['desc']} | {r['awdpcnn']} | {r['ms']} | {r['ca']} | {r['eh']} | "
                f"{_f(acc)} | {_f(f1)} | {_f(gm)} | {_f(ka)} |")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nMD:  {md_path}")

    print(f"\n{'='*80}\n  Done.\n{'='*80}")


if __name__ == "__main__":
    main()
