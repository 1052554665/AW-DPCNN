#!/usr/bin/env python3
"""
Unified Experiment Result Analyzer — LaTeX Table Generator
============================================================
Aggregates results from all experiment types across multiple trial seeds
and produces publication-ready LaTeX tables matching the paper format.

Supported analyses
------------------
  ablation        B0–B8 unified ablation (fusion + classifier components)
  backbone        Model performance comparison (exp1, multiple backbones)
  rep_compare     Representation comparison (12 TF × temporal combos)
  hyperparam      Hyperparameter sensitivity (γ, N, α)
  noise           Noise robustness (SNR sweep)
  all             All of the above

Usage::

    # Single trial (seed 42)
    python scripts/analyze_results.py --trial-seeds 42 --output-dir paper/auto_tables

    # Three independent trials with mean ± std
    python scripts/analyze_results.py --trial-seeds 42,123,456

    # Only ablation and backbone tables
    python scripts/analyze_results.py --analyses ablation,backbone --trial-seeds 42,123,456

    # Dry-run (print LaTeX to stdout)
    python scripts/analyze_results.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.dataset_registry import DATASET_KEYS  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
#  Paths & Constants
# ═══════════════════════════════════════════════════════════════════════

ABLATION_ROOT   = Path("experiments/experiment_result/ablation_results")
EXP1_ROOT       = Path("experiments/experiment_result/exp1")
REP_COMPARE_ROOT = Path("experiments/experiment_result/rep_compare")
HYPERPARAM_FILE  = Path("experiments/experiment_result/hyperparameter_sensitivity/sensitivity_summary.json")
NOISE_FILE       = Path("experiments/experiment_result/noise_robustness/noise_robustness.json")

# ── Ablation definitions (matches run_ablation_experiments.py) ──
ABLATION_EXPS = [
    {"id": "B0", "desc": "Mel-only",                   "aw": "$\\times$", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B1", "desc": "GADF-only",                  "aw": "$\\times$", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B2", "desc": "Mel+GADF concat (pixel avg)","aw": "$\\times$", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B3", "desc": "AW-DPCNN $\\gamma$=1 (equal weight)","aw": "$\\checkmark$ ($\\gamma$=1)", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B4", "desc": "AW-DPCNN $\\gamma$=10 — fusion baseline","aw": "$\\checkmark$ ($\\gamma$=10)", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B5", "desc": "AW-DPCNN + MS only",          "aw": "$\\checkmark$", "ms": "$\\checkmark$", "ca": "$\\times$", "eh": "$\\times$"},
    {"id": "B6", "desc": "AW-DPCNN + CA only",          "aw": "$\\checkmark$", "ms": "$\\times$", "ca": "$\\checkmark$", "eh": "$\\times$"},
    {"id": "B7", "desc": "AW-DPCNN + EH only",          "aw": "$\\checkmark$", "ms": "$\\times$", "ca": "$\\times$", "eh": "$\\checkmark$"},
    {"id": "B8", "desc": "AW-DPCNN + MSCA-VGG16 (full)","aw": "$\\checkmark$", "ms": "$\\checkmark$", "ca": "$\\checkmark$", "eh": "$\\checkmark$"},
]

# ── Exp1 model display names ──
EXP1_MODELS = [
    ("MSCA_VGG16",      "MSCA-VGG16 (Ours)"),
    ("resnet18",        "ResNet18"),
    ("vgg16",           "VGG16"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("vit",             "ViT"),
    ("mobilenetv3_small","MobileNetV3-Small"),
    ("convnext_tiny",   "ConvNeXt-Tiny"),
]

# ── Rep compare combos ──
REP_TF_METHODS = ["Mel", "STFT", "CWT"]
REP_TEMPORAL   = ["GADF", "GASF", "MTF", "RP"]

# ── Metric keys ──
_METRIC_KEYS = ["test_acc", "test_f1", "test_gmean", "test_kappa",
                "test_precision", "test_recall", "test_auc"]
_PERCENT_KEYS = {"test_acc", "test_precision", "test_recall",
                 "test_f1", "test_gmean", "test_auc",
                 "test_kappa", "test_bal_acc"}


# ═══════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_trial_results(trial_dirs: List[Path]) -> List[Dict]:
    """Load test_metrics.json from each trial directory."""
    results = []
    for d in trial_dirs:
        p = d / "results" / "test_metrics.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
    return results


def _aggregate(trial_dirs: List[Path]) -> Dict:
    """Aggregate across trial directories. Returns {key: {mean, std}} or single value."""
    from src.utils.aggregation import aggregate_trials
    return aggregate_trials(trial_dirs)


def _fmt_mean_std(mean: float, std: float, key: str,
                  decimal_places: int = 2) -> str:
    """Format as 'mean \\pm std' with percent scaling — LaTeX-safe (no Unicode)."""
    from src.utils.aggregation import format_mean_std
    s = format_mean_std(mean, std, key, decimal_places=decimal_places, as_percent=True)
    return s.replace(" ± ", " \\pm ")


def _fmt_val(v: float, key: str, decimal_places: int = 2) -> str:
    """Format a single value with percent scaling."""
    scale = 100.0 if key in _PERCENT_KEYS else 1.0
    return f"{v * scale:.{decimal_places}f}"


def _get_trial_dirs(base_dir: Path, trial_seeds: List[int]) -> List[Path]:
    """Find existing trial_seed* dirs under base_dir."""
    dirs = []
    for seed in trial_seeds:
        d = base_dir / f"trial_seed{seed}"
        metrics = d / "results" / "test_metrics.json"
        if metrics.exists():
            dirs.append(d)
    return dirs


def _bold_if_best(value_str: str, is_best: bool) -> str:
    """Wrap in \\textbf{} if this is the best value."""
    if is_best:
        return f"\\textbf{{{value_str}}}"
    return value_str


# ═══════════════════════════════════════════════════════════════════════
#  LaTeX Helpers
# ═══════════════════════════════════════════════════════════════════════

def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _write_latex_preamble() -> str:
    return "\n".join([
        "% Auto-generated by scripts/analyze_results.py",
        f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "% These tables can be \\input{} directly into tim.tex",
        "",
    ])


# ═══════════════════════════════════════════════════════════════════════
#  1. Ablation Table
# ═══════════════════════════════════════════════════════════════════════

def generate_ablation_table(trial_seeds: List[int]) -> str:
    """Generate LaTeX for tab:ablation_unified."""
    multi = len(trial_seeds) > 1
    # Determine actual max trials available across all experiments
    max_trials = 0
    for exp in ABLATION_EXPS:
        base_dir = ABLATION_ROOT / exp["id"] / f"ablation_{exp['id']}"
        n = len(_get_trial_dirs(base_dir, trial_seeds))
        if n > max_trials:
            max_trials = n
    actual_multi = max_trials >= 2

    lines = []
    lines.append("% ── Ablation Table (B0–B8) ──")
    lines.append("\\begin{table*}")
    lines.append("    \\centering")
    if actual_multi:
        n_trials_str = f"Mean $\\pm$ Std over {max_trials} Independent Trials"
    else:
        n_trials_str = f"Seed {trial_seeds[0]}"
    lines.append(f"    \\caption{{Unified Component Decomposition Ablation (B0–B8, {n_trials_str}). "
                 "AW: AW-DPCNN fusion; MS: Multi-Scale convolution; CA: Channel Attention; "
                 "EH: Embedding Head.}")
    lines.append("    \\label{tab:ablation_unified}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    cols = "lcccc" + "c" * 4  # Exp, AW, MS, CA, EH + Acc, F1, G-mean, Kappa
    lines.append(f"    \\begin{{tabular}}{{{cols}}}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Exp.} & \\textbf{AW-DPCNN} & \\textbf{MS} & "
                 "\\textbf{CA} & \\textbf{EH} & \\textbf{Acc (\\%)} & "
                 "\\textbf{F1 (\\%)} & \\textbf{G-mean (\\%)} & \\textbf{Kappa (\\%)} \\\\")
    lines.append("        \\midrule")

    # Fusion-level section
    lines.append("        \\multicolumn{9}{c}{\\textit{Fusion-level ablation (VGG16 classifier)}} \\\\")
    for exp in ABLATION_EXPS[:5]:  # B0-B4
        row = _build_ablation_row(exp, trial_seeds, multi)
        lines.append(f"        {row} \\\\")

    lines.append("        \\midrule")

    # Classifier-level section
    lines.append("        \\multicolumn{9}{c}{\\textit{Classifier-level ablation (full AW-DPCNN fusion)}} \\\\")
    for exp in ABLATION_EXPS[5:]:  # B5-B8
        row = _build_ablation_row(exp, trial_seeds, multi)
        lines.append(f"        {row} \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _build_ablation_row(exp: Dict, trial_seeds: List[int], multi: bool) -> str:
    """Build one row of the ablation table."""
    eid = exp["id"]
    base_dir = ABLATION_ROOT / eid / f"ablation_{eid}"
    trial_dirs = _get_trial_dirs(base_dir, trial_seeds)

    if not trial_dirs:
        na = "xx.xx" if not multi else "xx.xx ± x.xx"
        return (f"{eid} & {exp['aw']} & {exp['ms']} & {exp['ca']} & "
                f"{exp['eh']} & ${na}$ & ${na}$ & ${na}$ & ${na}$")

    if len(trial_dirs) >= 2:
        agg = _aggregate(trial_dirs)
        acc_s  = _fmt_mean_std(agg["test_acc"]["mean"], agg["test_acc"]["std"], "test_acc")
        f1_s   = _fmt_mean_std(agg["test_f1"]["mean"], agg["test_f1"]["std"], "test_f1")
        gm_s   = _fmt_mean_std(agg["test_gmean"]["mean"], agg["test_gmean"]["std"], "test_gmean")
        ka_s   = _fmt_mean_std(agg["test_kappa"]["mean"], agg["test_kappa"]["std"], "test_kappa")
    else:
        m = _load_trial_results(trial_dirs)[0]
        acc_s = _fmt_val(m.get("test_acc", float("nan")), "test_acc")
        f1_s  = _fmt_val(m.get("test_f1", float("nan")), "test_f1")
        gm_s  = _fmt_val(m.get("test_gmean", float("nan")), "test_gmean")
        ka_s  = _fmt_val(m.get("test_kappa", float("nan")), "test_kappa")

    # Bold B8 (full model)
    if eid == "B8":
        acc_s = f"\\mathbf{{{acc_s}}}"
        f1_s  = f"\\mathbf{{{f1_s}}}"
        gm_s  = f"\\mathbf{{{gm_s}}}"
        ka_s  = f"\\mathbf{{{ka_s}}}"
        return (f"\\textbf{{B8}} & {exp['aw']} & {exp['ms']} & "
                f"{exp['ca']} & {exp['eh']} & ${acc_s}$ & ${f1_s}$ & ${gm_s}$ & ${ka_s}$")

    return (f"{eid} & {exp['aw']} & {exp['ms']} & {exp['ca']} & "
            f"{exp['eh']} & ${acc_s}$ & ${f1_s}$ & ${gm_s}$ & ${ka_s}$")


# ═══════════════════════════════════════════════════════════════════════
#  2. Backbone Comparison Table (network_comparison)
# ═══════════════════════════════════════════════════════════════════════

def generate_backbone_table(trial_seeds: List[int],
                            dataset_key: str = "") -> str:
    """Generate LaTeX for tab:network_comparison.

    When ``dataset_key`` is provided (e.g. ``"12k_de"``), results are
    read from ``EXP1_ROOT / model / dataset_key / exp1_* / trial_seed*``.
    """
    multi = len(trial_seeds) > 1

    def _find_base_dir(exp_dir: Path) -> Optional[Path]:
        """Locate the innermost experiment directory (e.g. exp1_MSCA_VGG16)."""
        if dataset_key:
            # New structure: model/dataset/exp_name/trial_seed*
            pattern = f"{dataset_key}/exp1_*"
        else:
            # Old structure (backward compat): model/exp_name/trial_seed*
            # Also try the new structure without a dataset filter
            pattern = "exp1_*"
        candidates = sorted(exp_dir.glob(pattern))
        if not candidates and not dataset_key:
            # Fallback: look one level deeper
            candidates = sorted(exp_dir.glob("*/exp1_*"))
        return candidates[0] if candidates else None

    # Determine actual max trials available
    max_trials = 0
    for dir_name, _ in EXP1_MODELS:
        exp_dir = EXP1_ROOT / dir_name
        base_dir = _find_base_dir(exp_dir)
        if base_dir:
            n = len(_get_trial_dirs(base_dir, trial_seeds))
            if n > max_trials:
                max_trials = n
    actual_multi = max_trials >= 2

    lines = []
    lines.append("% ── Backbone Comparison Table ──")
    lines.append("\\begin{table*}")
    lines.append("    \\centering")
    if actual_multi:
        n_trials_str = f"Mean $\\pm$ Std over {max_trials} Independent Trials"
    else:
        n_trials_str = f"Seed {trial_seeds[0]}"
    lines.append(f"    \\caption{{Performance Comparison with Different Backbone Networks ({n_trials_str})}}")
    lines.append("    \\label{tab:network_comparison}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{lcccccc}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{Network} & \\textbf{Acc (\\%)} & \\textbf{Recall (\\%)} "
                 "& \\textbf{F1 (\\%)} & \\textbf{G-mean (\\%)} & "
                 "\\textbf{B-Acc (\\%)} & \\textbf{Kappa (\\%)} \\\\")
    lines.append("        \\midrule")

    rows_data = []
    raw_means = []
    for dir_name, display_name in EXP1_MODELS:
        exp_dir = EXP1_ROOT / dir_name
        base_dir = _find_base_dir(exp_dir)
        if base_dir is None:
            rows_data.append((display_name, None))
            raw_means.append((display_name, {}))
            continue
        trial_dirs = _get_trial_dirs(base_dir, trial_seeds)
        if not trial_dirs:
            rows_data.append((display_name, None))
            raw_means.append((display_name, {}))
            continue

        if len(trial_dirs) >= 2 and multi:
            agg = _aggregate(trial_dirs)
            metrics = {
                "acc": _fmt_mean_std(agg["test_acc"]["mean"], agg["test_acc"]["std"], "test_acc"),
                "rec": _fmt_mean_std(agg["test_recall"]["mean"], agg["test_recall"]["std"], "test_recall"),
                "f1":  _fmt_mean_std(agg["test_f1"]["mean"], agg["test_f1"]["std"], "test_f1"),
                "gm":  _fmt_mean_std(agg["test_gmean"]["mean"], agg["test_gmean"]["std"], "test_gmean"),
                "ba":  _fmt_mean_std(agg["test_bal_acc"]["mean"], agg["test_bal_acc"]["std"], "test_bal_acc"),
                "ka":  _fmt_mean_std(agg["test_kappa"]["mean"], agg["test_kappa"]["std"], "test_kappa"),
            }
            raw = {
                "acc": agg["test_acc"]["mean"],
                "rec": agg["test_recall"]["mean"],
                "f1":  agg["test_f1"]["mean"],
                "gm":  agg["test_gmean"]["mean"],
                "ba":  agg["test_bal_acc"]["mean"],
                "ka":  agg["test_kappa"]["mean"],
            }
        else:
            m = _load_trial_results(trial_dirs)[0]
            metrics = {
                "acc": _fmt_val(m.get("test_acc", float("nan")), "test_acc"),
                "rec": _fmt_val(m.get("test_recall", float("nan")), "test_recall"),
                "f1":  _fmt_val(m.get("test_f1", float("nan")), "test_f1"),
                "gm":  _fmt_val(m.get("test_gmean", float("nan")), "test_gmean"),
                "ba":  _fmt_val(m.get("test_bal_acc", float("nan")), "test_bal_acc"),
                "ka":  _fmt_val(m.get("test_kappa", float("nan")), "test_kappa"),
            }
            raw = {
                "acc": m.get("test_acc", float("nan")),
                "rec": m.get("test_recall", float("nan")),
                "f1":  m.get("test_f1", float("nan")),
                "gm":  m.get("test_gmean", float("nan")),
                "ba":  m.get("test_bal_acc", float("nan")),
                "ka":  m.get("test_kappa", float("nan")),
            }
        rows_data.append((display_name, metrics))
        raw_means.append((display_name, raw))

    # Determine best model per metric column
    metric_keys = ["acc", "rec", "f1", "gm", "ba", "ka"]
    best_model = {}
    for key in metric_keys:
        valid = [(name, rm[key]) for name, rm in raw_means
                 if key in rm and rm[key] == rm[key] and rm[key] != float("-inf")]
        if valid:
            best_model[key] = max(valid, key=lambda x: x[1])[0]

    for display_name, metrics in rows_data:
        if metrics is None:
            na = "xx.xx ± x.xx" if multi else "xx.xx"
            row = f"{display_name} & {na} & {na} & {na} & {na} & {na} & {na}"
        else:
            is_proposed = display_name.startswith("MSCA-VGG16")
            name_cell = f"\\textbf{{{display_name}}}" if is_proposed else display_name
            vals = []
            for key in metric_keys:
                v = metrics[key]
                if best_model.get(key) == display_name:
                    vals.append(f"$\\mathbf{{{v}}}$")
                else:
                    vals.append(f"${v}$")
            row = f"{name_cell} & " + " & ".join(vals)
        lines.append(f"        {row} \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  3. Representation Comparison Table
# ═══════════════════════════════════════════════════════════════════════

def generate_rep_compare_table(trial_seeds: List[int]) -> str:
    """Generate LaTeX for tab:rep_compare."""
    multi = len(trial_seeds) > 1
    lines = []
    lines.append("% ── Representation Comparison Table ──")
    lines.append("\\begin{table*}")
    lines.append("    \\centering")
    lines.append("    \\caption{Representation Comparison Across Time--Frequency and "
                 "Temporal Encoding Methods. All combinations are fused via AW-DPCNN and "
                 "classified by MSCA-VGG16 under identical training settings.}")
    lines.append("    \\label{tab:rep_compare}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{llcccccc}")
    lines.append("        \\toprule")
    lines.append("        \\textbf{TF Method} & \\textbf{Temporal} & \\textbf{Acc (\\%)} "
                 "& \\textbf{Prec. (\\%)} & \\textbf{Rec. (\\%)} & \\textbf{F1 (\\%)} "
                 "& \\textbf{G-mean (\\%)} & \\textbf{Kappa (\\%)} \\\\")
    lines.append("        \\midrule")

    for i, tf in enumerate(REP_TF_METHODS):
        for j, temporal in enumerate(REP_TEMPORAL):
            combo = f"{tf.lower()}_{temporal.lower()}"
            base_dir = REP_COMPARE_ROOT / combo / f"rep_compare_{combo}"
            trial_dirs = _get_trial_dirs(base_dir, trial_seeds)

            if not trial_dirs:
                na = "xx.xx \\pm x.xx" if multi else "xx.xx"
                vals = [f"${na}$"] * 6
            elif len(trial_dirs) >= 2 and multi:
                agg = _aggregate(trial_dirs)
                vals = [
                    _fmt_mean_std(agg["test_acc"]["mean"], agg["test_acc"]["std"], "test_acc"),
                    _fmt_mean_std(agg["test_precision"]["mean"], agg["test_precision"]["std"], "test_precision"),
                    _fmt_mean_std(agg["test_recall"]["mean"], agg["test_recall"]["std"], "test_recall"),
                    _fmt_mean_std(agg["test_f1"]["mean"], agg["test_f1"]["std"], "test_f1"),
                    _fmt_mean_std(agg["test_gmean"]["mean"], agg["test_gmean"]["std"], "test_gmean"),
                    _fmt_mean_std(agg["test_kappa"]["mean"], agg["test_kappa"]["std"], "test_kappa"),
                ]
            else:
                m = _load_trial_results(trial_dirs)[0]
                vals = [
                    _fmt_val(m.get("test_acc", float("nan")), "test_acc"),
                    _fmt_val(m.get("test_precision", float("nan")), "test_precision"),
                    _fmt_val(m.get("test_recall", float("nan")), "test_recall"),
                    _fmt_val(m.get("test_f1", float("nan")), "test_f1"),
                    _fmt_val(m.get("test_gmean", float("nan")), "test_gmean"),
                    _fmt_val(m.get("test_kappa", float("nan")), "test_kappa"),
                ]

            if j == 0:
                tf_cell = f"\\multirow{{4}}{{*}}{{{tf}}}"
            else:
                tf_cell = ""

            row = f"        {tf_cell} & {temporal} & " + " & ".join(f"${v}$" for v in vals)
            lines.append(f"{row} \\\\")

        # Add cmidrule between TF methods (except after the last one)
        if i < len(REP_TF_METHODS) - 1:
            lines.append("        \\cmidrule(lr){2-8}")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  4. Hyperparameter Sensitivity Table
# ═══════════════════════════════════════════════════════════════════════

def generate_hyperparam_table() -> str:
    """Generate LaTeX for tab:hyperparam_sensitivity."""
    data = _load_json(HYPERPARAM_FILE)
    if not data:
        return "% [WARN] Hyperparameter sensitivity data not found.\n"

    lines = []
    lines.append("% ── Hyperparameter Sensitivity Table ──")
    lines.append("\\begin{table}")
    lines.append("    \\centering")
    # Check if recall data exists in any parameter group
    has_recall = any("recall" in data.get(k, {}) and len(data[k].get("recall", [])) > 0
                     for k in data)

    lines.append("    \\caption{Hyperparameter Sensitivity Analysis. "
                 + ("Accuracy and Recall are" if has_recall else "Accuracy is")
                 + " evaluated on re-fused test windows using a frozen MSCA-VGG16 classifier. "
                 "Default values are highlighted in bold.}")
    lines.append("    \\label{tab:hyperparam_sensitivity}")
    lines.append("    \\renewcommand{\\arraystretch}{1.15}")
    lines.append("    \\small")
    if has_recall:
        lines.append("    \\begin{tabular}{cc|cc}")
    else:
        lines.append("    \\begin{tabular}{cc|c}")
    lines.append("        \\toprule")
    if has_recall:
        lines.append("        \\textbf{Parameter} & \\textbf{Value} & \\textbf{Acc (\\%)} & \\textbf{Rec (\\%)} \\\\")
    else:
        lines.append("        \\textbf{Parameter} & \\textbf{Value} & \\textbf{Acc (\\%)} \\\\")
    lines.append("        \\midrule")

    def _param_rows(param_label, param_data, default_val):
        vals = param_data.get("param", [])
        accs = param_data.get("accuracy", [])
        recs = param_data.get("recall", [])
        has_rec = len(recs) == len(vals)
        n = len(vals)
        lines.append(f"        \\multirow{{{n}}}{{*}}{{{param_label}}} ")
        for i in range(n):
            v = vals[i]
            a = accs[i] if i < len(accs) else 0.0
            r = recs[i] if has_rec else 0.0
            is_default = (v == default_val or str(v) == str(default_val))
            if is_default:
                if has_rec:
                    lines.append(f"        & \\textbf{{{v}}} & \\textbf{{{a:.1f}}} & \\textbf{{{r:.1f}}} \\\\")
                else:
                    lines.append(f"        & \\textbf{{{v}}} & \\textbf{{{a:.1f}}} \\\\")
            else:
                if has_rec:
                    lines.append(f"        & {v} & {a:.1f} & {r:.1f} \\\\")
                else:
                    lines.append(f"        & {v} & {a:.1f} \\\\")

    # Gamma
    _param_rows("$\\gamma$ (contrast amplification)", data.get("gamma", {}), 10)
    lines.append("        \\midrule")

    # N
    _param_rows("$N$ (PCNN iterations)", data.get("N", {}), 20)
    lines.append("        \\midrule")

    # Alpha
    _param_rows("$\\alpha_L=\\alpha_T$ (decay)", data.get("alpha_LT", {}), "0.001")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  5. Noise Robustness Table
# ═══════════════════════════════════════════════════════════════════════

def generate_noise_table() -> str:
    """Generate LaTeX for tab:noise_robustness."""
    data = _load_json(NOISE_FILE)
    if not data:
        return "% [WARN] Noise robustness data not found.\n"

    # Determine SNR columns from first model
    first_model = next(iter(data.values()))
    snr_labels = first_model.get("snr", [])

    lines = []
    lines.append("% ── Noise Robustness Table ──")
    lines.append("\\begin{table*}[!htbp]")
    lines.append("    \\centering")
    lines.append("    \\caption{Noise Robustness Comparison --- Accuracy (\\%) at Different SNR Levels.}")
    lines.append("    \\label{tab:noise_robustness}")
    lines.append("    \\renewcommand{\\arraystretch}{1.1}")
    lines.append("    \\small")

    col_spec = "l" + "c" * len(snr_labels)
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")
    header = " & ".join([f"\\textbf{{{s}}}" for s in ["Model"] + snr_labels])
    lines.append(f"        {header} \\\\")
    lines.append("        \\midrule")

    # Model ordering: MSCA-VGG16 first, then others
    model_order = ["msca-vgg16", "vgg16", "vit", "efficientnet-b0",
                   "mobilenetv3_small", "convnext-tiny"]
    model_display = {
        "msca-vgg16": "\\textbf{MSCA-VGG16}",
        "vgg16": "VGG16",
        "vit": "ViT",
        "efficientnet-b0": "EfficientNet-B0",
        "mobilenetv3_small": "MobileNetV3-Small",
        "convnext-tiny": "ConvNeXt-Tiny",
    }

    for model_key in model_order:
        if model_key not in data:
            continue
        model_data = data[model_key]
        accs = model_data.get("accuracy", [])
        display = model_display.get(model_key, model_key)
        vals = " & ".join(f"{a:.1f}" for a in accs)
        lines.append(f"        {display} & {vals} \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Result Analyzer — LaTeX Table Generator")
    parser.add_argument("--trial-seeds", default="42",
                        help="Comma-separated trial seeds, e.g. 42,123,456 (default: 42)")
    parser.add_argument("--analyses", default="all",
                        help="Comma-separated: ablation,backbone,rep_compare,hyperparam,noise,all")
    parser.add_argument("--output-dir", default="paper/auto_tables",
                        help="Output directory for .tex files (default: paper/auto_tables)")
    parser.add_argument("--dataset", default="", choices=[""] + DATASET_KEYS,
                        help=f"Dataset key to scope backbone results "
                             f"{{{','.join(DATASET_KEYS)}}}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print LaTeX to stdout instead of writing files")
    args = parser.parse_args()

    trial_seeds = [int(s.strip()) for s in args.trial_seeds.split(",")]
    multi = len(trial_seeds) > 1
    dataset_key = args.dataset

    # Resolve analyses
    if args.analyses == "all":
        analyses = ["ablation", "backbone", "rep_compare", "hyperparam", "noise"]
    else:
        analyses = [a.strip() for a in args.analyses.split(",")]

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Trial seeds: {trial_seeds}  ({'multi-trial mean±std' if multi else 'single-trial'})")
    if dataset_key:
        print(f"Dataset:     {dataset_key}")
    print(f"Analyses:    {analyses}")
    print()

    all_latex = {}
    generators = {
        "ablation":    ("ablation_table.tex",    lambda: generate_ablation_table(trial_seeds)),
        "backbone":    ("backbone_comparison_table.tex", lambda: generate_backbone_table(trial_seeds, dataset_key)),
        "rep_compare": ("rep_compare_table.tex", lambda: generate_rep_compare_table(trial_seeds)),
        "hyperparam":  ("hyperparam_sensitivity_table.tex", generate_hyperparam_table),
        "noise":       ("noise_robustness_table.tex", generate_noise_table),
    }

    for analysis in analyses:
        if analysis not in generators:
            print(f"[WARN] Unknown analysis: {analysis}")
            continue

        filename, generator = generators[analysis]
        print(f"[{analysis}] Generating {filename} ...")
        try:
            latex = generator()
            all_latex[filename] = latex

            if args.dry_run:
                print(latex)
                print()
            else:
                out_path = output_dir / filename
                preamble = _write_latex_preamble()
                out_path.write_text(preamble + latex)
                print(f"  → {out_path}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback
            traceback.print_exc()

    # ── Combined file ──
    if not args.dry_run and all_latex:
        combined_path = output_dir / "all_tables.tex"
        combined = _write_latex_preamble()
        for filename in sorted(all_latex):
            combined += f"\n% ==== {filename} ====\n"
            combined += all_latex[filename] + "\n"
        combined_path.write_text(combined)
        print(f"\nCombined: {combined_path}")

    print(f"\nDone. {len(all_latex)} table(s) generated.")


if __name__ == "__main__":
    main()
