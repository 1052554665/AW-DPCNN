#!/usr/bin/env python3
"""
CWRU Representation Comparison — Dataset Builder
=================================================
Builds AW‑DPCNN fused images for **all 12** time‑frequency × temporal
encoding combinations from the three CWRU bearing fault sub‑datasets.

This is the CWRU‑specific counterpart of ``representation_comparison.py``
and is designed for .mat input files (instead of .wav).

Time‑frequency methods
----------------------
  mel        Mel spectrogram
  stft       STFT spectrogram (dB)
  cwt        Morlet CWT scalogram

Temporal encoding methods
-------------------------
  gadf       Gramian Angular Difference Field
  gasf       Gramian Angular Summation Field
  mtf        Markov Transition Field
  rp         Recurrence Plot

All 12 combinations are fused via AW‑DPCNN (γ=4, N=20).

Supported datasets
------------------
  --dataset 12k_de    12 kHz Drive‑End   (DE_time,  sr=12000)
  --dataset 12k_fe    12 kHz Fan‑End     (FE_time,  sr=12000)
  --dataset 48k_de    48 kHz Drive‑End   (DE_time,  sr=48000)
  --all                Build all three

Output structure (ImageFolder‑compatible, 10 classes, no split)::

    raw-data/rep_compare_12k_de/
        mel_gadf/    mel_gasf/    mel_mtf/    mel_rp/
        stft_gadf/   stft_gasf/   stft_mtf/   stft_rp/
        cwt_gadf/    cwt_gasf/    cwt_mtf/    cwt_rp/
            {BF007,BF014,...,Normal}/  metadata.csv

Target classes
--------------
  BF007   BF014   BF021      (ball faults    — 0.007", 0.014", 0.021")
  IF007   IF014   IF021      (inner race     — 0.007", 0.014", 0.021")
  OF007   OF014   OF021      (outer race @6  — 0.007", 0.014", 0.021")
  Normal

Data source
-----------
  raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data/
    B/{007,014,021}/           (ball fault .mat files)
    IR/{007,014,021}/          (inner race .mat files)
    OR/{007,014,021}/@6/       (outer race @ 6 o'clock)
  raw-data/CWRU-dataset/Normal/ (normal baseline .mat files)

Usage::

    # All 12 combinations for one dataset
    python scripts/build_cwru_rep_datasets.py --dataset 12k_de --workers 32

    # Single combination
    python scripts/build_cwru_rep_datasets.py --dataset 12k_de \
        --tf mel --temporal gadf

    # All three datasets, all combinations
    python scripts/build_cwru_rep_datasets.py --all --workers 32

    # Dry-run
    python scripts/build_cwru_rep_datasets.py --dataset 12k_de --dry-run
"""

import argparse
import csv
import os
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TripleDES.*")

# ── Import generators & fusion from sibling scripts ──
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_cwru_dataset import aw_dpcnn_fusion_color  # noqa: E402
from representation_comparison import (  # noqa: E402
    TF_GENERATORS,
    TEMPORAL_GENERATORS,
)


# ═══════════════════════════════════════════════════════════════════════
#  Dataset definitions
# ═══════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "12k_de": {
        "name": "12k Drive‑End",
        "src_dir": "raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "DE_time",
        "sr": 12000,
        "fmax": 6000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_len": 2048,
        "hop_len": 1024,
        "out_root": "datasets/rep_compare_12k_de",
    },
    "12k_fe": {
        "name": "12k Fan‑End",
        "src_dir": "raw-data/CWRU-dataset/12k_Fan_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "FE_time",
        "sr": 12000,
        "fmax": 6000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_len": 2048,
        "hop_len": 1024,
        "out_root": "datasets/rep_compare_12k_fe",
    },
    "48k_de": {
        "name": "48k Drive‑End",
        "src_dir": "raw-data/CWRU-dataset/48k_Drive_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "DE_time",
        "sr": 48000,
        "fmax": 24000,
        "n_fft": 4096,
        "hop_length": 512,
        "win_len": 8192,
        "hop_len": 4096,
        "out_root": "datasets/rep_compare_48k_de",
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  CWRU class mapping
# ═══════════════════════════════════════════════════════════════════════

CLASS_MAP = OrderedDict([
    (("B",  "007"), "BF007"),
    (("B",  "014"), "BF014"),
    (("B",  "021"), "BF021"),
    (("IR", "007"), "IF007"),
    (("IR", "014"), "IF014"),
    (("IR", "021"), "IF021"),
    (("OR", "007"), "OF007"),
    (("OR", "014"), "OF014"),
    (("OR", "021"), "OF021"),
])

NORMAL_CLASS = "Normal"


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_mat_signal(mat_path: str, sensor_key: str) -> np.ndarray:
    """Load a specific sensor signal from a CWRU .mat file."""
    mat = loadmat(mat_path)
    for key in mat.keys():
        if sensor_key in key:
            signal = mat[key].squeeze().astype(np.float32)
            if signal.ndim != 1:
                signal = signal.ravel()
            return signal
    raise ValueError(f"No '{sensor_key}' key found in {mat_path} (keys: "
                     f"{[k for k in mat.keys() if not k.startswith('_')]})")


def _collect_mat_files(src_dir: str, normal_dir: str) -> Dict[str, List[str]]:
    """Walk CWRU directory tree and collect .mat file paths per output class."""
    files_by_class: Dict[str, List[str]] = OrderedDict()
    src_path = Path(src_dir)

    if not src_path.exists():
        print(f"[ERROR] Source directory not found: {src_path}")
        sys.exit(1)

    for (fault_type, severity), out_class in CLASS_MAP.items():
        if fault_type == "OR":
            sub_dir = src_path / fault_type / severity / "@6"
            if not sub_dir.exists():
                sub_dir = src_path / fault_type / severity
        else:
            sub_dir = src_path / fault_type / severity

        if sub_dir.exists():
            mat_files = sorted(sub_dir.glob("*.mat"))
            if mat_files:
                files_by_class[out_class] = [str(p) for p in mat_files]
            else:
                print(f"[WARN] No .mat files in {sub_dir}")
        else:
            print(f"[WARN] Directory not found: {sub_dir}")

    normal_path = Path(normal_dir)
    if normal_path.exists():
        mat_files = sorted(normal_path.glob("*.mat"))
        if mat_files:
            files_by_class[NORMAL_CLASS] = [str(p) for p in mat_files]

    return files_by_class


# ═══════════════════════════════════════════════════════════════════════
#  Per‑window generation task
# ═══════════════════════════════════════════════════════════════════════

def _generate_one(args: tuple) -> int:
    """Generate one fused image for a specific (TF, temporal) combination.

    Returns 1 on success, 0 on failure.
    """
    (signal, sr, out_path, img_size,
     tf_fn, temporal_fn, tf_kwargs,
     n_iter, gamma) = args
    try:
        tf_img = tf_fn(signal, sr, img_size=img_size, **tf_kwargs)
        temp_img = temporal_fn(signal, img_size=img_size)
        fused = aw_dpcnn_fusion_color(tf_img, temp_img,
                                       n_iter=n_iter, gamma=gamma)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, fused)
        return 1
    except Exception as exc:
        print(f"[ERROR] {out_path}: {exc}")
        return 0


# ═══════════════════════════════════════════════════════════════════════
#  Core: build one dataset
# ═══════════════════════════════════════════════════════════════════════

def build_dataset(
    config: dict,
    tf_methods: List[str] = None,
    temporal_methods: List[str] = None,
    workers: int = 8,
    overwrite: bool = False,
    img_size: int = 224,
    n_iter: int = 20,
    gamma: float = 4.0,
) -> None:
    """Build all (TF × temporal) fused datasets for one CWRU sub‑dataset."""

    name = config["name"]
    src_dir = config["src_dir"]
    normal_dir = config["normal_dir"]
    sensor_key = config["sensor_key"]
    sr = config["sr"]
    fmax = config["fmax"]
    n_fft = config["n_fft"]
    hop_length_stft = config["hop_length"]
    win_len = config["win_len"]
    hop_len = config["hop_len"]
    out_root = config["out_root"]

    if tf_methods is None:
        tf_methods = list(TF_GENERATORS)
    if temporal_methods is None:
        temporal_methods = list(TEMPORAL_GENERATORS)

    combinations = [(tf, te) for tf in tf_methods for te in temporal_methods]

    print(f"\n{'='*60}")
    print(f"  Building: {name}  (sensor={sensor_key}, sr={sr} Hz)")
    print(f"  Combinations: {len(combinations)}")
    for tf_name, te_name in combinations:
        print(f"    {tf_name} × {te_name}")
    print(f"{'='*60}")

    # ── Collect .mat files ──
    files_by_class = _collect_mat_files(src_dir, normal_dir)
    if not files_by_class:
        print("[ERROR] No .mat files collected.")
        return

    total_files = sum(len(v) for v in files_by_class.values())
    print(f"\nSource: {total_files} .mat files in {len(files_by_class)} classes")
    for cls, paths in files_by_class.items():
        print(f"  {cls:8s}: {len(paths)} files")

    # ── Load all signals into memory ──
    print("\nLoading signals into memory...")
    path_to_signal: Dict[str, np.ndarray] = {}
    for cls, paths in files_by_class.items():
        for p in tqdm(paths, desc=f"  {cls}", ncols=100):
            try:
                signal = _load_mat_signal(p, sensor_key)
                path_to_signal[p] = signal
            except Exception as exc:
                print(f"\n[WARN] Skipping {p}: {exc}")

    files_by_class = {
        cls: [p for p in paths if p in path_to_signal]
        for cls, paths in files_by_class.items()
    }
    files_by_class = {cls: paths for cls, paths in files_by_class.items() if paths}

    # ── Pre‑segment all windows once (shared across all combinations) ──
    windows: List[tuple] = []
    for cls, file_paths in sorted(files_by_class.items()):
        for fpath in file_paths:
            signal = path_to_signal[fpath]
            stem = Path(fpath).stem

            if win_len <= 0 or win_len >= len(signal):
                windows.append((signal, cls, stem, 0, 0))
            else:
                idx = 0
                for start in range(0, len(signal) - win_len + 1, hop_len):
                    window = signal[start:start + win_len]
                    windows.append((window, cls, stem, idx, start))
                    idx += 1

    print(f"Total signal windows: {len(windows)}")

    # ── Per‑TF‑method kwargs (cwt does not accept n_fft/hop_length/fmax) ──
    tf_kwargs_map = {
        "mel":  dict(n_fft=n_fft, hop_length=hop_length_stft, fmax=fmax),
        "stft": dict(n_fft=n_fft, hop_length=hop_length_stft),
        "cwt":  {},
    }

    # ── For each (TF × temporal) combination ──
    for tf_name, te_name in combinations:
        combo_name = f"{tf_name}_{te_name}"
        output_dir = os.path.join(out_root, combo_name)
        print(f"\n{'─'*60}\n  {combo_name}\n{'─'*60}")

        tf_fn = TF_GENERATORS[tf_name]
        te_fn = TEMPORAL_GENERATORS[te_name]

        tasks = []
        metadata_rows = []

        for signal, cls, stem, win_idx, start_sample in windows:
            fname = f"{stem}_{win_idx:05d}.png"
            out_cls_dir = os.path.join(output_dir, cls)
            out_path = os.path.join(out_cls_dir, fname)

            if not overwrite and os.path.exists(out_path):
                continue

            tasks.append((
                signal, sr, out_path, img_size,
                tf_fn, te_fn, tf_kwargs_map[tf_name],
                n_iter, gamma,
            ))
            metadata_rows.append({
                "filename": fname,
                "class_label": cls,
                "source_file": stem,
                "window_idx": win_idx,
                "window_start_sample": start_sample,
                "tf_method": tf_name,
                "temporal_method": te_name,
            })

        print(f"  Tasks: {len(tasks)}")
        if not tasks:
            print("  [SKIP] Nothing to do")
            continue

        # Save metadata early (before heavy computation)
        if metadata_rows:
            csv_path = os.path.join(output_dir, "metadata.csv")
            fieldnames = ["filename", "class_label", "source_file",
                          "window_idx", "window_start_sample",
                          "tf_method", "temporal_method"]
            os.makedirs(output_dir, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)

        # Parallel generation
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(_generate_one, tasks, chunksize=8),
                total=len(tasks), desc=f"  {combo_name}", ncols=100,
            ))
        ok = sum(results)
        print(f"  Done: {ok}/{len(tasks)} images ({100*ok/len(tasks):.1f}%)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  {name} — all combinations built.")
    for tf_name, te_name in combinations:
        combo_dir = Path(out_root) / f"{tf_name}_{te_name}"
        if combo_dir.exists():
            count = sum(1 for _ in combo_dir.rglob("*.png"))
            print(f"    {tf_name}_{te_name}: {count} images")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CWRU Representation Comparison — Dataset Builder",
    )
    parser.add_argument(
        "--dataset", default=None,
        choices=["12k_de", "12k_fe", "48k_de"],
        help="Which CWRU sub‑dataset to build",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Build all three datasets at once",
    )

    # Representation selection
    parser.add_argument("--tf", default="all",
                        help="Time‑frequency method(s): mel, stft, cwt, all")
    parser.add_argument("--temporal", default="all",
                        help="Temporal method(s): gadf, gasf, mtf, rp, all")

    # Image
    parser.add_argument("--img-size", type=int, default=224)

    # AW-DPCNN
    parser.add_argument("--n-iter", type=int, default=20,
                        help="PCNN iterations (default: 20)")
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="Contrast amplification factor")

    # Execution
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify --dataset or --all")

    # Resolve methods
    tf_methods = list(TF_GENERATORS) if args.tf == "all" else [args.tf]
    temporal_methods = (list(TEMPORAL_GENERATORS) if args.temporal == "all"
                        else [args.temporal])

    if args.dry_run:
        print("[Dry run] Would build:")
        datasets = list(DATASET_CONFIGS) if args.all else [args.dataset]
        for ds_key in datasets:
            config = DATASET_CONFIGS[ds_key]
            combos = [(tf, te) for tf in tf_methods for te in temporal_methods]
            print(f"\n  {config['name']}:")
            for tf_name, te_name in combos:
                print(f"    → {config['out_root']}/{tf_name}_{te_name}/")
        return

    datasets_to_build = list(DATASET_CONFIGS) if args.all else [args.dataset]

    for ds_key in datasets_to_build:
        config = DATASET_CONFIGS[ds_key]
        build_dataset(
            config=config,
            tf_methods=tf_methods,
            temporal_methods=temporal_methods,
            workers=args.workers,
            overwrite=args.overwrite,
            img_size=args.img_size,
            n_iter=args.n_iter,
            gamma=args.gamma,
        )

    print("\n✓ All datasets built.")


if __name__ == "__main__":
    main()
