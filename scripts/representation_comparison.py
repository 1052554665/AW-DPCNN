#!/usr/bin/env python3
"""
Representation Comparison — Dataset Builder
============================================
Generates parallel datasets for every combination of time‑frequency
representation × temporal encoding, using the same file‑level split
so that comparisons are strictly fair.

Time‑frequency methods
----------------------
  mel        Mel spectrogram (librosa)
  stft       STFT spectrogram (librosa → dB)
  cwt        Morlet CWT scalogram (pywt)

Temporal encoding methods
-------------------------
  gadf       Gramian Angular Difference Field (pyts)
  gasf       Gramian Angular Summation Field (pyts)
  mtf        Markov Transition Field (pyts)
  rp         Recurrence Plot (pyts)

All combinations are fused via AW‑DPCNN (γ=4, N=20).

Output structure::

    datasets/rep_compare/
        mel_gadf/    mel_gasf/    mel_mtf/    mel_rp/
        stft_gadf/   stft_gasf/   stft_mtf/   stft_rp/
        cwt_gadf/    cwt_gasf/    cwt_mtf/    cwt_rp/
            train/{Class}/  val/{Class}/  test/{Class}/  metadata.csv

Usage::

    # All 12 combinations (default)
    python scripts/representation_comparison.py --workers 16

    # Single combination
    python scripts/representation_comparison.py --tf mel --temporal gadf

    # Dry-run
    python scripts/representation_comparison.py --dry-run
"""

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TripleDES.*")

# ── Import core AW‑DPCNN pipeline ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_CWRU_dataset import (  # noqa: E402
    _file_level_split,
    _gray_to_pseudo,
    aw_dpcnn_fusion_color,
    process_one_window,
)


# ═══════════════════════════════════════════════════════════════════════
#  Time‑Frequency representation generators
# ═══════════════════════════════════════════════════════════════════════

def _to_pseudo_image(array_2d: np.ndarray, img_size: int,
                     cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Normalise a 2‑d array to [0,255] uint8 and apply a colormap."""
    arr = array_2d.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr_u8 = (arr * 255).astype(np.uint8)
    arr_u8 = cv2.resize(arr_u8, (img_size, img_size))
    return _gray_to_pseudo(arr_u8, cmap)


def generate_mel(signal: np.ndarray, sr: int, img_size: int = 224,
                 n_fft: int = 2048, hop_length: int = 256,
                 n_mels: int = 128, fmax: int = 8000,
                 cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Mel spectrogram (reference method)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=fmax, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _to_pseudo_image(mel_db, img_size, cmap)


def generate_stft(signal: np.ndarray, sr: int, img_size: int = 224,
                  n_fft: int = 2048, hop_length: int = 256,
                  cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Linear STFT spectrogram in dB."""
    import librosa
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return _to_pseudo_image(D_db, img_size, cmap)


def generate_cwt(signal: np.ndarray, sr: int, img_size: int = 224,
                 n_scales: int = 128, cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Morlet CWT scalogram via PyWavelets."""
    import pywt

    max_scale = n_scales
    scales = np.arange(1, max_scale + 1)

    # For speed, downsample long signals for CWT computation
    if len(signal) > 16384:
        step = max(1, len(signal) // 16384)
        signal_cwt = signal[::step]
        # Adjust scales proportionally
        scales = np.linspace(1, max_scale * step, n_scales)
    else:
        signal_cwt = signal

    coefs, _ = pywt.cwt(signal_cwt, scales, 'morl', sampling_period=1 / sr)

    # Take magnitude
    cwt_mag = np.abs(coefs)

    # Resize to match image dimensions
    return _to_pseudo_image(cwt_mag, img_size, cmap)


# ═══════════════════════════════════════════════════════════════════════
#  Temporal encoding generators
# ═══════════════════════════════════════════════════════════════════════

def _time_series_to_image(signal: np.ndarray, img_size: int,
                          transformer, flatten: bool = True,
                          sequence_length: int = None) -> np.ndarray:
    """Convert a 1‑d signal to an image using a pyts transformer.

    Parameters
    ----------
    sequence_length : int or None
        Max number of time‑steps to use before GAF/Markov/RP transform.
        If longer, the signal is downsampled to this length.
        Default ``None`` → ``img_size * 4``.
    """
    import librosa
    sig = signal.astype(np.float64)

    # Determine target sequence length
    target_len = sequence_length if sequence_length is not None else img_size * 4

    # If signal is longer than target, downsample temporally
    if len(sig) > target_len:
        sig = librosa.resample(sig, orig_sr=len(sig), target_sr=target_len)

    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
    sig = sig * 2.0 - 1.0
    sig = np.clip(sig, -1.0, 1.0)

    img = transformer.fit_transform(sig.reshape(1, -1))[0]

    # Normalise and colormap
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.resize(img_u8, (img_size, img_size))
    return _gray_to_pseudo(img_u8, cv2.COLORMAP_VIRIDIS)


def generate_gadf(signal: np.ndarray, img_size: int = 224,
                  sequence_length: int = None) -> np.ndarray:
    from pyts.image import GramianAngularField
    return _time_series_to_image(
        signal, img_size,
        GramianAngularField(image_size=img_size, method='difference'),
        sequence_length=sequence_length,
    )


def generate_gasf(signal: np.ndarray, img_size: int = 224) -> np.ndarray:
    from pyts.image import GramianAngularField
    return _time_series_to_image(
        signal, img_size,
        GramianAngularField(image_size=img_size, method='summation'),
    )


def generate_mtf(signal: np.ndarray, img_size: int = 224) -> np.ndarray:
    from pyts.image import MarkovTransitionField
    return _time_series_to_image(
        signal, img_size,
        MarkovTransitionField(image_size=img_size),
    )


def generate_rp(signal: np.ndarray, img_size: int = 224) -> np.ndarray:
    from pyts.image import RecurrencePlot
    return _time_series_to_image(
        signal, img_size,
        RecurrencePlot(dimension=1, time_delay=1, threshold='point',
                       percentage=20),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

TF_GENERATORS: Dict[str, callable] = {
    "mel":  generate_mel,
    "stft": generate_stft,
    "cwt":  generate_cwt,
}

TEMPORAL_GENERATORS: Dict[str, callable] = {
    "gadf": generate_gadf,
    "gasf": generate_gasf,
    "mtf":  generate_mtf,
    "rp":   generate_rp,
}


# ═══════════════════════════════════════════════════════════════════════
#  Task: generate one fused image from one signal window
# ═══════════════════════════════════════════════════════════════════════

def _generate_one(args: tuple) -> int:
    """Generate TF + temporal + AW‑DPCNN fusion for one window.  Returns 1/0."""
    (signal, sr, out_path, img_size,
     tf_fn, temporal_fn,
     n_iter, gamma, cmap) = args
    try:
        tf_img = tf_fn(signal, sr, img_size=img_size, cmap=cmap)
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
#  Main
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Representation comparison dataset builder",
    )
    p.add_argument("--input-dir", default="raw-data/transformer-five",
                   help="Source .wav directory (class sub‑folders)")
    p.add_argument("--output-root", default="datasets/rep_compare",
                   help="Root output directory")
    p.add_argument("--tf", default="all",
                   help="Time‑frequency method(s): mel, stft, cwt, all")
    p.add_argument("--temporal", default="all",
                   help="Temporal method(s): gadf, gasf, mtf, rp, all")

    # Sliding window
    p.add_argument("--win-len", type=int, default=8192)
    p.add_argument("--hop-len", type=int, default=4096)

    # Image
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cmap-name", default="viridis")

    # AW-DPCNN
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--gamma", type=float, default=4.0)

    # Split
    p.add_argument("--file-split", type=str, default="60,20,20")
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p


def _parse_cmap(name: str) -> int:
    mapping = {
        "viridis": cv2.COLORMAP_VIRIDIS, "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET, "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT, "cool": cv2.COLORMAP_COOL,
    }
    return mapping.get(name.lower(), cv2.COLORMAP_VIRIDIS)


def main():
    args = build_parser().parse_args()
    cmap = _parse_cmap(args.cmap_name)

    # Parse split ratios
    parts = [float(x.strip()) for x in args.file_split.split(",")]
    ratios = tuple(p / sum(parts) for p in parts)

    # Resolve methods
    tf_methods = list(TF_GENERATORS) if args.tf == "all" else [args.tf]
    temporal_methods = list(TEMPORAL_GENERATORS) if args.temporal == "all" else [args.temporal]

    # ── Collect source files ──
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        sys.exit(1)

    files_by_class: dict = {}
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        if wavs:
            files_by_class[class_dir.name] = [str(p) for p in wavs]

    if not files_by_class:
        print("[ERROR] No .wav files found")
        sys.exit(1)

    total_files = sum(len(v) for v in files_by_class.values())
    print(f"Source: {total_files} .wav files in {len(files_by_class)} classes")

    # ── File‑level split (same for ALL representations) ──
    train_map, val_map, test_map = _file_level_split(
        files_by_class, ratios, args.split_seed,
    )
    split_map = {"train": train_map, "val": val_map, "test": test_map}

    print(f"Split: train={sum(len(v) for v in train_map.values())}  "
          f"val={sum(len(v) for v in val_map.values())}  "
          f"test={sum(len(v) for v in test_map.values())}")

    # ── Load all signals into memory once ──
    from scipy.io import wavfile
    path_to_signal: dict = {}
    for cls, paths in files_by_class.items():
        for p in paths:
            sr, data = wavfile.read(p)
            if data.ndim > 1:
                data = data.mean(axis=1)
            path_to_signal[p] = (sr, data.astype(np.float32))

    # ── For each (tf × temporal) combination ──
    combinations = [(tf, te) for tf in tf_methods for te in temporal_methods]
    print(f"\nCombinations to build: {len(combinations)}")
    for tf_name, te_name in combinations:
        print(f"  {tf_name} × {te_name}")

    if args.dry_run:
        print("\n[Dry run — no images generated]")
        return

    for tf_name, te_name in combinations:
        combo_name = f"{tf_name}_{te_name}"
        output_dir = os.path.join(args.output_root, combo_name)
        print(f"\n{'='*60}\n  {combo_name}\n{'='*60}")

        tf_fn = TF_GENERATORS[tf_name]
        te_fn = TEMPORAL_GENERATORS[te_name]

        tasks = []
        metadata_rows = []

        for split_name in ["train", "val", "test"]:
            split_cls = split_map.get(split_name, {})
            for cls, file_paths in sorted(split_cls.items()):
                out_cls_dir = os.path.join(output_dir, split_name, cls)
                for fpath in file_paths:
                    sr_val, signal = path_to_signal[fpath]
                    stem = Path(fpath).stem

                    win_len, hop_len = args.win_len, args.hop_len
                    if win_len <= 0 or win_len >= len(signal):
                        out_path = os.path.join(out_cls_dir, f"{stem}.png")
                        if not args.overwrite and os.path.exists(out_path):
                            continue
                        tasks.append((
                            signal, sr_val, out_path, args.img_size,
                            tf_fn, te_fn, args.n_iter, args.gamma, cmap,
                        ))
                        metadata_rows.append({
                            "filename": f"{stem}.png",
                            "class_label": cls,
                            "source_file": os.path.basename(fpath),
                            "split": split_name,
                            "tf_method": tf_name,
                            "temporal_method": te_name,
                        })
                    else:
                        idx = 0
                        for start in range(0, len(signal) - win_len + 1, hop_len):
                            window = signal[start:start + win_len]
                            fname = f"{stem}_{idx:05d}.png"
                            out_path = os.path.join(out_cls_dir, fname)
                            if not args.overwrite and os.path.exists(out_path):
                                idx += 1
                                continue
                            tasks.append((
                                window, sr_val, out_path, args.img_size,
                                tf_fn, te_fn, args.n_iter, args.gamma, cmap,
                            ))
                            metadata_rows.append({
                                "filename": fname,
                                "class_label": cls,
                                "source_file": os.path.basename(fpath),
                                "split": split_name,
                                "tf_method": tf_name,
                                "temporal_method": te_name,
                            })
                            idx += 1

        print(f"  Tasks: {len(tasks)}")
        if not tasks:
            print("  [SKIP] Nothing to do")
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(
                executor.map(_generate_one, tasks, chunksize=8),
                total=len(tasks), desc=f"  {combo_name}", ncols=100,
            ))

        ok = sum(results)
        print(f"  Done: {ok}/{len(tasks)} images")

        # Save metadata
        if metadata_rows:
            csv_path = os.path.join(output_dir, "metadata.csv")
            fieldnames = ["filename", "class_label", "source_file", "split",
                          "tf_method", "temporal_method"]
            os.makedirs(output_dir, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)

    print(f"\n{'='*60}")
    print("All combinations built.")
    print(f"Output root: {args.output_root}")
    print(f"Train each with: python scripts/train.py --config configs/default.yaml "
          f"--exp-config <custom>  (point dataset.root_dir to the variant folder)")


if __name__ == "__main__":
    main()
