#!/usr/bin/env python3
"""
Unified AW-DPCNN Dataset Builder (v2)
======================================
Single-script pipeline that combines the three stages:
  1. Mel spectrogram generation
  2. GADF image generation
  3. AW-DPCNN fusion  (γ=4, N=20 — consistent with paper)

Supports **.wav** (transformer dataset) and **.mat** (CWRU dataset) inputs.
Optional file‑level train/val/test splitting prevents data leakage between
segments derived from the same recording.

All three stages run **in memory per window** — no intermediate PNG files are
written unless `--save-intermediates` is requested.

Usage (CWRU .mat, with file‑level split)::

    python scripts/build_fused_dataset.py \
        --input-dir ./raw-data/cwru_raw_007 \
        --output-dir ./datasets/cwru_within \
        --input-format mat --sr 12000 \
        --win-len 2048 --hop-len 1024 \
        --n-fft 1024 --n-mels 128 --fmax 6000 \
        --file-split 50,25,25 --split-seed 42 \
        --metadata --workers 16

Usage (CWRU cross‑severity — no split, two separate runs)::

    python scripts/build_fused_dataset.py \
        --input-dir ./raw-data/cwru_raw_007 --output-dir ./datasets/cwru_cross/train \
        --input-format mat --sr 12000 \
        --win-len 2048 --hop-len 1024 \
        --n-fft 1024 --n-mels 128 --fmax 6000

    python scripts/build_fused_dataset.py \
        --input-dir ./raw-data/cwru_raw_014 --output-dir ./datasets/cwru_cross/test \
        --input-format mat --sr 12000 \
        --win-len 2048 --hop-len 1024 \
        --n-fft 1024 --n-mels 128 --fmax 6000

Usage (transformer .wav, pre‑split)::

    python scripts/build_fused_dataset.py \
        --input-dir ./raw_wavs/train --output-dir ./datasets/train \
        --win-len 3000 --hop-len 750 --img-size 224 \
        --workers 16

Output structure (ImageFolder‑compatible)::

    datasets/
      train/
        B/  IR/  N/  OR/        # class sub‑folders
          B_00000.png  ...
"""

import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pyts.image import GramianAngularField
from scipy.io import loadmat, wavfile
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  AW-DPCNN core  (extracted from `awdpcnn.py`)
# ═══════════════════════════════════════════════════════════════════════

def aw_dpcnn_single_channel(S1: np.ndarray, S2: np.ndarray,
                            n_iter: int = 20,
                            gamma: float = 4.0) -> np.ndarray:
    """Fuse two single-channel images with AW-DPCNN.

    Parameters
    ----------
    S1, S2 : ndarray  – normalised input channels (Mel, GADF).
    n_iter : int      – PCNN iteration count (paper: N = 20).
    gamma  : float    – contrast amplification factor (paper: γ = 4).
    """
    S1 = S1.astype(np.float32)
    S2 = S2.astype(np.float32)

    S1 = (S1 - S1.min()) / (S1.max() - S1.min() + 1e-6)
    S2 = (S2 - S2.min()) / (S2.max() - S2.min() + 1e-6)

    alpha_L, alpha_T, V_T, sigma = 0.001, 0.001, 20, 0.1

    W1 = np.array([
        [-0.5, -0.5, 1, -0.5, -0.5],
        [-0.5, -0.5, 1, -0.5, -0.5],
        [-0.5, -0.5, 1, -0.5, -0.5]
    ], np.float32)

    W2 = np.array([
        [0, 0.01, 0],
        [0.01, 0.01, 0.01],
        [0, 0.01, 0]
    ], np.float32)

    M = np.eye(3, dtype=np.float32) * 0.25
    kernel = np.ones((3, 3), np.float32) / 9.0

    def local_contrast(img):
        mu = cv2.filter2D(img, -1, kernel)
        mu2 = cv2.filter2D(img * img, -1, kernel)
        return np.sqrt(np.abs(mu2 - mu * mu))

    C1 = local_contrast(S1)
    C2 = local_contrast(S2)

    beta1 = gamma * C1 / (gamma * C1 + C2 + 1e-6)
    beta2 = C2 / (gamma * C1 + C2 + 1e-6)

    L = np.zeros_like(S1)
    T = np.zeros_like(S1)
    Y = np.zeros_like(S1)
    U_sum = np.zeros_like(S1)

    for _ in range(n_iter):
        F1 = cv2.filter2D(Y, -1, W1) + S1
        F2 = cv2.filter2D(Y, -1, W2) + S2
        L = np.exp(-alpha_L) * L + cv2.filter2D(Y, -1, M)

        U = L * (1 + beta1 * F1) * (1 + beta2 * F2) + sigma
        Y = (U > T).astype(np.float32)
        T = np.exp(-alpha_T) * T + V_T * Y
        U_sum += U

    return (U_sum - U_sum.min()) / (U_sum.max() - U_sum.min() + 1e-6)


def aw_dpcnn_fusion_color(mel_img: np.ndarray, gaf_img: np.ndarray,
                          n_iter: int = 20,
                          gamma: float = 4.0) -> np.ndarray:
    """Fuse two BGR images channel‑wise with AW-DPCNN."""
    if mel_img.shape[:2] != gaf_img.shape[:2]:
        gaf_img = cv2.resize(gaf_img, (mel_img.shape[1], mel_img.shape[0]))

    mel_ch = cv2.split(mel_img)
    gaf_ch = cv2.split(gaf_img)

    fused = []
    for i in range(3):
        f = aw_dpcnn_single_channel(mel_ch[i], gaf_ch[i],
                                     n_iter=n_iter, gamma=gamma)
        fused.append((f * 255).astype(np.uint8))

    return cv2.merge(fused)


# ═══════════════════════════════════════════════════════════════════════
#  Mel spectrogram  (extracted from `mel.py` logic)
# ═══════════════════════════════════════════════════════════════════════

def _gray_to_pseudo(gray_uint8: np.ndarray,
                    cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Convert a single‑channel uint8 image to a 3‑channel colormap image."""
    return cv2.applyColorMap(gray_uint8, cmap)


def generate_mel_image(signal: np.ndarray, sr: int,
                       n_fft: int = 2048,
                       hop_length: int = 256,
                       n_mels: int = 256,
                       fmax: int = 8000,
                       img_size: int = 224,
                       cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Compute Mel spectrogram and return a BGR pseudo‑colour image (H,W,3)."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=fmax, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_uint8 = (mel_norm * 255).astype(np.uint8)
    mel_uint8 = cv2.resize(mel_uint8, (img_size, img_size))
    return _gray_to_pseudo(mel_uint8, cmap)


# ═══════════════════════════════════════════════════════════════════════
#  GADF image  (extracted from `GAF.py` logic)
# ═══════════════════════════════════════════════════════════════════════

def generate_gadf_image(signal: np.ndarray,
                        img_size: int = 224,
                        method: str = 'difference',
                        cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Compute GADF and return a BGR pseudo‑colour image (H,W,3)."""
    signal = signal.astype(np.float32)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    signal = signal * 2.0 - 1.0
    signal = np.clip(signal, -1.0, 1.0)

    # Reuse a single transformer — not thread‑safe, so instantiate per call
    gadf_trans = GramianAngularField(image_size=img_size, method=method)
    gadf = gadf_trans.fit_transform(signal.reshape(1, -1))[0]

    gadf_norm = (gadf - gadf.min()) / (gadf.max() - gadf.min() + 1e-8)
    gadf_uint8 = (gadf_norm * 255).astype(np.uint8)
    return _gray_to_pseudo(gadf_uint8, cmap)


# ═══════════════════════════════════════════════════════════════════════
#  .mat file support (CWRU dataset)
# ═══════════════════════════════════════════════════════════════════════

def load_mat_signal(mat_path: str) -> np.ndarray:
    """Load the DE_time (drive‑end) signal from a CWRU .mat file.

    CWRU .mat files use variable names like ``X118_DE_time`` — we search for
    any key containing ``DE_time``.
    """
    mat = loadmat(mat_path)
    for key in mat.keys():
        if 'DE_time' in key:
            signal = mat[key].squeeze().astype(np.float32)
            if signal.ndim != 1:
                # Some files store a row vector; flatten if needed
                signal = signal.ravel()
            return signal
    raise ValueError(f'No DE_time key found in {mat_path}')


# ═══════════════════════════════════════════════════════════════════════
#  File‑level train/val/test split
# ═══════════════════════════════════════════════════════════════════════

def _file_level_split(
    files_by_class: dict,
    ratios: tuple,
    seed: int = 42,
) -> tuple:
    """Split file paths per class into train / val / test.

    Parameters
    ----------
    files_by_class : dict  {class_name: [file_path, ...]}
    ratios : tuple         (train_ratio, val_ratio, test_ratio), e.g. (0.5, 0.25, 0.25)
    seed   : int           random seed for reproducibility.

    Returns
    -------
    (train_map, val_map, test_map)  – each is {class_name: [file_path, ...]}
    """
    rng = random.Random(seed)
    train_map, val_map, test_map = {}, {}, {}
    r_train, r_val, r_test = ratios

    for cls, files in files_by_class.items():
        files = sorted(files)  # deterministic ordering before shuffle
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, round(n * r_train))
        n_val   = max(1, round(n * r_val))
        # Ensure we don't exceed available files
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val   = max(1, n - n_train - 1)
        train_map[cls] = files[:n_train]
        val_map[cls]   = files[n_train:n_train + n_val]
        test_map[cls]  = files[n_train + n_val:]

    return train_map, val_map, test_map


# ═══════════════════════════════════════════════════════════════════════
#  Distribution verification helpers
# ═══════════════════════════════════════════════════════════════════════

def _compute_js_divergence(counts_a: dict, counts_b: dict) -> float:
    """Jensen–Shannon divergence between two class‑count dictionaries."""
    all_classes = sorted(set(counts_a) | set(counts_b))
    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1

    p = np.array([counts_a.get(c, 0) / total_a for c in all_classes])
    q = np.array([counts_b.get(c, 0) / total_b for c in all_classes])
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = (x > 0) & (y > 0)
        return np.sum(x[mask] * np.log(x[mask] / y[mask]))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _save_metadata(csv_path: str, rows: list):
    """Write metadata.csv with columns:
    filename, class_label, source_file, split, window_idx, window_start_sample.
    """
    fieldnames = ['filename', 'class_label', 'source_file', 'split',
                  'window_idx', 'window_start_sample']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _detect_input_format(input_dir: str) -> str:
    """Auto‑detect whether the input directory contains .mat or .wav files."""
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith('.mat'):
                return 'mat'
            if fname.lower().endswith('.wav'):
                return 'wav'
    return 'wav'  # default fallback

def process_one_window(args: tuple) -> int:
    """Generate Mel + GADF from one signal window, fuse, and save.

    Returns 1 on success, 0 on failure.
    """
    (signal, sr, out_path, img_size, n_iter, n_fft, hop_length,
     n_mels, fmax, cmap, gaf_method, save_intermediates, gamma) = args

    try:
        mel_img = generate_mel_image(
            signal, sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmax=fmax, img_size=img_size, cmap=cmap,
        )
        gadf_img = generate_gadf_image(
            signal, img_size=img_size, method=gaf_method, cmap=cmap,
        )
        fused = aw_dpcnn_fusion_color(mel_img, gadf_img,
                                       n_iter=n_iter, gamma=gamma)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, fused)

        if save_intermediates:
            base, _ = os.path.splitext(out_path)
            cv2.imwrite(base + '_mel.png', mel_img)
            cv2.imwrite(base + '_gadf.png', gadf_img)

        return 1
    except Exception as exc:
        print(f'[ERROR] {out_path}: {exc}')
        return 0


# ═══════════════════════════════════════════════════════════════════════
#  Task collector  (supports .wav, .mat, and optional file‑level split)
# ═══════════════════════════════════════════════════════════════════════

def _collect_tasks(
    input_dir: str,
    output_dir: str,
    win_len: int,
    hop_len: int,
    img_size: int,
    n_iter: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmax: int,
    cmap: int,
    gaf_method: str,
    save_intermediates: bool,
    overwrite: bool,
    sr_override: int = 0,
    input_format: str = 'auto',
    file_split_ratios: tuple = (),
    split_seed: int = 42,
    gamma: float = 4.0,
) -> tuple:
    """Walk the input directory and build a flat list of window tasks.

    Returns
    -------
    (tasks, metadata_rows)
        tasks          – list of tuples for ``process_one_window``
        metadata_rows  – list of dicts for ``metadata.csv`` (empty if not needed)
    """
    tasks = []
    metadata_rows = []

    # --- Auto‑detect input format ---
    if input_format == 'auto':
        input_format = _detect_input_format(input_dir)

    # --- Gather raw files grouped by class ---
    # files_by_class: {class_name: [(full_path, sr, data), ...]}
    files_by_class = {}

    for root, _, files in os.walk(input_dir):
        cls_name = os.path.basename(root)
        if not cls_name or cls_name == os.path.basename(input_dir):
            continue

        for fname in sorted(files):
            if input_format == 'mat' and fname.lower().endswith('.mat'):
                mat_path = os.path.join(root, fname)
                try:
                    signal = load_mat_signal(mat_path)
                except Exception as exc:
                    print(f'[WARN] Cannot read {mat_path}: {exc}')
                    continue
                sr = sr_override if sr_override > 0 else 12000
                files_by_class.setdefault(cls_name, []).append(
                    (mat_path, sr, signal.astype(np.float32)))

            elif input_format == 'wav' and fname.lower().endswith('.wav'):
                wav_path = os.path.join(root, fname)
                try:
                    sr, data = wavfile.read(wav_path)
                except Exception as exc:
                    print(f'[WARN] Cannot read {wav_path}: {exc}')
                    continue
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)
                files_by_class.setdefault(cls_name, []).append(
                    (wav_path, sr, data))

    if not files_by_class:
        print('[WARN] No input files found.')
        return tasks, metadata_rows

    # --- File‑level split (optional) ---
    if file_split_ratios:
        # Build a flat {cls: [file_path]} map for the splitter
        cls_file_paths = {
            cls: [item[0] for item in items]
            for cls, items in files_by_class.items()
        }
        train_map, val_map, test_map = _file_level_split(
            cls_file_paths, file_split_ratios, split_seed)

        # Build a lookup: file_path -> (sr, signal)
        path_to_signal = {}
        for items in files_by_class.values():
            for fpath, sr_val, sig in items:
                path_to_signal[fpath] = (sr_val, sig)

        split_maps = [('train', train_map), ('val', val_map), ('test', test_map)]

        for split_name, split_map in split_maps:
            for cls, file_paths in split_map.items():
                out_cls_dir = os.path.join(output_dir, split_name, cls)
                for fpath in file_paths:
                    sr_val, signal = path_to_signal[fpath]
                    prefix = os.path.splitext(os.path.basename(fpath))[0]
                    _append_window_tasks(
                        signal, sr_val, out_cls_dir, prefix, win_len, hop_len,
                        img_size, n_iter, n_fft, hop_length, n_mels, fmax,
                        cmap, gaf_method, save_intermediates, overwrite,
                        gamma, tasks,
                    )
                    # Metadata rows
                    if win_len > 0 and win_len < len(signal):
                        n_wins = len(range(0, len(signal) - win_len + 1, hop_len))
                    else:
                        n_wins = 1
                    for idx in range(n_wins):
                        metadata_rows.append({
                            'filename': f'{prefix}_{idx:05d}.png',
                            'class_label': cls,
                            'source_file': os.path.basename(fpath),
                            'split': split_name,
                            'window_idx': idx,
                            'window_start_sample': idx * hop_len,
                        })
    else:
        # --- No split — mirror input folder structure ---
        for cls, items in files_by_class.items():
            out_cls_dir = os.path.join(output_dir, cls)
            for fpath, sr_val, signal in items:
                prefix = os.path.splitext(os.path.basename(fpath))[0]
                _append_window_tasks(
                    signal, sr_val, out_cls_dir, prefix, win_len, hop_len,
                    img_size, n_iter, n_fft, hop_length, n_mels, fmax,
                    cmap, gaf_method, save_intermediates, overwrite,
                    gamma, tasks,
                )

    return tasks, metadata_rows


def _append_window_tasks(
    signal, sr, out_cls_dir, prefix, win_len, hop_len,
    img_size, n_iter, n_fft, hop_length, n_mels, fmax,
    cmap, gaf_method, save_intermediates, overwrite,
    gamma, tasks,
):
    """Create window tasks for a single signal and append to *tasks* list."""
    if win_len <= 0 or win_len >= len(signal):
        out_path = os.path.join(out_cls_dir, prefix + '.png')
        if not overwrite and os.path.exists(out_path):
            return
        tasks.append((
            signal, sr, out_path, img_size, n_iter,
            n_fft, hop_length, n_mels, fmax, cmap,
            gaf_method, save_intermediates, gamma,
        ))
    else:
        idx = 0
        for start in range(0, len(signal) - win_len + 1, hop_len):
            window = signal[start:start + win_len]
            out_path = os.path.join(out_cls_dir, f'{prefix}_{idx:05d}.png')
            if not overwrite and os.path.exists(out_path):
                idx += 1
                continue
            tasks.append((
                window, sr, out_path, img_size, n_iter,
                n_fft, hop_length, n_mels, fmax, cmap,
                gaf_method, save_intermediates, gamma,
            ))
            idx += 1


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_cmap(name: str) -> int:
    """Map a human‑readable colormap name to an OpenCV constant."""
    mapping = {
        'viridis':  cv2.COLORMAP_VIRIDIS,
        'turbo':    cv2.COLORMAP_TURBO,
        'jet':      cv2.COLORMAP_JET,
        'plasma':   cv2.COLORMAP_PLASMA,
        'inferno':  cv2.COLORMAP_INFERNO,
        'magma':    cv2.COLORMAP_MAGMA,
        'hot':      cv2.COLORMAP_HOT,
        'cool':     cv2.COLORMAP_COOL,
    }
    name_l = name.lower()
    if name_l in mapping:
        return mapping[name_l]
    raise ValueError(f'Unknown cmap "{name}". Choices: {list(mapping.keys())}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Unified AW-DPCNN fused dataset builder (v2)')

    # --- I/O ---
    p.add_argument('--input-dir', required=True,
                   help='Root directory of input files (class sub‑folders)')
    p.add_argument('--output-dir', required=True,
                   help='Root directory for fused PNG images')
    p.add_argument('--input-format', default='auto', choices=['auto', 'wav', 'mat'],
                   help='Input file format (default: auto‑detect)')
    p.add_argument('--sr', type=int, default=0,
                   help='Sample rate override (required for .mat files; '
                        'default: 12000 for CWRU)')

    # --- Sliding window ---
    p.add_argument('--win-len', type=int, default=3000,
                   help='Sliding window length in samples (0 = one image per file)')
    p.add_argument('--hop-len', type=int, default=750,
                   help='Hop length between windows (samples)')

    # --- Image ---
    p.add_argument('--img-size', type=int, default=224,
                   help='Output image size (square)')
    p.add_argument('--cmap', default='viridis',
                   help='Colormap for Mel & GAF pseudo‑colour')

    # --- Mel parameters ---
    p.add_argument('--n-fft', type=int, default=2048)
    p.add_argument('--hop-length', type=int, default=256)
    p.add_argument('--n-mels', type=int, default=256)
    p.add_argument('--fmax', type=int, default=8000)

    # --- GADF parameters ---
    p.add_argument('--gaf-method', default='difference',
                   choices=['difference', 'summation'])

    # --- AW-DPCNN parameters ---
    p.add_argument('--n-iter', type=int, default=20,
                   help='PCNN iterations (paper: N = 20)')
    p.add_argument('--gamma', type=float, default=4.0,
                   help='Contrast amplification factor (paper: γ = 4)')

    # --- File‑level split ---
    p.add_argument('--file-split', type=str, default='',
                   help='File‑level train/val/test ratios, e.g. "50,25,25"')
    p.add_argument('--split-seed', type=int, default=42,
                   help='Random seed for file‑level split')

    # --- Metadata & verification ---
    p.add_argument('--metadata', action='store_true',
                   help='Generate metadata.csv alongside fused images')
    p.add_argument('--verify', action='store_true',
                   help='Print per‑split class distribution and JS divergence')

    # --- Execution ---
    p.add_argument('--workers', type=int, default=os.cpu_count() or 4,
                   help='Number of parallel workers')
    p.add_argument('--overwrite', action='store_true',
                   help='Re‑compute and overwrite existing outputs')
    p.add_argument('--save-intermediates', action='store_true',
                   help='Also write intermediate Mel and GAF PNGs to disk')

    return p


def main():
    args = build_parser().parse_args()

    cmap_code = _parse_cmap(args.cmap)

    # --- Parse file‑split ratios ---
    file_split_ratios = ()
    if args.file_split:
        parts = [float(x.strip()) for x in args.file_split.split(',')]
        if len(parts) != 3:
            raise ValueError('--file-split requires three comma‑separated values, '
                             'e.g. "50,25,25"')
        total = sum(parts)
        file_split_ratios = tuple(p / total for p in parts)

    # --- Resolve sample rate for .mat files ---
    sr_override = args.sr
    if sr_override <= 0 and (args.input_format == 'mat' or (
        args.input_format == 'auto' and _detect_input_format(args.input_dir) == 'mat')):
        sr_override = 12000
        print(f'[INFO] Auto‑detected .mat input; using sr = {sr_override} Hz '
              f'(override with --sr)')

    print('═' * 60)
    print('AW-DPCNN Unified Dataset Builder (v2)')
    print('═' * 60)
    print(f'  Input           : {args.input_dir}')
    print(f'  Output          : {args.output_dir}')
    print(f'  Input format    : {args.input_format}')
    print(f'  Sample rate     : {sr_override if sr_override else "from file"}')
    print(f'  Window / Hop    : {args.win_len} / {args.hop_len}')
    print(f'  Image size      : {args.img_size}')
    print(f'  Colormap        : {args.cmap}')
    print(f'  PCNN iter / γ   : {args.n_iter} / {args.gamma}')
    print(f'  File split      : {args.file_split if args.file_split else "none"}')
    print(f'  Metadata        : {args.metadata}')
    print(f'  Workers         : {args.workers}')
    print(f'  Save intermed.  : {args.save_intermediates}')
    print(f'  Overwrite       : {args.overwrite}')
    print('═' * 60)

    tasks, metadata_rows = _collect_tasks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        win_len=args.win_len,
        hop_len=args.hop_len,
        img_size=args.img_size,
        n_iter=args.n_iter,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.fmax,
        cmap=cmap_code,
        gaf_method=args.gaf_method,
        save_intermediates=args.save_intermediates,
        overwrite=args.overwrite,
        sr_override=sr_override,
        input_format=args.input_format,
        file_split_ratios=file_split_ratios,
        split_seed=args.split_seed,
        gamma=args.gamma,
    )

    print(f'\n[INFO] Total fusion tasks: {len(tasks)}')
    if not tasks:
        print('[INFO] Nothing to do.')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_one_window, tasks, chunksize=8),
            total=len(tasks),
            desc='Building fused dataset',
            ncols=100,
        ))

    ok = sum(results)
    print(f'\n✅ Done — {ok}/{len(tasks)} images written to {args.output_dir}')

    # --- Save metadata ---
    if args.metadata and metadata_rows:
        csv_path = os.path.join(args.output_dir, 'metadata.csv')
        _save_metadata(csv_path, metadata_rows)
        print(f'📋 Metadata saved to {csv_path} ({len(metadata_rows)} rows)')

    # --- Distribution verification ---
    if args.verify and file_split_ratios:
        print('\n' + '─' * 60)
        print('Distribution verification')
        print('─' * 60)
        split_counts = {}
        for row in metadata_rows:
            s = row['split']
            c = row['class_label']
            split_counts.setdefault(s, {}).setdefault(c, 0)
            split_counts[s][c] += 1

        for split_name in ['train', 'val', 'test']:
            counts = split_counts.get(split_name, {})
            total = sum(counts.values())
            print(f'  {split_name}: {total} samples, '
                  f'classes: {dict(sorted(counts.items()))}')

        splits_present = [s for s in ['train', 'val', 'test'] if s in split_counts]
        for i in range(len(splits_present)):
            for j in range(i + 1, len(splits_present)):
                sa, sb = splits_present[i], splits_present[j]
                js = _compute_js_divergence(split_counts[sa], split_counts[sb])
                print(f'  JS({sa}, {sb}) = {js:.6f}')
        print('─' * 60)


if __name__ == '__main__':
    main()
