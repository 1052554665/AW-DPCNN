#!/usr/bin/env python3
"""
Build AW-DPCNN Fused Dataset — Transformer Fault Diagnosis
===========================================================
Reads **pre‑split** .wav files from::

    raw-data/transformer/{train,val,test}/{ClassName}/*.wav

and produces AW-DPCNN fused images in::

    datasets/transformer/{train,val,test}/{ClassName}/*.png

The pipeline for each sliding window is:
  1. Mel spectrogram (pseudo‑colour via colormap)
  2. GADF image (pseudo‑colour via colormap)
  3. AW‑DPCNN fusion (γ = 4, N = 20 — consistent with the paper)

No file‑level splitting is performed — the existing train / val / test
partition is honoured exactly.

Usage::

    # Default parameters (recommended for transformer 44.1 kHz audio)
    python scripts/build_transformer_dataset.py

    # Custom parameters
    python scripts/build_transformer_dataset.py \
        --win-len 4096 --hop-len 1024 --img-size 224 \
        --n-fft 2048 --n-mels 128 --fmax 8000 \
        --workers 16 --metadata

Output structure (ImageFolder‑compatible)::

    datasets/transformer/
        train/
            Loosen/  10kvOverload/  Normal/  ...
        val/
            Loosen/  10kvOverload/  Normal/  ...
        test/
            Loosen/  10kvOverload/  Normal/  ...
        metadata.csv          # if --metadata
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cv2
import numpy as np
from pyts.image import GramianAngularField
from scipy.io import wavfile
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  AW‑DPCNN core  (same algorithm as `build_fused_dataset.py`)
# ═══════════════════════════════════════════════════════════════════════

def aw_dpcnn_single_channel(S1: np.ndarray, S2: np.ndarray,
                            n_iter: int = 20,
                            gamma: float = 4.0) -> np.ndarray:
    """Fuse two single‑channel images with AW‑DPCNN.

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
    """Fuse two BGR images channel‑wise with AW‑DPCNN."""
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
#  Mel spectrogram
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
    """Compute Mel spectrogram and return a BGR pseudo‑colour image (H, W, 3)."""
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
#  GADF image
# ═══════════════════════════════════════════════════════════════════════

def generate_gadf_image(signal: np.ndarray,
                        img_size: int = 224,
                        method: str = 'difference',
                        cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Compute GADF and return a BGR pseudo‑colour image (H, W, 3)."""
    signal = signal.astype(np.float32)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    signal = signal * 2.0 - 1.0
    signal = np.clip(signal, -1.0, 1.0)

    gadf_trans = GramianAngularField(image_size=img_size, method=method)
    gadf = gadf_trans.fit_transform(signal.reshape(1, -1))[0]

    gadf_norm = (gadf - gadf.min()) / (gadf.max() - gadf.min() + 1e-8)
    gadf_uint8 = (gadf_norm * 255).astype(np.uint8)
    return _gray_to_pseudo(gadf_uint8, cmap)


# ═══════════════════════════════════════════════════════════════════════
#  Worker function  (picklable → multiprocessing)
# ═══════════════════════════════════════════════════════════════════════

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
#  Task collector
# ═══════════════════════════════════════════════════════════════════════

def collect_tasks(
    input_root: str,
    output_root: str,
    splits: list,
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
    gamma: float,
) -> tuple:
    """Walk the pre‑split input and build a flat list of window tasks.

    Returns
    -------
    (tasks, metadata_rows)
        tasks          – list of tuples for ``process_one_window``
        metadata_rows  – list of dicts for ``metadata.csv``
    """
    tasks = []
    metadata_rows = []

    for split_name in splits:
        split_input_dir = os.path.join(input_root, split_name)
        if not os.path.isdir(split_input_dir):
            print(f'[WARN] Split directory not found: {split_input_dir}')
            continue

        for cls_name in sorted(os.listdir(split_input_dir)):
            cls_input_dir = os.path.join(split_input_dir, cls_name)
            if not os.path.isdir(cls_input_dir):
                continue

            wav_files = sorted([
                f for f in os.listdir(cls_input_dir)
                if f.lower().endswith('.wav')
            ])

            for fname in wav_files:
                wav_path = os.path.join(cls_input_dir, fname)
                try:
                    sr, data = wavfile.read(wav_path)
                except Exception as exc:
                    print(f'[WARN] Cannot read {wav_path}: {exc}')
                    continue

                # Convert to mono if multi‑channel
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)

                prefix = os.path.splitext(fname)[0]
                out_cls_dir = os.path.join(output_root, split_name, cls_name)

                # --- Sliding window ---
                if win_len <= 0 or win_len >= len(data):
                    # One image per file
                    out_path = os.path.join(out_cls_dir, f'{prefix}.png')
                    if not overwrite and os.path.exists(out_path):
                        continue
                    tasks.append((
                        data, sr, out_path, img_size, n_iter,
                        n_fft, hop_length, n_mels, fmax, cmap,
                        gaf_method, save_intermediates, gamma,
                    ))
                    metadata_rows.append({
                        'filename': f'{prefix}.png',
                        'class_label': cls_name,
                        'source_file': fname,
                        'split': split_name,
                        'window_idx': 0,
                        'window_start_sample': 0,
                    })
                else:
                    idx = 0
                    for start in range(0, len(data) - win_len + 1, hop_len):
                        window = data[start:start + win_len]
                        out_path = os.path.join(
                            out_cls_dir, f'{prefix}_{idx:05d}.png')
                        if not overwrite and os.path.exists(out_path):
                            idx += 1
                            continue
                        tasks.append((
                            window, sr, out_path, img_size, n_iter,
                            n_fft, hop_length, n_mels, fmax, cmap,
                            gaf_method, save_intermediates, gamma,
                        ))
                        metadata_rows.append({
                            'filename': f'{prefix}_{idx:05d}.png',
                            'class_label': cls_name,
                            'source_file': fname,
                            'split': split_name,
                            'window_idx': idx,
                            'window_start_sample': idx * hop_len,
                        })
                        idx += 1

    return tasks, metadata_rows


# ═══════════════════════════════════════════════════════════════════════
#  Distribution verification
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
    """Write metadata.csv."""
    fieldnames = ['filename', 'class_label', 'source_file', 'split',
                  'window_idx', 'window_start_sample']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        description='AW‑DPCNN fused dataset builder — Transformer dataset')

    # --- I/O ---
    p.add_argument('--input-dir', default='raw-data/transformer',
                   help='Root of the pre‑split transformer dataset '
                        '(default: raw-data/transformer)')
    p.add_argument('--output-dir', default='datasets/transformer',
                   help='Output root for fused PNG images '
                        '(default: datasets/transformer)')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                   help='Which split subdirectories to process '
                        '(default: train val test)')

    # --- Sliding window ---
    p.add_argument('--win-len', type=int, default=4096,
                   help='Sliding window length in samples '
                        '(default: 4096 ≈ 93 ms @ 44.1 kHz)')
    p.add_argument('--hop-len', type=int, default=1024,
                   help='Hop length between windows in samples (default: 1024)')

    # --- Image ---
    p.add_argument('--img-size', type=int, default=224,
                   help='Output image size — square (default: 224)')
    p.add_argument('--cmap', default='viridis',
                   help='Colormap for Mel & GAF pseudo‑colour (default: viridis)')

    # --- Mel parameters ---
    p.add_argument('--n-fft', type=int, default=2048)
    p.add_argument('--hop-length', type=int, default=256,
                   help='STFT hop length for Mel spectrogram (default: 256)')
    p.add_argument('--n-mels', type=int, default=128)
    p.add_argument('--fmax', type=int, default=8000,
                   help='Maximum frequency for Mel scale (default: 8000 Hz)')

    # --- GADF parameters ---
    p.add_argument('--gaf-method', default='difference',
                   choices=['difference', 'summation'])

    # --- AW‑DPCNN parameters ---
    p.add_argument('--n-iter', type=int, default=20,
                   help='PCNN iterations — paper: N = 20 (default: 20)')
    p.add_argument('--gamma', type=float, default=4.0,
                   help='Contrast amplification factor — paper: γ = 4 (default: 4.0)')

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

    print('═' * 60)
    print('AW‑DPCNN Fused Dataset Builder — Transformer')
    print('═' * 60)
    print(f'  Input           : {args.input_dir}')
    print(f'  Output          : {args.output_dir}')
    print(f'  Splits          : {args.splits}')
    print(f'  Window / Hop    : {args.win_len} / {args.hop_len}')
    print(f'  Image size      : {args.img_size}')
    print(f'  Colormap        : {args.cmap}')
    print(f'  Mel: n_fft      : {args.n_fft}')
    print(f'  Mel: n_mels     : {args.n_mels}')
    print(f'  Mel: fmax       : {args.fmax} Hz')
    print(f'  GAF method      : {args.gaf_method}')
    print(f'  PCNN iter / γ   : {args.n_iter} / {args.gamma}')
    print(f'  Workers         : {args.workers}')
    print(f'  Metadata        : {args.metadata}')
    print(f'  Save intermed.  : {args.save_intermediates}')
    print(f'  Overwrite       : {args.overwrite}')
    print('═' * 60)

    tasks, metadata_rows = collect_tasks(
        input_root=args.input_dir,
        output_root=args.output_dir,
        splits=args.splits,
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
    if args.verify and metadata_rows:
        print('\n' + '─' * 60)
        print('Distribution verification')
        print('─' * 60)
        split_counts = {}
        for row in metadata_rows:
            s = row['split']
            c = row['class_label']
            split_counts.setdefault(s, {}).setdefault(c, 0)
            split_counts[s][c] += 1

        for split_name in args.splits:
            counts = split_counts.get(split_name, {})
            total = sum(counts.values())
            print(f'  {split_name}: {total} samples, '
                  f'classes: {dict(sorted(counts.items()))}')

        splits_present = [s for s in args.splits if s in split_counts]
        for i in range(len(splits_present)):
            for j in range(i + 1, len(splits_present)):
                sa, sb = splits_present[i], splits_present[j]
                js = _compute_js_divergence(split_counts[sa], split_counts[sb])
                print(f'  JS({sa}, {sb}) = {js:.6f}')
        print('─' * 60)


if __name__ == '__main__':
    main()
