#!/usr/bin/env python3
"""
Unified AW-DPCNN Dataset Builder
=================================
Single-script pipeline that combines the three stages:
  1. Mel spectrogram generation  (originally `mel.py`)
  2. GADF image generation       (originally `GAF.py`)
  3. AW-DPCNN fusion             (originally `awdpcnn.py`)

All three stages run **in memory per window** — no intermediate PNG files are
written unless `--save-intermediates` is requested.  This guarantees that every
fused image is built from *exactly the same signal segment*, eliminating the
index‑matching fragility of the original three‑script workflow.

Usage (batch mode, respecting class sub‑folders)::

    python scripts/build_fused_dataset.py \
        --input-dir  ./raw_wavs/train \
        --output-dir ./datasets/train \
        --win-len 3000 --hop-len 750 --img-size 224 \
        --n-iter 8 --workers 16

Output structure (ImageFolder‑compatible)::

    datasets/
      train/
        Normal/
          Normal_00000.png
          ...
        PartialDischarge/
          ...

If the WAV files are already pre‑segmented and you only want one image per WAV
(no sliding window), set `--win-len` to 0::

    python scripts/build_fused_dataset.py \
        --input-dir ./pre_segmented/train \
        --output-dir ./datasets/train \
        --win-len 0 --img-size 224
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pyts.image import GramianAngularField
from scipy.io import wavfile
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
#  AW-DPCNN core  (extracted from `awdpcnn.py`)
# ═══════════════════════════════════════════════════════════════════════

def aw_dpcnn_single_channel(S1: np.ndarray, S2: np.ndarray,
                            n_iter: int = 8) -> np.ndarray:
    """Fuse two single-channel images with AW-DPCNN."""
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

    gamma = 10
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
                          n_iter: int = 8) -> np.ndarray:
    """Fuse two BGR images channel‑wise with AW-DPCNN."""
    if mel_img.shape[:2] != gaf_img.shape[:2]:
        gaf_img = cv2.resize(gaf_img, (mel_img.shape[1], mel_img.shape[0]))

    mel_ch = cv2.split(mel_img)
    gaf_ch = cv2.split(gaf_img)

    fused = []
    for i in range(3):
        f = aw_dpcnn_single_channel(mel_ch[i], gaf_ch[i], n_iter)
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
#  Single‑window processor  (the heart of the unified pipeline)
# ═══════════════════════════════════════════════════════════════════════

def process_one_window(args: tuple) -> int:
    """Generate Mel + GADF from one signal window, fuse, and save.

    Returns 1 on success, 0 on failure.
    """
    (signal, sr, out_path, img_size, n_iter, n_fft, hop_length,
     n_mels, fmax, cmap, gaf_method, save_intermediates) = args

    try:
        mel_img = generate_mel_image(
            signal, sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmax=fmax, img_size=img_size, cmap=cmap,
        )
        gadf_img = generate_gadf_image(
            signal, img_size=img_size, method=gaf_method, cmap=cmap,
        )
        fused = aw_dpcnn_fusion_color(mel_img, gadf_img, n_iter=n_iter)

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
#  WAV‑level dispatcher
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
) -> list:
    """Walk the input directory and build a flat list of window tasks."""
    tasks = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue

            wav_path = os.path.join(root, fname)
            try:
                sr, data = wavfile.read(wav_path)
            except Exception:
                print(f'[WARN] Cannot read {wav_path}, skipping.')
                continue

            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)

            # Determine relative class path to mirror folder structure
            rel_dir = os.path.relpath(root, input_dir)
            out_cls_dir = os.path.join(output_dir, rel_dir)
            prefix = os.path.splitext(fname)[0]

            if win_len <= 0 or win_len >= len(data):
                # No sliding window — one fused image per WAV
                out_path = os.path.join(out_cls_dir, prefix + '.png')
                if not overwrite and os.path.exists(out_path):
                    continue
                tasks.append((
                    data, sr, out_path, img_size, n_iter,
                    n_fft, hop_length, n_mels, fmax, cmap,
                    gaf_method, save_intermediates,
                ))
            else:
                # Sliding window
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
                        gaf_method, save_intermediates,
                    ))
                    idx += 1

    return tasks


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
        description='Unified AW-DPCNN fused dataset builder')

    # --- I/O ---
    p.add_argument('--input-dir', required=True,
                   help='Root directory of WAV files (class sub‑folders)')
    p.add_argument('--output-dir', required=True,
                   help='Root directory for fused PNG images')

    # --- Sliding window ---
    p.add_argument('--win-len', type=int, default=3000,
                   help='Sliding window length in samples (0 = one image per WAV)')
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
    p.add_argument('--n-iter', type=int, default=8,
                   help='PCNN iterations')

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
    print('AW-DPCNN Unified Dataset Builder')
    print('═' * 60)
    print(f'  Input          : {args.input_dir}')
    print(f'  Output         : {args.output_dir}')
    print(f'  Window / Hop   : {args.win_len} / {args.hop_len}')
    print(f'  Image size     : {args.img_size}')
    print(f'  Colormap       : {args.cmap}')
    print(f'  PCNN iter      : {args.n_iter}')
    print(f'  Workers        : {args.workers}')
    print(f'  Save intermed. : {args.save_intermediates}')
    print(f'  Overwrite      : {args.overwrite}')
    print('═' * 60)

    tasks = _collect_tasks(
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


if __name__ == '__main__':
    main()
