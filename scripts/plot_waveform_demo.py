#!/usr/bin/env python3
"""
Plot waveform samples for all 10 CWRU 12k DE bearing fault classes.

Output: paper/waveform_demo.png  (10 subplots, one per class)

usage::
    python scripts/plot_waveform_demo.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.io import loadmat

CWRU_ROOT = Path("raw-data/CWRU-dataset")

CLASSES = [
    ("Normal", "Normal", "DE_time"),
    ("BF007", "12k_Drive_End_Bearing_Fault_Data/B/007", "DE_time"),
    ("BF014", "12k_Drive_End_Bearing_Fault_Data/B/014", "DE_time"),
    ("BF021", "12k_Drive_End_Bearing_Fault_Data/B/021", "DE_time"),
    ("IF007", "12k_Drive_End_Bearing_Fault_Data/IR/007", "DE_time"),
    ("IF014", "12k_Drive_End_Bearing_Fault_Data/IR/014", "DE_time"),
    ("IF021", "12k_Drive_End_Bearing_Fault_Data/IR/021", "DE_time"),
    ("OF007", "12k_Drive_End_Bearing_Fault_Data/OR/007/@6", "DE_time"),
    ("OF014", "12k_Drive_End_Bearing_Fault_Data/OR/014", "DE_time"),
    ("OF021", "12k_Drive_End_Bearing_Fault_Data/OR/021/@6", "DE_time"),
]

SR = 12000          # sample rate
N_SAMPLES = 2048    # plot first N samples


def load_first_signal(mat_dir: Path, sensor_key: str) -> np.ndarray:
    """Load the first .mat file and extract the sensor signal."""
    files = sorted(mat_dir.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files in {mat_dir}")
    mat = loadmat(str(files[0]))
    for key in mat.keys():
        if sensor_key in key:
            sig = mat[key].squeeze().astype(np.float32)
            if sig.ndim != 1:
                sig = sig.ravel()
            return sig[:N_SAMPLES]
    raise KeyError(f"No '{sensor_key}' in {files[0]}")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 300,
    })

    out_dir = Path("paper/waveforms")
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, subdir, sensor in CLASSES:
        mat_dir = CWRU_ROOT / subdir
        if not mat_dir.exists():
            print(f"[WARN] {label}: dir not found: {mat_dir}")
            continue

        try:
            sig = load_first_signal(mat_dir, sensor)
        except Exception as e:
            print(f"[WARN] {label}: {e}")
            continue

        t = np.arange(len(sig)) / SR * 1000  # ms

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, sig, linewidth=0.5, color="steelblue")
        # ax.set_title(f"{label} — CWRU 12k Drive‑End")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        out_path = out_dir / f"{label}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
