"""
Shared CWRU dataset registry.

Provides a single source of truth for all three CWRU sub‑datasets:
    - 12k_de  : 12 kHz Drive‑End (DE_time)
    - 12k_fe  : 12 kHz Fan‑End   (FE_time)
    - 48k_de  : 48 kHz Drive‑End (DE_time)

Usage::

    from src.utils.dataset_registry import DATASET_CONFIGS, get_dataset_config

    cfg = get_dataset_config("12k_de")
    print(cfg["root_dir"])   # "datasets/cwru_de"
    print(cfg["sr"])         # 12000
"""

from collections import OrderedDict
from pathlib import Path

DATASET_CONFIGS = {
    "12k_de": {
        "name": "CWRU 12k Drive-End",
        "root_dir": "datasets/cwru_12k_de",
        "src_dir": "raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "DE_time",
        "sr": 12000,
        "fmax": 6000,
        "n_fft": 1024,
        "hop_len": 256,
        "win_len": 2048,
        "seg_hop": 1024,
        "n_mels": 128,
    },
    "12k_fe": {
        "name": "CWRU 12k Fan-End",
        "root_dir": "datasets/cwru_12k_fe",
        "src_dir": "raw-data/CWRU-dataset/12k_Fan_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "FE_time",
        "sr": 12000,
        "fmax": 6000,
        "n_fft": 1024,
        "hop_len": 256,
        "win_len": 2048,
        "seg_hop": 1024,
        "n_mels": 128,
    },
    "48k_de": {
        "name": "CWRU 48k Drive-End",
        "root_dir": "datasets/cwru_48k_de",
        "src_dir": "raw-data/CWRU-dataset/48k_Drive_End_Bearing_Fault_Data",
        "normal_dir": "raw-data/CWRU-dataset/Normal",
        "sensor_key": "DE_time",
        "sr": 48000,
        "fmax": 24000,
        "n_fft": 4096,
        "hop_len": 512,
        "win_len": 8192,
        "seg_hop": 4096,
        "n_mels": 128,
    },
}

DATASET_KEYS = list(DATASET_CONFIGS.keys())

CWRU_CLASS_NAMES = [
    "BF007", "BF014", "BF021",
    "IF007", "IF014", "IF021",
    "OF007", "OF014", "OF021",
    "Normal",
]

CWRU_CLASS_MAP = OrderedDict([
    ("BF007", ("12k_Drive_End_Bearing_Fault_Data/B/007",       "DE_time")),
    ("BF014", ("12k_Drive_End_Bearing_Fault_Data/B/014",       "DE_time")),
    ("BF021", ("12k_Drive_End_Bearing_Fault_Data/B/021",       "DE_time")),
    ("IF007", ("12k_Drive_End_Bearing_Fault_Data/IR/007",      "DE_time")),
    ("IF014", ("12k_Drive_End_Bearing_Fault_Data/IR/014",      "DE_time")),
    ("IF021", ("12k_Drive_End_Bearing_Fault_Data/IR/021",      "DE_time")),
    ("OF007", ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6",   "DE_time")),
    ("OF014", ("12k_Drive_End_Bearing_Fault_Data/OR/014",      "DE_time")),
    ("OF021", ("12k_Drive_End_Bearing_Fault_Data/OR/021/@6",   "DE_time")),
    ("Normal", ("Normal",                                       "DE_time")),
])

CWRU_ROOT = Path("raw-data/CWRU-dataset")


def get_dataset_config(dataset_key):
    """Return config dict for *dataset_key* (e.g. ``'12k_de'``)."""
    if dataset_key not in DATASET_CONFIGS:
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Valid keys: {DATASET_KEYS}"
        )
    return dict(DATASET_CONFIGS[dataset_key])
