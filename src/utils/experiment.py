import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for all libraries and configure deterministic mode.

    Args:
        seed: Integer seed for reproducibility.
        deterministic: If True, enable cudnn deterministic mode (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Ensure deterministic algorithms are used where available (PyTorch >= 1.9)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (TypeError, RuntimeError):
                pass  # Older PyTorch or unsupported ops


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_run_dir(config: Dict) -> Path:
    """Create the run directory and return its path.

    Directory layout::

        {root_dir}/{model_dir}/{dataset}/{experiment_name}/trial_seed{seed}

    Where *model_dir* is the stem of the experiment config filename
    (e.g. ``MSCA_VGG16``), and *dataset* is the short dataset key
    (e.g. ``12k_de``).  Both are stored in *config* by the caller
    (``scripts/train.py``).  When omitted, the layout falls back to
    ``{root_dir}/{experiment_name}/trial_seed{seed}``.
    """
    output_cfg = config.get("output", {})
    root = Path(output_cfg.get("root_dir", "experiments/runs")).resolve()
    exp_name = str(config.get("experiment_name", "experiment"))
    model_dir = config.get("model_dir", "")
    dataset_key = config.get("dataset", {}).get("key", "")

    # Build the path components — skip empty levels for backwards
    # compatibility (e.g. single-shot runs without --dataset).
    parts = [root]
    if model_dir:
        parts.append(model_dir)
    if dataset_key:
        parts.append(dataset_key)
    parts.append(exp_name)

    seed = config.get("seed", None)
    if seed is not None:
        run_dir = Path(*parts) / f"trial_seed{seed}"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(*parts).with_name(f"{exp_name}_{ts}")

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)

