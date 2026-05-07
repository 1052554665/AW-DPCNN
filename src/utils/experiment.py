import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_run_dir(config: Dict) -> Path:
    output_cfg = config.get("output", {})
    root = Path(output_cfg.get("root_dir", "experiments/runs")).resolve()
    exp_name = str(config.get("experiment_name", "experiment"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)

