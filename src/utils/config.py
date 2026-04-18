from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str) -> Dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {yaml_path} must be a YAML mapping.")
    return data


def load_config(base_config_path: str, exp_config_path: str = "") -> Dict[str, Any]:
    base = load_yaml(base_config_path)
    if exp_config_path:
        exp = load_yaml(exp_config_path)
        return _deep_update(base, exp)
    return base


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=False)

