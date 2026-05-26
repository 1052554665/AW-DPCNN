import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all exp1 YAML configs sequentially using scripts/train.py."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Base config YAML path",
    )
    parser.add_argument(
        "--exp-dir",
        default="experiments/exp1",
        help="Directory containing experiment YAML files",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern for experiment YAML files",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override device (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs when a run fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands without executing",
    )
    return parser.parse_args()


def list_experiment_configs(exp_dir: str, pattern: str) -> List[Path]:
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")
    configs = sorted([p for p in exp_path.glob(pattern) if p.is_file()])
    if not configs:
        raise FileNotFoundError(
            f"No experiment configs found in {exp_path} with pattern {pattern}"
        )
    return configs


def build_command(base_config: str, exp_config: Path, device: str) -> List[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        base_config,
        "--exp-config",
        str(exp_config),
    ]
    if device:
        command.extend(["--device", device])
    return command


def main() -> int:
    args = parse_args()
    configs = list_experiment_configs(args.exp_dir, args.pattern)

    print(f"Found {len(configs)} configs under {args.exp_dir}")
    for index, exp_config in enumerate(configs, start=1):
        command = build_command(args.config, exp_config, args.device)
        print(f"[{index}/{len(configs)}] {exp_config}")
        print("Command:", " ".join(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            message = f"Run failed for {exp_config} (exit code {result.returncode})"
            if args.continue_on_error:
                print(f"[WARN] {message}; continuing")
                continue
            print(f"[ERROR] {message}")
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

