import argparse
import warnings

import torch

from src.trainers.workflow import train_and_evaluate
from src.utils.config import load_config, save_yaml
from src.utils.experiment import dump_json, prepare_run_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training entrypoint for AW-DPCNN experiments.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config YAML path")
    parser.add_argument("--exp-config", default="", help="Experiment override YAML path")
    parser.add_argument("--device", default="", help="Override device (e.g. cuda, cuda:0, cpu)")
    return parser.parse_args()


def resolve_device(config_device: str, cli_device: str) -> torch.device:
    def _cuda_available_safely() -> bool:
        # Some environments expose CUDA runtime mismatches as warnings during probing.
        caught_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                available = torch.cuda.is_available()
            except Exception as exc:
                print(f"[WARN] CUDA probe failed ({exc}); falling back to CPU.")
                return False
            caught_warnings = list(caught)

        for warn in caught_warnings:
            message = str(warn.message)
            if "cudaGetDeviceCount" in message or "forward compatibility" in message:
                print(f"[WARN] CUDA probe failed ({message}); falling back to CPU.")
                return False
            warnings.warn(str(warn.message), warn.category)

        return available

    requested = cli_device.strip() or str(config_device).strip()
    if not requested or requested.lower() == "auto":
        return torch.device("cuda" if _cuda_available_safely() else "cpu")

    if requested.startswith("cuda") and not _cuda_available_safely():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return torch.device(requested)


def main():
    args = parse_args()
    config = load_config(args.config, args.exp_config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"), args.device)

    run_dir = prepare_run_dir(config)
    save_yaml(str(run_dir / "resolved_config.yaml"), config)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    results = train_and_evaluate(config, run_dir, device)
    dump_json(run_dir / "results" / "test_metrics.json", results)

    print("==== Final Test Metrics ====")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        elif isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

