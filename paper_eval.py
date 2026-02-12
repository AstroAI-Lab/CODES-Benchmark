"""
Convenience script to re-run paper evaluations across multiple training runs.

Behavior:
- For each specified training_id, load its saved config from trained/<training_id>/config.yaml
- Override the devices in that config with the hardcoded DEVICE below
- Run the same evaluation flow as run_eval.py using that modified config

Note: This script intentionally does NOT read the top-level config.yaml.
"""

import os
from typing import Dict

from codes.benchmark import (
    check_benchmark,
    check_surrogate,
    compare_models,
    get_surrogate,
    run_benchmark,
)
from codes.utils import download_data, nice_print, read_yaml_config

# Hardcoded device override for all evaluations
DEVICE = "cuda:6"

# Training IDs to evaluate
TRAINING_IDS = [
    "_cloud_finetuned",
    "_cloud_parametric_finetuned",
    "_primordial_finetuned",
    "_primordial_parametric_finetuned",
]


def evaluate_with_config(config: Dict) -> None:
    """Run the evaluation loop for a single configuration dict."""
    # Basic checks and data
    check_benchmark(config)
    download_data(config["dataset"]["name"], verbose=config.get("verbose", False))

    surrogates = config["surrogates"]
    all_metrics = {surrogate: {} for surrogate in surrogates}

    # Evaluate each surrogate
    for surrogate_name in surrogates:
        surrogate_class = get_surrogate(surrogate_name)
        if surrogate_class is None:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")
            continue

        nice_print(f"Running benchmark for {surrogate_name}")
        check_surrogate(surrogate_name, config)
        metrics = run_benchmark(surrogate_name, surrogate_class, config)
        all_metrics[surrogate_name] = metrics

    # Compare models if requested
    if config.get("compare", False):
        if len(surrogates) < 2:
            nice_print("At least two surrogate models are required to compare.")
        else:
            nice_print("Comparing models")
            compare_models(all_metrics, config)


def load_trained_config(training_id: str) -> Dict | None:
    """Load the saved config for a given training_id from trained/<id>/config.yaml."""
    cfg_path = os.path.join("trained", training_id, "config.yaml")
    if not os.path.exists(cfg_path):
        print(
            f"Config not found for training_id '{training_id}': {cfg_path}. Skipping."
        )
        return None
    config = read_yaml_config(cfg_path)
    return config


def main():
    for tid in TRAINING_IDS:
        nice_print(f"Evaluating {tid}")

        config = load_trained_config(tid)
        if config is None:
            continue

        # Override devices with the hardcoded DEVICE
        config["devices"] = [DEVICE]

        try:
            evaluate_with_config(config)
        except Exception as e:
            print(f"Evaluation failed for {tid}: {e}")
            # Continue with the next training_id
            continue

    nice_print("All requested evaluations processed")


if __name__ == "__main__":
    main()
