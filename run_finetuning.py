import os
from argparse import ArgumentParser

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from codes.finetune import run_finetuning
from codes.utils import download_data, nice_print, read_yaml_config


def main(args):
    """
    Finetune trained surrogates for autoregressive rollout.

    Reads the same config as run_training.py, which must have been run first, and
    the finetune block in it. The finetuned models are saved as a new training run,
    so run_eval.py can evaluate them with the config written next to them.

    Args:
        args (Namespace): The command line arguments.
    """
    torch.use_deterministic_algorithms(True)
    config = read_yaml_config(args.config)
    download_data(config["dataset"]["name"], verbose=config.get("verbose", False))

    devices = config.get("devices", ["cpu"])
    devices = [devices] if isinstance(devices, str) else devices
    device = args.device or devices[0]

    run_finetuning(config, device)

    ft_id = config["finetune"].get("training_id") or f"{config['training_id']}_finetune"
    nice_print("Finetuning completed")
    print(f"Models saved in /trained/{ft_id}/, summary in /results/{ft_id}/")
    print(f"Evaluate with: python run_eval.py --config trained/{ft_id}/config.yaml\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device to finetune on. Defaults to the first entry of devices.",
    )
    args = parser.parse_args()
    main(args)
