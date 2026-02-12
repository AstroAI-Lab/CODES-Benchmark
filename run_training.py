import os
from argparse import ArgumentParser
from datetime import timedelta

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from tqdm import tqdm

from codes.train import (
    create_task_list_for_surrogate,
    parallel_training,
    sequential_training,
)
from codes.utils import (
    check_training_status,
    load_and_save_config,
    load_task_list,
    nice_print,
    save_task_list,
)
from codes.utils.data_utils import download_data


def main(args):
    """
    Main function to train the models. If the training is already completed, it will
    print a message and exit. Otherwise, it will create a task list for each surrogate
    model and train the models sequentially or in parallel depending on the number of
    devices.

    Args:
        args (Namespace): The command line arguments.
    """
    torch.use_deterministic_algorithms(True)
    config = load_and_save_config(config_path=args.config, save=False)

    if torch.cuda.is_available() and config.get("verbose", False):
        nice_print(
            f"CUDA devices detected: count={torch.cuda.device_count()}, current={torch.cuda.current_device()}"
        )

    download_data(config["dataset"]["name"], verbose=config.get("verbose", False))
    task_list_filepath, copy_config = check_training_status(config)
    if copy_config:
        config = load_and_save_config(config_path=args.config, save=True)
    else:
        previous_config_path = f"trained/{config['training_id']}/config.yaml"
        config = load_and_save_config(config_path=previous_config_path, save=False)
    tasks = load_task_list(task_list_filepath)

    if not tasks:
        tasks = []

        nice_print("Starting training")

        for surr_name in config["surrogates"]:
            tasks += create_task_list_for_surrogate(config, surr_name)

        save_task_list(tasks, task_list_filepath)

    device_list = config["devices"]
    device_list = [device_list] if isinstance(device_list, str) else device_list
    if len(device_list) > 1:
        tqdm.write(f"Training models in parallel on devices  : {device_list} \n")
        elapsed_time = parallel_training(tasks, device_list, task_list_filepath)
    else:
        tqdm.write(f"Training models sequentially on device {device_list[0]}")
        elapsed_time = sequential_training(tasks, device_list, task_list_filepath)

    print("\n")
    nice_print("Training completed")

    print(f"{len(tasks)} Models saved in /trained/{config['training_id']}/")
    print(f"Total training time: {timedelta(seconds=int(elapsed_time))} \n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)
