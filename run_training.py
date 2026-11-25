import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def _run_local(args, config):
    """
    Main function to train the models. If the training is already completed, it will
    print a message and exit. Otherwise, it will create a task list for each surrogate
    model and train the models sequentially or in parallel depending on the number of
    devices.

    Args:
        args (Namespace): The command line arguments.
    """
    if torch.cuda.is_available() and config.get("verbose", False):
        nice_print(f"Visible CUDA devices: count={torch.cuda.device_count()}")

    download_data(config["dataset"]["name"], verbose=config.get("verbose", False))
    task_list_filepath, copy_config = check_training_status(config)
    if copy_config:
        config = load_and_save_config(config_path=args.config, save=True)
    else:
        previous_config_path = f"trained/{config['training_id']}/config.yaml"
        config = load_and_save_config(config_path=previous_config_path, save=False)
    tasks = load_task_list(task_list_filepath)

    if not os.path.exists(task_list_filepath):
        tasks = []

        nice_print("Starting training")

        for surr_name in config["surrogates"]:
            tasks += create_task_list_for_surrogate(config, surr_name)

        save_task_list(tasks, task_list_filepath)

    device_list = config["devices"]
    device_list = [device_list] if isinstance(device_list, str) else device_list
    if not device_list:
        raise ValueError("At least one training device is required")
    cpu_threads = config.get("cpu_threads", 1)
    if len(device_list) > 1:
        tqdm.write(f"Training models in parallel on devices  : {device_list} \n")
        elapsed_time = parallel_training(
            tasks,
            device_list,
            task_list_filepath,
            cpu_threads=cpu_threads,
            progress=config.get("progress", "auto"),
        )
    else:
        tqdm.write(f"Training models sequentially on device {device_list[0]}")
        elapsed_time = sequential_training(
            tasks,
            device_list,
            task_list_filepath,
            cpu_threads=cpu_threads,
            progress=config.get("progress", "auto"),
        )

    print("\n")
    nice_print("Training completed")

    print(f"{len(tasks)} Models saved in /trained/{config['training_id']}/")
    print(f"Total training time: {timedelta(seconds=int(elapsed_time))} \n")


def main(args):
    if getattr(args, "slurm", None):
        from codes.train.slurm import prepare_training, run_worker, training_status

        if args.slurm == "submit":
            from codes.slurm import SlurmOptions, submit_array
            from codes.train.slurm import _load_manifest, unfinished_training_indices
            from codes.utils import read_yaml_config

            config = read_yaml_config(args.config)
            options = SlurmOptions.from_config(config.get("slurm", {}))
            if config.get("cpu_threads", 1) > options.cpus_per_task:
                raise ValueError(
                    "training cpu_threads cannot exceed slurm.cpus_per_task; adjust both settings explicitly"
                )
            dataset = Path("datasets") / config["dataset"]["name"].lower() / "data.hdf5"
            if not os.path.isfile(dataset):
                raise FileNotFoundError(
                    f"Dataset must be staged before Slurm submission: {dataset}"
                )
            manifest = Path(
                args.manifest
                or Path("trained") / config["training_id"] / "manifest.json"
            ).resolve()
            prepare_training(args.config, manifest)
            frozen = _load_manifest(manifest)
            indices = unfinished_training_indices(manifest, manifest=frozen)
            if not indices:
                print("All training tasks are already complete")
                return
            job_id = submit_array(
                manifest,
                indices,
                options,
                [
                    str(Path(__file__).resolve()),
                    "--slurm",
                    "worker",
                    "--manifest",
                    str(manifest),
                    "--device",
                    options.device,
                ],
            )
            print(f"Submitted Slurm job {job_id} for manifest {manifest}")
            return
        if not args.manifest and args.slurm == "status":
            from codes.utils import read_yaml_config

            config = read_yaml_config(args.config)
            args.manifest = str(
                Path("trained") / config["training_id"] / "manifest.json"
            )
        if not args.manifest:
            raise ValueError("--manifest is required for Slurm training")
        if args.slurm == "prepare":
            prepare_training(args.config, args.manifest)
        elif args.slurm == "worker":
            run_worker(args.manifest, args.device)
        else:
            training_status(args.manifest)
        return

    from codes.train.slurm import training_lock

    config = load_and_save_config(config_path=args.config, save=False)
    with training_lock(config["training_id"]):
        run_dir = os.path.join("trained", config["training_id"])
        if any(
            os.path.exists(os.path.join(run_dir, name))
            for name in ("slurm_manifest.json", "slurm_initializing.json")
        ):
            raise RuntimeError(
                "This training run belongs to a Slurm manifest; use its workers or a new training_id"
            )
        _run_local(args, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    parser.add_argument("--slurm", choices=("prepare", "worker", "status", "submit"))
    parser.add_argument("--manifest", help="Path to the fixed Slurm training manifest")
    parser.add_argument(
        "--device", default="cpu", help="Logical device used by a Slurm worker"
    )
    args = parser.parse_args()
    main(args)
