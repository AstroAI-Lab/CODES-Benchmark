# Getting Started

The smoke examples train a small FullyConnected model on Lotka–Volterra trajectories and tune one hyperparameter over four trials. Both run on CPU and use a subset of the training trajectories.

## Installation

From the root of your clone, use [uv](https://docs.astral.sh/uv/) and the project's lockfile. The repo pins Python 3.11 for uv. If your site manages Python through environment modules, load a supported Python before syncing.

```bash
cd CODES-Benchmark
uv sync --locked
uv run python -c 'import codes; print("CODES ready")'
```

If you use pip, create an environment with Python 3.11 or newer and install with `python -m pip install -e .`. In that activated environment, use `python` in place of `uv run` for the Python scripts below.

(stage-dataset)=
## Download the dataset

```bash
uv run python -c 'from codes import download_data; download_data("lotka_volterra")'
```

Run this from the repository root on a host that can reach Zenodo. It caches the 4.9 MB [Lotka–Volterra dataset](https://zenodo.org/records/14703867) at `datasets/lotka_volterra/data.hdf5`. If the cluster blocks outbound access, download `data.hdf5` from the same record and copy it to that path on the shared filesystem. See the [dataset catalog](reference/datasets.md) for other options.

## First run on one machine

```bash
uv run run_training.py --config configs/train_eval/config_smoke.yaml
uv run run_eval.py --config configs/train_eval/config_smoke.yaml
uv run run_tuning.py --config configs/tuning/config_smoke.yaml
```

The training example fits two one-epoch models under `trained/codes_smoke_training/`. Evaluation writes metrics under `results/codes_smoke_training/`. Tuning writes its Optuna study and trial checkpoints under `tuned/codes_smoke_tuning/`.

(first-run-on-slurm)=
## First run on Slurm

Clone and sync the environment on a filesystem shared with the compute nodes. The checkout, Python interpreter and its dependencies, staged dataset, and output directories must be readable at the same paths from the submit host and the workers. Workers need write access to the run directories. Submit from the repository root on the cluster's submission host.

Copy the smoke configs and add your site's scheduler values under `slurm` in each copy. The examples already request CPU, one thread, 2 GB of memory, and ten minutes per task. Check which `partition`, `account`, or `qos` your site requires. For these CPU examples, use a CPU-capable partition.

```bash
cp configs/train_eval/config_smoke.yaml train-smoke.yaml
cp configs/tuning/config_smoke.yaml tune-smoke.yaml
```

If you already ran the local commands, change `training_id` and `tuning_id` in the copies before submitting. Use a new ID whenever you want an independent run.

```bash
uv run run_training.py --slurm submit --config train-smoke.yaml
uv run run_tuning.py --slurm submit --config tune-smoke.yaml
```

Each command prints its Slurm job ID and manifest path. The training array has two assignments and the tuning array starts two workers that share one Optuna journal. Check Slurm with `squeue -j JOB_ID` while an array runs and inspect `slurm-logs/` beside the manifest for worker output. After the jobs end, check the application results.

```bash
uv run run_training.py --slurm status --config train-smoke.yaml
uv run run_tuning.py --slurm status --config tune-smoke.yaml
```

Training should report `Completed: 2/2`. Tuning should report at least four `valid` trials. If workers fail, start with their Slurm logs. Check that the dataset and interpreter paths resolve on the workers. If your site requires environment modules, load them from a {ref}`launcher script <slurm-submit-config>`. Re-submit the same config once the earlier array has stopped to continue an unfinished run.

See the [training guide](guides/running-benchmarks/training.md) for GPU resources and the [tuning guide](guides/running-benchmarks/tuning.md) for storage options, including an existing PostgreSQL service. For a larger run, copy `configs/train_eval/config_minimal.yaml` or `configs/train_eval/config_full.yaml` and choose the appropriate dataset, models, and devices. The [configuration reference](reference/configuration.md) lists the available settings.
