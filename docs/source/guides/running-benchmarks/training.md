# Training Surrogates

`run_training.py` converts your benchmark config into a concrete list of training jobs, executes them sequentially or in parallel, and saves everything under `trained/<training_id>/`. All surrogates share the same pipeline: they learn trajectories represented at fixed timesteps, but the models learn to predict the state at any continuous time within the training horizon given an initial condition (and optional physical parameters).

## From config to task list

1. **Ordered surrogates** — the entries under `surrogates`, `batch_size`, and `epochs` must align. The `i`‑th surrogate uses `batch_size[i]` and `epochs[i]`.
2. **Main run + modalities** — for each surrogate, CODES always schedules a `"main"` task. Modalities (`interpolation`, `extrapolation`, `sparse`, `batch_scaling`, `uncertainty`) add additional tasks: one per value you list, per surrogate. For example, enabling interpolation with 4 intervals on 2 surrogates yields `2 (surrogates) × 4 (intervals)` extra trainings. Modalities act as stress tests (scaling curves) rather than new architectures.
3. **Task persistence** — tasks are written to `trained/<training_id>/train_tasks.json`. The parent removes a task only after its worker reports success and exits successfully. If you interrupt and re-run `run_training.py`, remaining tasks restart from their beginning; this does not resume a partially trained model.
4. **Config snapshot** — the script copies your `config.yaml` into `trained/<training_id>/config.yaml`. Evaluation uses this snapshot to verify compatibility.

## Local execution

```bash
python run_training.py --config config.yaml
```

- **Devices** — `config.devices` defines concurrent worker slots. A single entry (e.g., `["cuda:0"]` or `"cpu"`) runs one task at a time. Multiple entries allow that many concurrent tasks. Each task gets a fresh spawned process, including in sequential runs. Repeated entries deliberately share a device, for example `["cpu", "cpu"]`; allow enough CPU or GPU memory for every concurrent task.
- **CPU threads** — `cpu_threads` defaults to `1` per worker, independently of the number of slots. It controls PyTorch CPU threads and is checked against CPU affinity and `SLURM_CPUS_PER_TASK` when available. Allocate enough CPUs for the combined worker slots. Using the same setting in sequential and parallel runs avoids changing this numerical execution setting between comparisons.
- **Determinism** — each worker configures deterministic PyTorch execution and applies the existing task seed before model and loader initialization. Ensemble members retain their distinct seeds. Random draws in one worker do not affect another; custom training loops can therefore use stochastic operations such as dropout. Reproducibility still depends on the software, hardware and execution settings.
- **Data loading** — the parent ensures the dataset is downloaded before launching workers. Each worker loads and preprocesses its own task data using the saved configuration. Large datasets therefore consume memory per concurrent worker.
- **Checkpointing** — setting `checkpoint: true` stores intermediate checkpoints via the shared `validate` logic. Otherwise only the final model (per task) is saved as `<surrogate>_<mode>_<metric>.pth` under `trained/<training_id>/`.
- **Failure handling** — successful tasks are recorded even when another task fails. Failed tasks stay in `train_tasks.json`, and the command exits with an error. On interruption, the parent stops and joins its workers and retains unfinished work. When all pending tasks finish successfully, `completed.txt` is written and the task list is removed. An existing empty task list can be finalized without repeating completed work.

Run only one local coordinator for a given `training_id`. Atomic task-list replacement protects against interrupted writes; it does not coordinate several independent launchers sharing the same run.

### Progress and task logs

The parent process owns the display. Workers send epoch counts and validation
metrics instead of drawing their own bars. With `progress: auto`, an interactive
terminal shows an overall bar and one row per worker slot. Piped output (including
`tee` and the GPU smoke runner) and Slurm batch logs use timestamped text, with
periodic updates and task start/completion messages. Set `progress` to `bars`,
`plain` or `off` to override detection. Warnings and failures remain visible even
with ordinary progress disabled.

Task output and progress are saved under `trained/<training_id>/task_logs/`.
Retries append to the corresponding task log. A custom `fit` that uses
`setup_progress_bar` gets centralized epoch reporting automatically; other Python
output is forwarded to the parent and saved in the task log. Direct model fitting
and tuning retain their existing progress behavior.

## Running inside a Slurm allocation

The local runner can execute inside one Slurm job. It does not submit jobs or distribute tasks across nodes. Use the logical CUDA indices visible within the allocation, without replacing Slurm's `CUDA_VISIBLE_DEVICES`. For a single allocated GPU:

```yaml
devices: ["cuda:0"]
cpu_threads: 4
```

An illustrative submission script, after installing and activating the environment:

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail
cd /path/to/CODES-Benchmark
srun python run_training.py --config configs/train_eval/my_run.yaml
```

Adjust the resource requests, account, partition, time limit, paths and environment for your cluster. A multi-GPU allocation can use multiple visible indices in `devices`, with corresponding CPU and memory resources.

The single-task executor, `codes.train.tasks.execute_training_task`, is separate from local scheduling. It trains and saves one task without changing the parent's pending-task list or completion marker. This provides the execution boundary for future Slurm array workers; immutable manifests and independent-job coordination are not implemented yet. Tuning still uses its existing execution and storage mechanisms.

## Modalities vs. evaluation switches

Modalities directly increase the number of models to train. See :doc:`modalities` for the available studies and their impact. Evaluation switches (`losses`, `gradients`, `timing`, `compute`, `compare`, `iterative`) do **not** create extra training tasks—they only control which analyses run later (details in :doc:`evaluation`). Plan your training duration around the modalities you enable.

## Practical tips

- Profiling new configs? Lower `epochs`/`batch_size` temporarily, or set `dataset.subset_factor` to work on fewer trajectories.
- Keep `training_id` unique per experiment so you can compare checkpoints across branches.
- If you plan to resume frequently, commit the generated `trained/<training_id>/config.yaml` alongside your original config for traceability.
