# Training Surrogates

`run_training.py` converts your benchmark config into a concrete list of training jobs, executes them sequentially or in parallel, and saves everything under `trained/<training_id>/`. All surrogates share the same pipeline: they learn trajectories represented at fixed timesteps, but the models learn to predict the state at any continuous time within the training horizon given an initial condition (and optional physical parameters).

## From config to task list

1. **Ordered surrogates** — the entries under `surrogates`, `batch_size`, and `epochs` must align. The `i`‑th surrogate uses `batch_size[i]` and `epochs[i]`.
2. **Main run + modalities** — for each surrogate, CODES always schedules a `"main"` task. Modalities (`interpolation`, `extrapolation`, `sparse`, `batch_scaling`, `uncertainty`) add additional tasks: one per value you list, per surrogate. For example, enabling interpolation with 4 intervals on 2 surrogates yields `2 (surrogates) × 4 (intervals)` extra trainings. Modalities act as stress tests (scaling curves) rather than new architectures.
3. **Task persistence** — tasks are written to `trained/<training_id>/task_list.json`. Completed tasks are removed from this file. If you interrupt and re-run `run_training.py`, remaining tasks resume automatically.
4. **Config snapshot** — the script copies your `config.yaml` into `trained/<training_id>/config.yaml`. Evaluation uses this snapshot to verify compatibility.

## Execution model

```bash
python run_training.py --config config.yaml
```

- **Devices** — `run_training.py` reads `config.devices`. If you pass a single entry (e.g., `["cuda:0"]` or `"cpu"`), tasks run sequentially. Multiple entries trigger a thread per device; each worker pops tasks from a queue and calls `train_and_save_model`.
- **Determinism** — seeds are derived from the base `seed` plus the modality value so ensemble members differ but remain reproducible.
- **Data loading** — the script downloads/preprocesses the requested dataset once per run and reuses logarithmic/normalization settings from the config.
- **Checkpointing** — setting `checkpoint: true` stores intermediate checkpoints via the shared `validate` logic. Otherwise only the final model (per task) is saved as `<surrogate>_<mode>_<metric>.pth` under `trained/<training_id>/`.
- **Failure handling** — failed tasks stay in `task_list.json`. Fix the root cause and rerun to continue. When all tasks finish successfully, the task list is deleted and `completed.txt` is written.

## Modalities vs. evaluation switches

Modalities directly increase the number of models to train. See :doc:`modalities` for the available studies and their impact. Evaluation switches (`losses`, `gradients`, `timing`, `compute`, `compare`, `iterative`) do **not** create extra training tasks—they only control which analyses run later (details in :doc:`evaluation`). Plan your training duration around the modalities you enable.

## Practical tips

- Profiling new configs? Lower `epochs`/`batch_size` temporarily, or set `dataset.subset_factor` to work on fewer trajectories.
- Keep `training_id` unique per experiment so you can compare checkpoints across branches.
- If you plan to resume frequently, commit the generated `trained/<training_id>/config.yaml` alongside your original config for traceability.
