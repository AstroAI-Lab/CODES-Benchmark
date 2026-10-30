# Evaluation & Reporting

`run_eval.py` replays your configuration against the saved checkpoints, executes whichever evaluation suites you enabled, and collates the results under `results/<training_id>/` and `plots/<training_id>/`. All accuracy metrics are reported in log-space (see :doc:`accuracy-metrics`) so the evaluations align with how models were trained and tuned.

## Launching evaluations

```bash
python run_eval.py --config config.yaml
```

- **Config fidelity** — `check_benchmark` compares the provided config to `trained/<training_id>/config.yaml`. Dataset settings, modalities, and surrogate names must match those used during training so that required checkpoints exist. You may disable modalities (e.g., skip an interpolation analysis even if it was trained) by toggling the evaluation switches, but you cannot evaluate a modality that lacks trained checkpoints.
- **Devices** — the same `devices` list is reused for evaluation. Override `CUDA_VISIBLE_DEVICES` if you want to force CPU/GPU placement without editing the config.
- **Per-surrogate loop** — for every entry in `surrogates`, `run_eval.py` loads the registered class, rehydrates the checkpoints, and calls `run_benchmark`. Missing classes or checkpoints are flagged early via `check_surrogate`.

## What gets produced

- `results/<training_id>/<surrogate>/` — YAML and CSV files capturing numerical metrics (log-space MAE, LAE99, inference time, compute footprint, etc.).
- `plots/<training_id>/` — visual artifacts: error heatmaps, loss curves, catastrophic detection plots, uncertainty charts, and more depending on the enabled evaluation switches.
- `results/<training_id>/all_metrics.csv` + `metrics_table.csv` — flattened tables for spreadsheet analysis and the optional `compare: true` stage.

## Evaluation switches recap

| Switch | Effect |
| --- | --- |
| `losses` | Saves epoch-wise train/test loss plots. |
| `iterative` | Runs multi-step rollouts to measure drift over time. Set to `true`, or to a block of options (see below). |
| `gradients` | Correlates gradient norms with prediction errors. |
| `timing` | Measures inference latency (multiple passes). |
| `compute` | Records parameter counts and memory usage. |
| `compare` | Builds cross-surrogate comparison tables/plots (requires ≥2 surrogates). |

Leave switches off to skip expensive analyses and plotting functions. But compared to the training, eval is very lightweight: it only loads models and runs inference on the test set, and hence usually takes a few minutes at most. 

## Iterative rollouts

With `iterative` enabled, the model is fed its own predictions instead of the true states, as it would be when coupled to a solver. `iterative: true` uses the defaults, options are set with:

```yaml
iterative:
  mode: fixed_step   # "grid" (default) or "fixed_step"
  dt: 5.0            # fixed_step only: step size in the units of the timesteps
```

| Key | Default | Notes |
| --- | --- | --- |
| `mode` | `"grid"` | `"grid"` restarts the model from its own prediction every `interval` stored timesteps. `"fixed_step"` advances the model by one constant step size `dt` per call, as a solver with a fixed step would. |
| `interval` | `10` | `grid` only: stored timesteps per restart. |
| `dt` | required for `fixed_step` | Step size in the units of the dataset timesteps. The rollout runs to the end of the stored time span; the reference is interpolated onto the times it lands on. |
| `max_steps` | none | `fixed_step` only: cap on the number of steps. |
| `dynamic_span` | none | If set, the median error is also reported separately for trajectories in which some quantity changes by at least this much over the time span and for the rest, in the units of the `*_log` metrics. |
| `example` | `"median_error"` | Sample in the example plot: `"median_error"`, `"dynamic"` (the trajectory that changes most) or a sample index. |

Besides the RMSE, MAE and percentile errors, the iterative evaluation reports the median absolute log error and the fraction of rollout entries that are not finite, so a diverging rollout shows up instead of distorting the averages. The example plot is saved as `example_preds_iterative.png`, or `example_preds_iterative_dt<dt>.png` in `fixed_step` mode.

Iterative rollouts measure how a model behaves when used autoregressively. To improve that behaviour, see :doc:`finetuning`.

## Troubleshooting

- **Config mismatch** — `check_benchmark` errors usually mean `training_id` or modality toggles differ between training and evaluation. Point the evaluator at the stored config (`trained/<training_id>/config.yaml`) or reconcile the differences manually.
- **Missing checkpoints** — make sure the corresponding modality ran successfully. The evaluator cannot invent models that were never trained. You want to check the folder with the models at `trained/<training_id>/`. If the training was complete, you will find a `completed.txt` file there, otherwise inspect `task_list.json` for failed or pending tasks.
- **Large runs** — evaluations are read-heavy. Run them on fast storage and keep the dataset cached locally if possible.
