# Evaluation & Reporting

`run_eval.py` evaluates your saved checkpoints and writes metrics under `results/<training_id>/` and plots under `plots/<training_id>/`. The main accuracy metrics use the log-space conventions described in [Accuracy metrics](accuracy-metrics.md).


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
| `timing` | Measures device-resident and host-to-host endpoint and trajectory latency. |
| `compute` | Records parameter counts and memory usage. |
| `compare` | Builds cross-surrogate comparison tables/plots (requires ≥2 surrogates). |

Leave switches off to skip analyses you do not need. The inference timing sweep uses the configured batch sizes and can take longer for expensive trajectories or long autoregressive rollouts.

## Inference workloads

With `timing: true`, CODES loads each main checkpoint once and times two requests using validation initial states and parameters already resident on the selected device.

- **Endpoint** returns one updated state per cell at the requested elapsed time. It also reports mean absolute error over finite predictions in normalized model coordinates and the fraction of nonfinite predictions.
- **Trajectory** returns states at 16 equally spaced positive elapsed times ending at the requested horizon. A model can share computation across those queries. The recorded time grid maps physical elapsed times to the dummy time coordinate used during training.

The same horizons are used for both requests. CODES selects the first positive elapsed time, the recorded elapsed time closest on a log scale to the geometric midpoint, and the full horizon. Short grids may produce fewer distinct horizons. Targets for endpoint error come from the stored validation trajectories. Large batches cycle through the validation initial states in order, so the requested batch sizes remain comparable even on small datasets.

```yaml
timing: true
inference_timing:
  batch_sizes: [1, 8, 32, 128, 512, 2048, 8192]
  host_to_host: true
  # Optional maximum autoregressive substep in dataset time units
  # rollout_step:
  #   FullyConnectedResidual: 0.01
```

Set `rollout_step` to a positive duration for all surrogates, or map surrogate names to durations. A requested interval is split into substeps of at most that duration, with the last step landing on the requested time. Each substep treats the previous prediction as a fresh initial state and maps its elapsed duration from the start of the training grid, as in fixed-step evaluation. Trajectory requests reuse the state from the previous requested time. Choose a substep suited to the checkpoint. Surrogates without a specified substep use their direct query interface. This setting is independent of `iterative`, which runs a separate accuracy analysis.

The curves report median and p95 batch latency from five measured calls, cells per second, and additional CUDA peak allocated memory above the preloaded model and batch. The detailed values are in `timing.workloads` of each metrics YAML. A trajectory cell counts once in cells per second even though its request returns multiple states. The previous full-loader timing stays available for historical comparisons.

### Host-to-host measurement

This measures a blocking call from raw, CPU-resident validation states and parameters to updated states in a CPU output buffer in physical units. Each timed call packs and transforms the input using the dataset settings and saved normalization statistics, invokes the same endpoint or trajectory request, then converts the prediction back to physical units. On CUDA, the call uses reusable pinned buffers, copies inputs and outputs on the same stream as inference, and waits for an event after the output copy before converting the result. CPU evaluation measures the equivalent physical-unit call in memory. MPS uses ordinary host buffers and synchronizes before reading the result.

Checkpoint loading, disk reads, and buffer setup happen before timing. This models a single blocking request, so a simulation integration can measure rank-level gathering, scattering, or pipelining separately. Results are saved under `timing.host_to_host` in the metrics YAML and in separate host-to-host endpoint and trajectory plots. Set `inference_timing.host_to_host: false` to skip this measurement.

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
