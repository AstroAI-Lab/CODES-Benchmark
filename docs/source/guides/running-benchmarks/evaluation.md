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
| `iterative` | Runs multi-step rollouts to measure drift over time, defaulting to brackets of 10 steps. |
| `gradients` | Correlates gradient norms with prediction errors. |
| `timing` | Measures inference latency (multiple passes). |
| `compute` | Records parameter counts and memory usage. |
| `compare` | Builds cross-surrogate comparison tables/plots (requires ≥2 surrogates). |

Leave switches off to skip expensive analyses and plotting functions. But compared to the training, eval is very lightweight: it only loads models and runs inference on the test set, and hence usually takes a few minutes at most. 

## Troubleshooting

- **Config mismatch** — `check_benchmark` errors usually mean `training_id` or modality toggles differ between training and evaluation. Point the evaluator at the stored config (`trained/<training_id>/config.yaml`) or reconcile the differences manually.
- **Missing checkpoints** — make sure the corresponding modality ran successfully. The evaluator cannot invent models that were never trained. You want to check the folder with the models at `trained/<training_id>/`. If the training was complete, you will find a `completed.txt` file there, otherwise inspect `task_list.json` for failed or pending tasks.
- **Large runs** — evaluations are read-heavy. Run them on fast storage and keep the dataset cached locally if possible.
