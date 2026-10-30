# Configuration Reference

All orchestration happens through YAML configuration files. This page documents every configuration option used by `run_training.py`, `run_finetuning.py` and `run_eval.py`, including which keys are required and which ones have defaults.

## How defaults work

- Missing keys fall back to the defaults listed below via `.get(...)`.
- Modality blocks (`interpolation`, `extrapolation`, `sparse`, `batch_scaling`, `uncertainty`) are **disabled** if the entire block is omitted.
- Evaluation switches (`losses`, `iterative`, `gradients`, `timing`, `compute`, `compare`) default to **False** if omitted.
- Required keys must be provided or the run will fail.

## Top-level keys

| Key | Required | Default | Used by | Notes |
| --- | --- | --- | --- | --- |
| `training_id` | Yes | None | Training, Eval | Folder name under `trained/`, `results/`, and `plots/`. |
| `surrogates` | Yes | None | Training, Eval | Ordered list of surrogate class names. |
| `batch_size` | Yes | None | Training, Eval | Int or list aligned with `surrogates`. |
| `epochs` | Yes | None | Training, Eval | Int or list aligned with `surrogates`. |
| `devices` | Yes | None | Training, Eval | List of device strings (`cpu`, `cuda:0`, `mps`, ...). |
| `seed` | No | `42` | Training | Random seed for training. |
| `verbose` | No | `false` | Training, Eval | Extra data-loading logs. |
| `checkpoint` | No | `false` | Training | Enables best-checkpoint saving per model. |

## Dataset block

| Key | Required | Default | Used by | Notes |
| --- | --- | --- | --- | --- |
| `dataset.name` | Yes | None | Training, Eval | Folder inside `datasets/`. |
| `dataset.log10_transform` | No | `true` | Training, Eval | Log10 transform the data. |
| `dataset.log10_transform_params` | No | `true` | Training, Eval | Log10 transform the parameters (if present). |
| `dataset.normalise` | No | `"minmax"` | Training, Eval | `"minmax"`, `"standardise"`, or `"disable"`. |
| `dataset.normalise_per_species` | No | `false` | Training, Eval | Normalize each species independently. |
| `dataset.tolerance` | No | `None` | Training, Eval | Lower bound before log transform (`None` means no lower bound). |
| `dataset.subset_factor` | No | `1` | Training | Down-samples data (smoke tests). |
| `dataset.log_timesteps` | No | `false` | Eval | Used for plotting/log-time axes. |
| `dataset.use_optimal_params` | No | `true` | Training, Eval | Load surrogate-specific defaults from dataset configs. |

## Modality blocks (optional)

All modality blocks are disabled if omitted. If `enabled: true`, the corresponding list/value is required.

| Block | Required | Default | Keys when enabled |
| --- | --- | --- | --- |
| `interpolation` | No | disabled | `intervals` (list of ints) |
| `extrapolation` | No | disabled | `cutoffs` (list of ints) |
| `sparse` | No | disabled | `factors` (list of ints) |
| `batch_scaling` | No | disabled | `sizes` (list of factors, e.g. `["1/2", "1/4"]`) |
| `uncertainty` | No | disabled | `ensemble_size` (int) |

## Evaluation switches

All switches default to `false` if omitted.

| Key | Default | Notes |
| --- | --- | --- |
| `losses` | `false` | Plots training and test losses. |
| `iterative` | `false` | Iterative roll-out evaluation. `true` or a block of options, see below. |
| `gradients` | `false` | Gradient vs error analysis. |
| `timing` | `false` | Inference timing benchmarks. |
| `compute` | `false` | Memory/parameter count benchmarks. |
| `compare` | `false` | Cross-surrogate comparison plots/tables. |

## Iterative block (optional)

`iterative` may be a block instead of `true`. Details in the evaluation guide, section *Iterative rollouts*.

| Key | Default | Notes |
| --- | --- | --- |
| `iterative.mode` | `"grid"` | `"grid"` or `"fixed_step"`. |
| `iterative.interval` | `10` | `grid`: stored timesteps per restart. |
| `iterative.dt` | None | `fixed_step`: step size in the units of the timesteps. Required for `fixed_step`. |
| `iterative.max_steps` | None | `fixed_step`: cap on the number of steps. |
| `iterative.dynamic_span` | None | Also report errors separately for trajectories that change by at least this much, in the units of the `*_log` metrics. |
| `iterative.example` | `"median_error"` | Plotted sample: `"median_error"`, `"dynamic"` or an index. |

## Finetune block (optional)

Read by `run_finetuning.py` only. Step sizes are in the units of the dataset timesteps, errors in the units of the `*_log` metrics. Details in the finetuning guide.

| Key | Default | Notes |
| --- | --- | --- |
| `finetune.dt_range` | None | Required. Smallest and largest step size to train on. |
| `finetune.training_id` | `<training_id>_finetune` | Run the finetuned models are saved as. |
| `finetune.surrogates` | `surrogates` | Surrogates to finetune. |
| `finetune.seed` | `seed` | Seed of the finetuning and of the search. |
| `finetune.epochs` | `75` | Finetuning epochs. |
| `finetune.batch_size` | `1024` | Trajectories per batch. |
| `finetune.learning_rate` | `1e-5` | Adam learning rate. |
| `finetune.gradient_clip` | `1.0` | Maximum gradient norm, `0` disables clipping. |
| `finetune.rollout_weight` | `0.05` | Share of the loss given to the rollout term. |
| `finetune.rollout_steps` | `10` | Graded rollout steps per batch. |
| `finetune.rollout_aggregation` | `"mean"` | `"mean"` or `"sum_decay"`. |
| `finetune.rollout_decay` | `1.0` | Step weight ratio for `sum_decay`. |
| `finetune.start_bias` | `1.0` | Above 1 starts the graded steps earlier. |
| `finetune.noise` | `0.0` | Noise on the rollout start state, normalised units. |
| `finetune.max_warmup_steps` | `1000` | Cap on the steps without gradients before grading. |
| `finetune.eval_dt_range` | `dt_range` | Step sizes the score is taken over. |
| `finetune.eval_n_dt` | `4` | Number of step sizes in the score. |
| `finetune.eval_samples` | `2000` | Test trajectories used for the score. |
| `finetune.dynamic_span` | None | Score only trajectories that change by at least this much. |
| `finetune.optuna` | None | Search: `n_trials`, `params` (same format as tuning), optional `storage`, `study_name`. |

## Metric options

| Key | Default | Notes |
| --- | --- | --- |
| `relative_error_threshold` | `0.0` | Denominator floor for relative error. |
| `error_percentile` | `99` | Percentile used in error summaries. |

## Full example config (defaults)

This example includes every key with the default behavior applied. Required values are filled with common placeholders.

```yaml
# Required
training_id: "example_run"
surrogates: ["MultiONet"]
batch_size: [65536]
epochs: [200]
devices: ["cpu"]

# Optional (defaults)
seed: 42
verbose: false
checkpoint: false

dataset:
  name: "osu2008"
  log10_transform: true
  log10_transform_params: true
  normalise: "minmax"
  normalise_per_species: false
  tolerance: null
  subset_factor: 1
  log_timesteps: false
  use_optimal_params: true

# Modalities (disabled unless enabled)
interpolation:
  enabled: false
  intervals: [2, 3, 4]
extrapolation:
  enabled: false
  cutoffs: [50, 60, 70]
sparse:
  enabled: false
  factors: [2, 4, 8]
batch_scaling:
  enabled: false
  sizes: ["1/16", "1/8", "1/4", "1/2"]
uncertainty:
  enabled: false
  ensemble_size: 5

# Evaluation switches (default false)
losses: false
iterative: false
gradients: false
timing: false
compute: false
compare: false

# Metric options
relative_error_threshold: 0.0
error_percentile: 99
```
