# Rollout Finetuning

CODES trains every surrogate to predict the state at any time from an initial condition. In coupled simulations, surrogates are used differently, as the solver advances the system in multiple steps, and the surrogate is fed the modified state at each step. Errors then compound, and a model that is accurate in one shot can drift or diverge after a few dozen steps. `run_finetuning.py` runs a finetuning schedule on pre-trained CODES models, training a surrogate on its own rollouts to improve stability under deployment conditions.

Finetuning is optional and runs after training:

```bash
python run_training.py   --config config.yaml
python run_finetuning.py --config config.yaml
python run_eval.py       --config trained/<training_id>_finetune/config.yaml
```

`configs/train_eval/config_finetune.yaml` is a complete example.

## Finetuning algorithm

Each batch optimises `(1 - rollout_weight) * data_loss + rollout_weight * rollout_loss`, with gradients clipped to `gradient_clip`.

- **Data loss** — the ordinary one shot loss the model was trained with. It keeps the finetuned model accurate on one-shot predictions and ensures training stability.
- **Rollout loss** — built by rolling the model out:
  1. A target elapsed time is drawn as `0.9 * horizon * u ** start_bias`, with `u` uniform in [0, 1]. `start_bias > 1` moves it towards early times.
  2. The model steps from the true initial state until it has covered that time, **without gradients** (at most `max_warmup_steps` steps). The graded part of the rollout therefore starts from a state the model reached itself, which is where rollout errors arise.
  3. Optionally, Gaussian noise of scale `noise` is added to that state.
  4. The model takes up to `rollout_steps` further steps **with gradients**, stopping at the end of the stored time span. Each step is compared with the reference trajectory at the time reached, and the step losses are averaged over the steps actually taken (`rollout_aggregation: mean`) or summed with weights `rollout_decay ** k` (`sum_decay`).

  Step size is sampled log uniformly from `dt_range` for each step, because coupled simulations may utilize varying step sizes. A model finetuned at one fixed step size tends to work at that step size only.

Batches whose loss is not finite are skipped, so a diverging rollout cannot destroy the weights. The number skipped per epoch is reported.

## Model scoring

After finetuning, and for every trial of a search, the model is rolled out from the initial states of the test split with a constant step size until the end of the stored time span. This is the same procedure as the `fixed_step` mode of the iterative evaluation. For each step size the median absolute error is taken over trajectories, steps and quantities. The score is the geometric mean of these medians over `eval_n_dt` step sizes spread log uniformly across `eval_dt_range` (lower is better). The geometric mean weights every step size equally, even when their errors differ by orders of magnitude. Non finite errors count as `1e30`.

With `dynamic_span` set, only trajectories in which some quantity changes by at least that much over the time span are scored. This helps when most trajectories of a dataset barely change and there are only a few dynamic ones, in which case a very good rollout loss could be achieved by simply predicting a change of zero at each step.

## Configuration

The `finetune` block sits in the same config as training. Only `dt_range` is required.

| Key | Default | Notes |
| --- | --- | --- |
| `dt_range` | required | Smallest and largest step size the finetuned model should handle, in the units of the timesteps. Training draws its step sizes from this range. |
| `training_id` | `<training_id>_finetune` | Name of the run the finetuned models are saved as. |
| `surrogates` | all in `surrogates` | Which trained surrogates to finetune. |
| `seed` | top level `seed` | Seed of all random draws and of the search. |
| `epochs` | `75` | Finetuning epochs. |
| `batch_size` | `1024` | Trajectories per batch. Each trajectory contributes its whole time series to the data loss. |
| `learning_rate` | `1e-5` | Adam learning rate. Finetuning starts from a trained model, so a small learning rate is sufficient. |
| `gradient_clip` | `1.0` | Maximum gradient norm; `0` disables clipping. |
| `rollout_weight` | `0.05` | Share of the loss given to the rollout term. The rollout loss is usually far larger than the data loss at the start, so small values already balance the two. High values can cause training instability. |
| `rollout_steps` | `10` | Graded rollout steps per batch. Backpropagation runs through all of them. |
| `rollout_aggregation` | `"mean"` | `"mean"` or `"sum_decay"` (explained under Finetuning Algorithm). |
| `rollout_decay` | `1.0` | Weight ratio between consecutive steps for `sum_decay`. |
| `start_bias` | `1.0` | Values above 1 start the graded steps earlier in the trajectory. |
| `noise` | `0.0` | Standard deviation of the noise added to the start state, in normalised units. |
| `max_warmup_steps` | `1000` | Cap on the steps taken without gradients before the graded steps. |
| `eval_dt_range` | `dt_range` | Step sizes the score is taken over. |
| `eval_n_dt` | `4` | Number of step sizes in the score. |
| `eval_samples` | `2000` | Test trajectories used for the score. |
| `dynamic_span` | `None` | Score only trajectories that change by at least this much, in the units of the `*_log` metrics (log10 when `dataset.log10_transform` is on). |
| `optuna` | `None` | Hyperparameter search, see below. |

## Hyperparameter search

Add an `optuna` block to search over any of the keys above. The search space uses the same format as for :doc:`tuning`:

```yaml
finetune:
  dt_range: [1.0, 10.0]
  optuna:
    n_trials: 30
    params:
      learning_rate:
        type: float
        low: 1.0e-6
        high: 1.0e-4
        log: true
      rollout_weight:
        type: float
        low: 1.0e-3
        high: 0.3
        log: true
      rollout_steps:
        type: int
        low: 3
        high: 20
      noise:
        type: float
        low: 0.0
        high: 0.3
```

Each trial finetunes the trained model from scratch with the suggested values, and the best trial is saved. Every trial is scored on step sizes drawn fresh within `eval_dt_range`.

The study is stored in SQLite under `trained/<finetune training_id>/` (set `optuna.storage` to use another database) and resumes when rerun: raise `n_trials` to add trials. The weights of every trial are kept next to the study, so the best trial can come from an earlier session. The seed of the search is `finetune.seed`; reusing a seed reruns the same trials.

## Outputs and evaluation

- `trained/<finetune training_id>/<Surrogate>/<surrogate>_main.pth` — the finetuned models, saved as an ordinary training run. Their recorded training duration is the finetuning time, or the selected trial's finetuning time for a search. Older search trials without timing metadata have no recorded duration.
- `trained/<finetune training_id>/config.yaml` — a copy of your config for that run. Modality blocks and loss plots are switched off, because only the main model is finetuned and the stored loss curves belong to the original training.
- `results/<finetune training_id>/<surrogate>_finetune.yaml` — the score and the median error per step size before and after finetuning, the settings used, and for a search the best trial.

Evaluate the finetuned models like any other run with `run_eval.py --config trained/<finetune training_id>/config.yaml`. To see their rollout behaviour, enable the iterative evaluation in `fixed_step` mode (see :doc:`evaluation`).

## Supported surrogates

`FullyConnected`, `MultiONet` and their residual variants (see :doc:`architectures`) support finetuning. Residual variants can be the better starting point for rollouts, because a step that changes little is easy for them to represent. Finetuning a surrogate requires its `forward_from_state` method, which predicts states from an initial state at given time inputs without a dataloader; a custom surrogate supports finetuning once it implements it (see :doc:`../extending-benchmark`).
