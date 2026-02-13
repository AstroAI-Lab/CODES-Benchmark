# Hyperparameter Tuning

Optuna powers the tuning stage. Each study runs inside `run_tuning.py`, which manages device pools, Optuna storage, and all surrogate-specific settings.

## Workspace layout

1. Create a config file under `configs/tuning/` (for example `configs/tuning/sqlite_quickstart.yaml`). Each config must contain a unique `tuning_id`.
2. Launch the tuner and pass the config path:
   ```bash
   python run_tuning.py --config configs/tuning/sqlite_quickstart.yaml
   ```
3. CODES copies the config into `tuned/<tuning_id>/optuna_config.yaml` for reproducibility. On subsequent runs it compares the stored copy against the newly provided file and asks whether you want to reuse the stored config, overwrite it (the old one is backed up), or abort.

`tuning_id` becomes the prefix for all Optuna studies (e.g., `primordial_MultiONet`). The script also creates `tuned/<tuning_id>/models/` to store intermediate checkpoints when you enable pruning.

## Config anatomy

Below is a condensed version of `configs/tuning/sqlite_quickstart.yaml` using SQLite storage for zero setup. Adjust datasets/surrogates as needed.

```yaml
seed: 42
tuning_id: "primordial"
dataset:
  name: primordial
  log10_transform: true
  normalise: minmax
devices: ["cuda:0"]
prune: true
multi_objective: true
population_size: 50

storage:
  backend: "sqlite"
  path: "tuned/primordial/primordial.db"

surrogates:
  - name: MultiONet
    batch_size: 4096
    epochs: 8192
    trials: 120
    optuna_params:
      activation:
        type: categorical
        choices: ["ReLU", "LeakyReLU", "Tanh", "GELU", "Softplus"]
      hidden_size:
        type: int
        low: 10
        high: 500
      learning_rate:
        type: float
        low: 1.0e-6
        high: 1.0e-3
        log: true
      output_factor:
        type: int
        low: 1
        high: 200
```

### Key sections

- **dataset** — mirrors the training config and ensures Optuna downloads/configures the same data pipeline.
- **devices** — every entry becomes a worker slot. `run_tuning.py` keeps a queue (`queue.Queue`) of device tokens and runs Optuna with `n_jobs = len(devices)`. SQLite storage warns about concurrent writers; you can list multiple devices, but heavy parallelism works best with Postgres.
- **storage** — choose between `sqlite` (single-file DB, no external services) and `postgres` (scales to many workers). If omitted, CODES defaults to Postgres for backward compatibility and expects `postgres_config` to be present.
- **postgres_config** — only required when `storage.backend: "postgres"`. Supports
  - `mode: local`: launches/validates a local PostgreSQL instance (binaries in `database_folder`).
  - `mode: remote`: connects to an existing server (set `host`, `port`, `user`, and optionally `password` or rely on `PGPASSWORD`).
- **surrogates** — per-architecture specs. Each entry sets:
  - `batch_size` / `epochs`: used to build the objective.
  - `trials`: maximum valid trials for that surrogate.
  - `optuna_params`: the search space. The keys correspond to attributes on the surrogate’s config dataclass; Optuna writes the sampled values into that config before training.

You can add `global_optuna_params` for common parameters, enable `fine: true` for automatic “around-the-best” refinement, or toggle between single-objective (`direction="minimize"`) and dual-objective (`directions=["minimize","minimize"]`) mode. In multi-objective runs we typically optimize log-space accuracy (LAE$_{99}$) and inference time, but you can choose any pair of metrics.

## What `run_tuning.py` does

1. Loads the YAML and copies it into `tuned/<tuning_id>/` (via `prepare_workspace`).
2. Initializes the Optuna database (SQLite file or Postgres), prompting if a study with the same `tuning_id` already exists.
3. Downloads the dataset once per run.
    4. Iterates over the `surrogates` list. For each surrogate it:
       - Builds a study name (`<tuning_id>_<surrogate>`).
       - Selects the sampler/pruner (`TPESampler` + Hyperband or NSGA-II with no pruning).
       - Creates a device queue and `objective_fn = create_objective(...)`.
   - Calls `study.optimize()` with `n_jobs=len(devices)`. Each Optuna worker pulls a device token, trains a model with the sampled hyperparameters, and returns the objective(s).
   - Tracks ETA via `tqdm`. The helper `MaxValidTrialsCallback` stops once enough successful trials finished (OOM and time-pruned trials are ignored).

You can resume a study by rerunning the same command; Optuna reuses the storage and continues sampling until `n_trials` valid runs exist. Trial budgets are usually sized heuristically (e.g., `~15 ×` the number of tuned hyperparameters), but you can override per surrogate via the `trials` field.

## Capturing the best hyperparameters

CODES does not auto-promote trial settings. Use Optuna’s tooling to inspect studies:

- Python REPL / script:
  ```python
  import optuna
  study = optuna.create_study(
      study_name="primordial_MultiONet",
      storage="postgresql+psycopg2://optuna_user@localhost:5432/primordial",
      direction="minimize",
      load_if_exists=True,
  )
  print(study.best_trial.params)
  ```
- [Optuna Dashboard](https://optuna.github.io/optuna-dashboard/) or `optuna.visualization`.

Dual-objective runs produce Pareto fronts like the ones shown in the paper excerpt. You can manually pick a “knee point” trade-off (accuracy vs. latency) or script your own selection rule. Whatever you choose, feed the accepted settings back into `config.yaml` under `surrogate_configs` or store dataset-specific defaults in `datasets/<name>/surrogates_config.py` (dataclasses). Those defaults load automatically when `dataset.use_optimal_params` is `true`; setting `use_optimal_params: false` switches back to plain config-defined hyperparameters.

## Advanced options

### Postgres storage

For large-scale or multi-GPU sweeps, switch the storage block to Postgres:

```yaml
storage:
  backend: "postgres"

postgres_config:
  mode: "local"         # or "remote"
  host: "localhost"
  port: 5432
  user: "optuna_user"
  database_folder: "/path/to/postgres/"
```

If the `storage` section is omitted entirely, CODES assumes `backend: "postgres"` to remain backward compatible. Postgres handles concurrent writers gracefully, so `devices` can list many GPUs without hitting `database is locked` errors.

### Fine-tuning stage

Setting `fine: true` tells CODES to derive a narrow search space around the best-known configuration for each surrogate (taken from previous runs or dataset defaults):

```yaml
fine: true
```

The first run (with `fine: false`) explores the full search space and establishes a good baseline. The optional fine stage then:

1. Builds tight bounds around every tunable scalar (log-space ±factor) using `build_fine_optuna_params`.
2. Overrides the trial budget to `max(10 × N, 10)` where `N` is the number of fine-tunable parameters.
3. Prints the refined ranges per surrogate and stores them in `tuned/<tuning_id>/fine_summary.yaml`.

This two-step process saves compute by spending most trials in promising regions rather than re-sampling the entire space.

### Conditional parameter sampling

Some hyperparameters only matter when a parent switch takes a specific value (e.g., `momentum` is relevant only for SGD, `poly_power` only for the polynomial scheduler). The tuner encodes these relationships in `make_optuna_params`: it samples parent switches first and then conditionally samples child parameters. This prevents Optuna from proposing incompatible combinations (such as a momentum value while using Adam) that would otherwise waste trials or require manual filtering. You can still expose child parameters directly if desired; the helper samples them once the relevant switch is active.

### Time-based pruning

To avoid exceptionally slow trials hogging resources, CODES automatically sets a runtime threshold after the initial warm-up period. Once enough successful trials complete, it computes `mean + std` of their durations and prunes future trials whose wall-clock time exceeds that threshold. In multi-objective mode, this applies to both accuracy/time objectives—the accuracy value is capped while the runtime objective records the observed duration.

Disable this behaviour by adding:

```yaml
time_pruning: false
```

to your tuning config. When disabled, all trials run to completion unless Optuna’s other pruners intervene.

Remember that tuning explores unconstrained space—double-check the resulting configs before launching expensive training sweeps.
