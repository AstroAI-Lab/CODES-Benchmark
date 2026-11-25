# Hyperparameter Tuning

Optuna powers the tuning stage. Each study runs inside `run_tuning.py`, which manages device pools, Optuna storage, and all surrogate-specific settings.

For a first run, follow [Getting Started](../../getting-started.md) to install CODES, stage the Lotka–Volterra dataset, and run four CPU trials locally or on Slurm.

## Workspace layout

1. Copy `configs/tuning/config_smoke.yaml` and give the copy its own `tuning_id` when changing the dataset or search space.
2. Launch the tuner and pass the config path:
   ```bash
   uv run run_tuning.py --config configs/tuning/config_smoke.yaml
   ```
3. CODES copies the config into `tuned/<tuning_id>/optuna_config.yaml` for reproducibility. On subsequent runs it compares the stored copy against the newly provided file and asks whether you want to reuse the stored config, overwrite it (the old one is backed up), or abort.

`tuning_id` becomes the prefix for all Optuna studies, for example `primordial_multionet`. The script creates `tuned/<tuning_id>/models/` to store trial checkpoints.

## Config anatomy

This example uses the default journal storage. For a smoke test, use `configs/tuning/config_smoke.yaml` as shipped.

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
- **devices** — each entry is a local worker slot. For Slurm use `slurm.workers` and `slurm.device` to set the array size and per-worker device.
- **storage** — omit this block to use `tuned/<tuning_id>/optuna.journal`. Set `storage.backend` explicitly to use a different backend. See {ref}`storage options <tuning-storage-options>`.
- **surrogates** — per-architecture specs. Each entry sets:
  - `batch_size` / `epochs`: used to build the objective.
  - `trials`: maximum valid trials for that surrogate.
  - `optuna_params`: the search space. The keys correspond to attributes on the surrogate’s config dataclass; Optuna writes the sampled values into that config before training.

You can add `global_optuna_params` for common parameters, enable `fine: true` for automatic “around-the-best” refinement, or toggle between single-objective (`direction="minimize"`) and dual-objective (`directions=["minimize","minimize"]`) mode. In multi-objective runs we typically optimize log-space accuracy (LAE$_{99}$) and inference time, but you can choose any pair of metrics.

## What `run_tuning.py` does

1. Loads the YAML and copies it into `tuned/<tuning_id>/` (via `prepare_workspace`).
2. Opens the configured Optuna storage and loads or creates each study.
3. Downloads the dataset once per run.
4. Iterates over the `surrogates` list. For each surrogate it:
   - Builds a study name (`<tuning_id>_<surrogate>`).
   - Selects the sampler/pruner (`TPESampler` + Hyperband or NSGA-II with no pruning).
   - Creates a device queue and `objective_fn = create_objective(...)`.
   - Calls `study.optimize()` with `n_jobs=len(devices)`. Each Optuna worker pulls a device token, trains a model with the sampled hyperparameters, and returns the objective(s).
   - Tracks ETA via `tqdm`. The helper `MaxValidTrialsCallback` stops once enough successful trials finished (OOM and time-pruned trials are ignored).

You can resume a study by rerunning the same command; Optuna reuses the storage and continues sampling until `n_trials` valid runs exist. Trial budgets are usually sized heuristically (e.g., `~15 ×` the number of tuned hyperparameters), but you can override per surrogate via the `trials` field.

### Inference timing and existing studies

Multi-objective tuning and refinement use the same timing method as evaluation.
Each measurement has one unmeasured warm-up pass followed by five measured passes.
CUDA and MPS work is synchronized immediately before and after each forward call.
The objective is the mean summed forward-call wall time over the held-out loader.
Data-loader iteration is outside the measured intervals. CPU timing uses the same
warm-up without accelerator waits.

The `inference_timing` attribute records the protocol on the study and its trials.
Trials also record the device, batch size, and number of predicted states. Existing
multi-objective studies with legacy or unmarked timing objectives cannot be resumed.
Use a new `tuning_id` or study name and retain the old study for historical analysis.
Single-objective accuracy studies can still be resumed.

Keep the GPU type, batch size, and workload fixed when comparing trial timings.
Other work on the same GPU can affect measured times even when calls are synchronized.

## Tuning with a Slurm array

Use one array per study and a `tuning_id` dedicated to this campaign. Submit
from a workspace visible at the same path on the submit host and all workers.
{ref}`Stage the dataset <stage-dataset>` at
`datasets/<name>/data.hdf5` before submitting. All workers use the journal on
that shared filesystem by default. To use an existing PostgreSQL service,
follow the {ref}`PostgreSQL setup <tuning-postgresql-service>`.

### Submit from the tuning config

Add a top-level `slurm` block to your tuning YAML. `workers` sets the number
of array tasks for the selected study. The tuning config's `trials` setting
sets the target number of valid trials for the whole study.

```yaml
slurm:
  time: "00:10:00"
  cpus_per_task: 1
  mem: 2G
  workers: 2
  max_parallel: 2
  device: cpu
```

```bash
uv run run_tuning.py --slurm submit --config tuning.yaml --study MultiONet
uv run run_tuning.py --slurm status --config tuning.yaml --study MultiONet
```

`submit` prepares `tuned/<tuning_id>/<study_name>.slurm.json`, submits the
workers, and prints the Slurm job ID. Use the same resource and launcher
settings described in {ref}`the training guide <slurm-training-array>`.
`--study` is required when the tuning config lists several surrogates. Use
`--slurm status --config tuning.yaml --study MultiONet` to read the trial counts.
Slurm logs default to `slurm-logs/` next to the manifest.

### Prepare and submit an array manually

```bash
uv run run_tuning.py --slurm prepare --config tuning.yaml --study MultiONet --manifest tuning.json
mkdir -p logs
```

Select a model with `--study` when the config lists several surrogates. Prepare
from one process per study. Reuse the original manifest to resume the study.
The `--max-attempts` option sets the maximum number of trial attempts that can
train and defaults to twice the target.

Using the launcher from {ref}`the training guide <slurm-training-array>`,
this example starts four workers with one GPU and four CPU threads each. Set
the scheduler options for your cluster.

```bash
sbatch --chdir="$PWD" --array=0-3 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=4 --time=02:00:00 --output='logs/tune-%A_%a.log' \
    array-worker.sbatch run_tuning.py --slurm worker --manifest tuning.json \
    --device cuda:0 --cpu-threads 4
```

Workers sample and train independently against the shared study. Only completed
trials with finite objectives and no recorded exception count toward the target.
Trials already training can finish after the target is reached. The attempt cap
also counts failed or pruned work. Journal storage has no Optuna heartbeat, so
trials from workers killed abruptly can remain `RUNNING` and still use an attempt.
If `status` reports `budget_exhausted`, the study cannot start more trials even
if a trial is still marked `RUNNING`.
PostgreSQL heartbeats let a later worker mark interrupted trials failed once
their heartbeat expires. [Optuna RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)

```bash
uv run run_tuning.py --slurm status --manifest tuning.json
```

Check `valid` against `target` after the array finishes. Use the original
manifest for another array when the study still has budget. Keep the GPU model,
software, and inference workload consistent for every worker in a timed study.
Pass the `tuning_id` as the study prefix when generating Pareto plots.

```bash
uv run python -m codes.tune.evaluate_tuning --study_name my_tuning_id --chosen_indices ''
```

For a PostgreSQL study configured through `.env`, use this command.

```bash
uv run --env-file .env python -m codes.tune.evaluate_tuning --study_name my_tuning_id --chosen_indices ''
```

## Capturing the best hyperparameters

CODES does not auto-promote trial settings. Use Optuna’s tooling to inspect studies:

- Python REPL or script, using the journal from the smoke example:
  ```python
  import optuna
  from optuna.storages.journal import JournalFileBackend

  storage = optuna.storages.JournalStorage(
      JournalFileBackend("tuned/codes_smoke_tuning/optuna.journal")
  )
  study = optuna.load_study(
      study_name="codes_smoke_tuning_fullyconnected",
      storage=storage,
  )
  print(study.best_trial.params)
  ```
- [Optuna Dashboard](https://optuna.github.io/optuna-dashboard/) or `optuna.visualization`.

Dual-objective runs produce Pareto fronts like the ones shown in the paper excerpt. You can manually pick a “knee point” trade-off (accuracy vs. latency) or script your own selection rule. Whatever you choose, feed the accepted settings back into `config.yaml` under `surrogate_configs` or store dataset-specific defaults in `datasets/<name>/surrogates_config.py` (dataclasses). Those defaults load automatically when `dataset.use_optimal_params` is `true`; setting `use_optimal_params: false` switches back to plain config-defined hyperparameters.

(tuning-storage-options)=
## Storage options

### Journal (default)

Journal storage uses a file under the tuning run directory. Set `storage.path`
to put it elsewhere. For Slurm, every worker must be able to access and write
the same file and its parent directory at the same path. Optuna's journal file
backend supports NFS. [Optuna Journal Storage](https://optuna.readthedocs.io/en/v4.8.0/tutorial/20_recipes/011_journal_storage.html)

```yaml
storage:
  backend: journal
  path: tuned/my_campaign/optuna.journal
```

(tuning-postgresql-service)=
### Existing PostgreSQL service

Ask your cluster administrator for a database, login role, hostname, port,
and TLS settings. Both the submission host and compute nodes must be able to
connect. The database must already exist, and the role needs permission to
create and update Optuna's tables in the target schema. Copy the smoke config
with `cp configs/tuning/config_smoke.yaml tune-postgres.yaml`, add your site's
scheduler settings under `slurm`, and set the new tuning ID and storage backend.
Stage the dataset first if you have not already done so.

```yaml
tuning_id: codes_smoke_tuning_postgres
storage:
  backend: postgres
```

In the CODES repository root, create `.env` alongside `pyproject.toml` by
copying `.env.example` if you do not already have a file. Git ignores `.env`.
Restrict the file to your account, then edit it to use the host, port, database,
role, password, and TLS settings from your site.

```bash
cp .env.example .env
chmod 600 .env
```

```text
CODES_OPTUNA_STORAGE_URL=postgresql+psycopg2://DB_USER@DB_HOST:5432/DB_NAME
PGPASSWORD='DB_PASSWORD'
# PGSSLMODE=verify-full
# PGSSLROOTCERT=/absolute/path/to/ca.pem
```

Slurm workers inherit `PGPASSWORD` in their job environment. To keep the
password out of that environment, remove `PGPASSWORD` and set
`PGPASSFILE="${HOME}/.pgpass-codes"` in `.env`. Add a line in the form
`DB_HOST:5432:DB_NAME:DB_USER:DB_PASSWORD` to the passfile, using your connection
values. Set its permissions to `0600` and put it on a path the workers can read.
[PostgreSQL password files](https://www.postgresql.org/docs/current/libpq-pgpass.html)

Use `--env-file .env` for the connection check, submission, and status. If you
previously exported any of these variables, unset them before using `.env` so
the shell values do not override the file.

```bash
uv run --env-file .env python - <<'PY'
import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["CODES_OPTUNA_STORAGE_URL"])
with engine.connect() as connection:
    print(connection.execute(text("SELECT 1")).scalar_one())
PY
uv run --env-file .env run_tuning.py --slurm submit --config tune-postgres.yaml
uv run --env-file .env run_tuning.py --slurm status --config tune-postgres.yaml
```

CODES passes the submission environment to Slurm's workers. If your site
launcher filters it, preserve the connection variables and access to any
passfile or TLS files you configured. If you use the pip workflow, export the
same variables in your activated environment before invoking the Python commands.

Existing local configs that specify `postgres_config` without a backend continue
to use PostgreSQL. Explicit `storage.backend: sqlite` with `storage.path` is
available for local runs. Use one worker with SQLite to avoid writer contention.

## Advanced options

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
