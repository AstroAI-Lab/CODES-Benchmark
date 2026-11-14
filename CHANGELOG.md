# Changelog

All notable changes to CODES are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and CODES uses
[semantic versioning](https://semver.org/). Before 1.0.0, a minor release may contain
breaking changes. Changes that alter benchmark results are listed under "Changes
affecting results".

## [Unreleased]

### Added

- Versioned reference benchmark protocols, timing pilots and resumable per-dataset
  runs for the four A&A datasets and architectures, independent of the compute host.
- Validated static benchmark result exports and documentation, with an explicit
  separation between public scientific provenance and local execution metadata.
- Opt-in synchronized accelerator inference timing (`synchronize_timing`) with
  an unmeasured warm-up and explicit measurement metadata for reference runs.

- Registry-driven training smoke and end-to-end tests automatically include new
  surrogates. Common interface tests report unexpected failures instead of skipping
  them; numerical golden references remain explicitly maintained.
- Contributor credit for Gianluca Möschter's code, data, research, documentation,
  tests and user testing, and inclusion in the software citation author list.
- Autoregressive rollout finetuning of trained surrogates, with an optional Optuna
  search that can resume earlier trials.
- Fixed-step rollout evaluation, including median absolute log error, non-finite
  rollout fractions and optional scoring of dynamically changing trajectories.
- FullyConnectedResidual and MultiONetResidual surrogates, with fallback to their
  parent's dataset configuration.
- Optional `gradient_clip` for FullyConnected, MultiONet and LatentNeuralODE training.

### Changes affecting results

- Iterative evaluation now restarts each chunk from the prediction at that chunk's
  start, rather than the prediction one timestep earlier.
- Rollout time inputs now follow the stored time grid, including non-uniform grids,
  rather than assuming evenly spaced linear or logarithmic times. Together these
  fixes change existing iterative evaluation results, including `mLAE_iter`.

### Fixed

- Failed local training runs now raise an error instead of reporting successful
  completion. Successful sibling tasks remain recorded; unfinished tasks are kept
  for retry after exceptions, worker crashes or interruption.
- Preserved the 0.2.0 `SmoothL1Loss` default of `beta: 1.0`; finetuning also uses the
  configured loss and beta.
- Mean rollout loss uses the number of steps actually taken when a rollout stops
  at the end of the stored time span.
- Saved finetuned models record their finetuning duration, including the selected
  trial of a resumed search. Trials saved without timing metadata use no duration.
- Error percentiles work for arrays exceeding PyTorch's quantile size limit.
- Rollout plots handle zero and non-finite errors.
- LatentNeuralODE's `loss_weights` can be set in its configuration.

### Changed

- Local training progress is rendered by the parent, using stacked bars for
  interactive terminals and timestamped text for redirected or batch output.
  The `progress` setting supports `auto`, `bars`, `plain` and `off`; per-task
  logs retain progress, worker output and failure details.
- Local training now runs each task in a fresh spawned process, with the parent
  coordinating device slots and pending work. Single-device runs use the same
  isolated execution path. The scheduler-independent executor can also be called
  directly by a future external worker.
- Training workers initialize their own seeds and deterministic settings. The new
  `cpu_threads` setting defaults to one PyTorch CPU thread per worker and respects
  CPU affinity and Slurm's per-task CPU limit.
- Standard surrogate training now shares a common loop, with no intended numerical
  changes. Custom `fit` implementations remain supported.
- `ChemDataset` remains available but is deprecated. MultiONet's `setup_criterion`
  is also deprecated; training uses the configured loss.
- Updated the checkout, uv setup, Codecov and Pages Actions to current releases
  compatible with Node 24. CI also checks the pinned requirements against `uv.lock`.
- Development builds now identify themselves as `0.3.0.dev0`; the documentation
  displays the installed package version.
- Added the 0.2.0 release DOI and publication date to citation metadata, an
  all-versions DOI badge, and corrected repository and Codecov links.
- Replaced Black and isort with Ruff for linting, import sorting and formatting.
  CI and pre-commit use the same pinned Ruff version. The rule set covers `E`, `F`,
  `I`, `SIM` and `UP` on Python files and the tutorial notebook.
- Citation guidance now prioritises the software release, with the A&A paper
  requested when using its astrochemical datasets or analysis.

### Removed

- The old threaded training `worker` and `DummyLock` exports. Use the local
  training runners or `execute_training_task` for independent task execution.
- The unused direct-module helpers `DeepONet.don_utils.PreBatchedDataset`,
  `DeepONet.don_utils.custom_collate_fn`, `FCNN.fcnn.FCPrebatchedDataset` and
  `FCNN.fcnn.fc_collate_fn`,
  the legacy `denormalize_old` method, duplicate `DeepONet.train_utils` module,
  and LatentNeuralODE `fit_profile`. Use
  `DeepONet.don_utils.mass_conservation_loss` for the loss helper and wrap a
  normal `fit` call with `torch.profiler` for profiling.
- Unused file/path and wall-clock seeding utilities in `DeepONet.don_utils`.

## [0.2.0] - 2026-10-23

This release fixes bugs that affect benchmark results, including those of the A&A paper
(Janssen, Branca & Buck 2026), which was produced with 0.1.0. Retrain models before
comparing them with results from 0.2.0. We retrained the A&A models with the checkpoint,
optimizer and seeding fixes below; the conclusions of the paper are unchanged.

### Changes affecting results

- Fixed best-checkpoint handling when several models are trained in parallel.
- Fixed the weights saved for models trained with schedule-free optimizers.
- Fixed the seeding of deep-ensemble members.
- Training is now reproducible when models are trained in parallel. The data shuffling
  changed, so models differ from 0.1.0 models trained with the same seed.
- Fixed a bug where interpolation and extrapolation timescales were mismatched between
  training and evaluation.
- `beta` is now applied to `nn.SmoothL1Loss`, and LatentPoly now uses the configured
  loss function. The dataset configs are adjusted so the published models train as
  before.
- `eta_min` is now a fraction of `learning_rate`, as documented. Convert absolute values
  by dividing them by `learning_rate`; the dataset configs are converted.
- Fixed the denormalisation of data normalised with `normalise: "standardise"`.
- Fixed the recorded training duration of models trained in parallel.

### Fixed

- Log-transforming non-positive data now raises an error. Set `dataset.tolerance` to clip
  such data.
- Fixed loading parametric datasets with `normalise: "disable"`.
- `batch_size` and `epochs` can be single values in evaluation, as in training.
- Fixed LatentNeuralODE with `model_version: "v1"`.
- Fixed the `n_train_samples` saved with FullyConnected and MultiONet models.
- Re-running a tuning study with an unchanged config no longer asks to overwrite it.
- `run_eval.py --device` now takes effect.
- Fixed race conditions between parallel training workers.
- Regular (non-editable) installs now include all subpackages.
- CI now runs for pull requests from forks.

### Added

- `codes.__version__`. Saved models and benchmark metrics record the CODES version and
  git commit that produced them.
- End-to-end and golden-value tests.

### Removed

- Support for Python 3.10, which has reached its end of life. CODES now requires
  Python 3.11 or newer.

### Security

- Updated dependencies with known vulnerabilities, including PyTorch (now 2.14).

## [0.1.0] - 2026-02-12

The version used for the A&A paper.

[Unreleased]: https://github.com/AstroAI-Lab/CODES-Benchmark/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AstroAI-Lab/CODES-Benchmark/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AstroAI-Lab/CODES-Benchmark/releases/tag/v0.1.0
