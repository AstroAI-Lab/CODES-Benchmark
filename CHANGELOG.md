# Changelog

All notable changes to CODES are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and CODES uses
[semantic versioning](https://semver.org/). Before 1.0.0, a minor release may contain
breaking changes. Changes that alter benchmark results are listed under "Changes
affecting results".

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

[0.2.0]: https://github.com/AstroAI-Lab/CODES-Benchmark/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AstroAI-Lab/CODES-Benchmark/releases/tag/v0.1.0
