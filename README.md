# CODES Benchmark

[![codecov](https://codecov.io/github/robin-janssen/CODES-Benchmark/branch/main/graph/badge.svg?token=TNF9ISCAJK)](https://codecov.io/github/robin-janssen/CODES-Benchmark) ![Static Badge](https://img.shields.io/badge/license-GPLv3-blue) ![Static Badge](https://img.shields.io/badge/NeurIPS-2024-green)
[![All Contributors](https://img.shields.io/github/all-contributors/AstroAI-Lab/CODES-Benchmark?color=ee8449&style=flat-square)](#contributors)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

🎉 Accepted to the ML4PS workshop @ NeurIPS 2024

Benchmark coupled ODE surrogate models on curated datasets with reproducible training, evaluation, and visualization pipelines. CODES helps you answer: _Which surrogate architecture fits my data, accuracy target, and runtime budget?_

## What you get

- Baseline surrogates (MultiONet, FullyConnected, LatentNeuralODE, LatentPoly) with configurable hyperparameters
- Rich datasets spanning chemistry, astrophysics, and dynamical systems
- Optional studies for interpolation/extrapolation, sparse data regimes, uncertainty estimation, and batch scaling
- Automated reporting: accuracy tables, resource usage, gradient analyses, and dozens of diagnostic plots

## Two-minute quickstart

**uv (recommended)**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
uv sync                       # creates .venv from pyproject/uv.lock
source .venv/bin/activate
uv run python run_training.py --config configs/train_eval/config_minimal.yaml
uv run python run_eval.py --config configs/train_eval/config_minimal.yaml
```

**pip alternative**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python run_training.py --config configs/train_eval/config_minimal.yaml
python run_eval.py --config configs/train_eval/config_minimal.yaml
```

Outputs land in `trained/<training_id>`, `results/<training_id>`, and `plots/<training_id>`. The `configs/` folder contains ready-to-use templates (`train_eval/config_minimal.yaml`, `config_full.yaml`, etc.). Copy a file there and adjust datasets/surrogates/modalities before running the CLIs.

## Documentation

- [Main docs & tutorials](https://robin-janssen.github.io/CODES-Benchmark/)
- [API reference (Sphinx)](https://robin-janssen.github.io/CODES-Benchmark/modules.html)
- [Paper on arXiv](https://arxiv.org/abs/2410.20886)

The GitHub Pages site now hosts the narrative guides, configuration reference, and interactive notebooks alongside the generated API docs.

## Repository map

| Path                                              | Purpose                                                                          |
| ------------------------------------------------- | -------------------------------------------------------------------------------- |
| `configs/`                                        | Ready-to-edit benchmark configs (`train_eval/`, `tuning/`, etc.)                 |
| `datasets/`                                       | Bundled datasets + download helper (`data_sources.yaml`)                         |
| `codes/`                                          | Python package with surrogates, training, tuning, and benchmarking utilities     |
| `run_training.py`, `run_eval.py`, `run_tuning.py` | CLI entry points for the main workflows                                          |
| `docs/`                                           | Sphinx project powering the GitHub Pages site (guides, tutorials, API reference) |
| `scripts/`                                        | Convenience tooling (dataset downloads, analysis utilities)                      |

## Contributing

Contribution guidelines are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

In short: open or pick an issue, make your changes in a branch, and submit a pull request with tests/docs updates as needed.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
