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
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/robin-janssen"><img src="https://avatars.githubusercontent.com/u/82322346?v=4?s=100" width="100px;" alt="Robin Janssen"/><br /><sub><b>Robin Janssen</b></sub></a><br /><a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=robin-janssen" title="Code">💻</a> <a href="#content-robin-janssen" title="Content">🖋</a> <a href="#data-robin-janssen" title="Data">🔣</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=robin-janssen" title="Documentation">📖</a> <a href="#design-robin-janssen" title="Design">🎨</a> <a href="#example-robin-janssen" title="Examples">💡</a> <a href="#ideas-robin-janssen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-robin-janssen" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-robin-janssen" title="Maintenance">🚧</a> <a href="#research-robin-janssen" title="Research">🔬</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/pulls?q=is%3Apr+reviewed-by%3Arobin-janssen" title="Reviewed Pull Requests">👀</a> <a href="#tool-robin-janssen" title="Tools">🔧</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=robin-janssen" title="Tests">⚠️</a> <a href="#talk-robin-janssen" title="Talks">📢</a> <a href="#userTesting-robin-janssen" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://tobibu.github.io"><img src="https://avatars.githubusercontent.com/u/7574273?v=4?s=100" width="100px;" alt="Tobias Buck"/><br /><sub><b>Tobias Buck</b></sub></a><br /><a href="#ideas-TobiBu" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-TobiBu" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-TobiBu" title="Project Management">📆</a> <a href="#research-TobiBu" title="Research">🔬</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/pulls?q=is%3Apr+reviewed-by%3ATobiBu" title="Reviewed Pull Requests">👀</a> <a href="#talk-TobiBu" title="Talks">📢</a> <a href="#mentoring-TobiBu" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lorenzobranca"><img src="https://avatars.githubusercontent.com/u/57775402?v=4?s=100" width="100px;" alt="Lorenzo Branca"/><br /><sub><b>Lorenzo Branca</b></sub></a><br /><a href="#data-lorenzobranca" title="Data">🔣</a> <a href="#ideas-lorenzobranca" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-lorenzobranca" title="Research">🔬</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=lorenzobranca" title="Tests">⚠️</a> <a href="#talk-lorenzobranca" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Immi000"><img src="https://avatars.githubusercontent.com/u/100942429?v=4?s=100" width="100px;" alt="Immanuel Sulzer"/><br /><sub><b>Immanuel Sulzer</b></sub></a><br /><a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=Immi000" title="Code">💻</a> <a href="#content-Immi000" title="Content">🖋</a> <a href="#data-Immi000" title="Data">🔣</a> <a href="#design-Immi000" title="Design">🎨</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=Immi000" title="Documentation">📖</a> <a href="#infra-Immi000" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#userTesting-Immi000" title="User Testing">📓</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
