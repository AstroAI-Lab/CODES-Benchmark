# CODES Benchmark

[![codecov](https://codecov.io/github/AstroAI-Lab/CODES-Benchmark/branch/main/graph/badge.svg)](https://codecov.io/github/AstroAI-Lab/CODES-Benchmark) ![Static Badge](https://img.shields.io/badge/license-GPLv3-blue) ![Static Badge](https://img.shields.io/badge/NeurIPS-2024-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23810176.svg)](https://doi.org/10.5281/zenodo.23810176)
[![All Contributors](https://img.shields.io/github/all-contributors/AstroAI-Lab/CODES-Benchmark?color=ee8449&style=flat-square)](#contributors)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

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
git clone https://github.com/AstroAI-Lab/CODES-Benchmark.git
cd CODES-Benchmark
uv sync                       # creates .venv from pyproject/uv.lock
source .venv/bin/activate
uv run python run_training.py --config configs/train_eval/config_minimal.yaml
uv run python run_eval.py --config configs/train_eval/config_minimal.yaml
```

**pip alternative**

```bash
git clone https://github.com/AstroAI-Lab/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
python run_training.py --config configs/train_eval/config_minimal.yaml
python run_eval.py --config configs/train_eval/config_minimal.yaml
```

Outputs land in `trained/<training_id>`, `results/<training_id>`, and `plots/<training_id>`. The `configs/` folder contains ready-to-use templates (`train_eval/config_minimal.yaml`, `config_full.yaml`, etc.). Copy a file there and adjust datasets/surrogates/modalities before running the CLIs.

## Documentation

- [Main docs & tutorials](https://astroai-lab.de/CODES-Benchmark/)
- [API reference (Sphinx)](https://astroai-lab.de/CODES-Benchmark/api-reference.html)

The GitHub Pages site now hosts the narrative guides, configuration reference, and interactive notebooks alongside the generated API docs.

## Repository map

| Path                                              | Purpose                                                                          |
| ------------------------------------------------- | -------------------------------------------------------------------------------- |
| `configs/`                                        | Ready-to-edit benchmark configs (`train_eval/`, `tuning/`, etc.)                 |
| `datasets/`                                       | Bundled datasets + download helper (`data_sources.yaml`)                         |
| `codes/`                                          | Python package with surrogates, training, tuning, and benchmarking utilities     |
| `run_training.py`, `run_eval.py`, `run_tuning.py` | CLI entry points for the main workflows                                          |
| `run_finetuning.py`                               | Optional: finetune trained surrogates for autoregressive rollout                 |
| `docs/`                                           | Sphinx project powering the GitHub Pages site (guides, tutorials, API reference) |
| `test/`                                           | Unit and integration tests                                                        |

## Versions

CODES follows [semantic versioning](https://semver.org/). [CHANGELOG.md](CHANGELOG.md) lists the changes in each release, including every change that affects benchmark results. Saved models and metrics record the CODES version and commit that produced them.

Development builds for the next release use `0.3.0.dev0`. When using an unreleased
build, record the git commit as well as the package version.

## Citation & Publications using CODES

If you use `CODES` in your research, please cite the archived software release you
used. [CODES 0.2.0](https://doi.org/10.5281/zenodo.23810177) is archived on Zenodo;
you can export its citation there or use GitHub's **Cite this repository** button,
which reads [CITATION.cff](CITATION.cff). Include the version used in your methods.
The DOI badge above links to [all archived versions](https://doi.org/10.5281/zenodo.23810176).

If you use the astrochemical datasets or analysis presented in **Janssen, Branca &
Buck (2026)**, please also cite that paper, listed below.

### Related works using CODES

Robin Janssen, Immanuel Sulzer, Tobias Buck  
**CODES: Benchmarking Coupled ODE Surrogates**  
*NeurIPS ML4PS Workshop, 2024*  
Initial publication of the CODES framework.  
[arXiv](https://arxiv.org/abs/2410.20886) | [bib](https://arxiv.org/bibtex/2410.20886)


Robin Janssen, Lorenzo Branca, Tobias Buck  
**Systematic selection of surrogate models for nonequilibrium chemistry**  
*Astronomy & Astrophysics, 2026*  
Extensive application of CODES to four challenging astrochemical datasets, demonstrating practical model selection strategies for stiff chemical ODE systems.  
[arXiv](https://arxiv.org/abs/2603.08567) | [bib](https://arxiv.org/bibtex/2603.08567)  

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Prime-73"><img src="https://avatars.githubusercontent.com/u/77167261?v=4?s=100" width="100px;" alt="Gianluca Möschter"/><br /><sub><b>Gianluca Möschter</b></sub></a><br /><a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=Prime-73" title="Code">💻</a> <a href="#data-Prime-73" title="Data">🔣</a> <a href="#research-Prime-73" title="Research">🔬</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=Prime-73" title="Documentation">📖</a> <a href="https://github.com/AstroAI-Lab/CODES-Benchmark/commits?author=Prime-73" title="Tests">⚠️</a> <a href="#userTesting-Prime-73" title="User Testing">📓</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
