# Getting Started

CODES Benchmark helps you compare surrogate models for coupled ODE systems by running consistent training, evaluation, and reporting pipelines. This page summarizes the minimum you need to install the project, configure a run, and validate that everything is wired correctly.

## Prerequisites

- Python 3.10 with `pip` (we recommend `uv` for faster installs, but plain pip works)
- (Optional) CUDA-capable GPU if you want to reproduce the default configurations
- Enough disk space to store downloaded datasets and the checkpoints created under `trained/` and `results/`

## Installation

We recommend [`uv`](https://docs.astral.sh/uv/) because the repository already ships with `pyproject.toml` + `uv.lock`. Cloning and syncing is enough—`uv run` will automatically create/update the environment the first time you execute a command.

**uv workflow (recommended)**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
uv sync            # creates .venv using uv.lock
uv run python -c "import codes; print('CODES ready!')"  # optional smoke test
```

After this, you can prefix any command with `uv run` (for example `uv run python run_training.py --config config.yaml`) and uv will ensure dependencies are in place.

**pip / virtualenv fallback**

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Both approaches expose the `codes` package locally so scripts such as `run_training.py` and `run_eval.py` can import it without extra path hacks. Installing the pinned requirements ensures feature parity with CI and the documentation build.

## Configure a run

Benchmarks usually follow the pattern: **hyperparameter tuning → full training → evaluation/reporting**. Before diving into the advanced knobs, start from the minimal configuration we ship in `config.yaml`:

```yaml
training_id: "my_first_benchmark"
surrogates: ["MultiONet"]
batch_size: [65536]
epochs: [200]
dataset:
  name: "osu2008"
devices: ["cuda:0"] # or ["cpu"] if you lack a GPU
```

1. Copy `config.yaml` (or `config_full.yaml` for inspiration) to a new name inside the repo.
2. Update the surrogate list, dataset, devices, and optional study switches such as `interpolation`, `extrapolation`, `sparse`, `batch_scaling`, or `uncertainty`.
3. Keep the file under version control so you can trace results.

See the [configuration reference](reference/configuration.md) for a complete list of keys, defaults, and tips.

## Run your first benchmark

Use the minimal configuration above (or a copy of it) to perform a smoke test:

1. **Train** the requested surrogate (creates `trained/<training_id>`):
   ```bash
   python run_training.py --config path/to/your_config.yaml
   ```
2. **Evaluate / benchmark** everything that was trained:
   ```bash
   python run_eval.py --config path/to/your_config.yaml
   ```
3. Inspect the generated tables under `results/<training_id>` and the plots inside `plots/<training_id>`.

Every CLI script honours the `--config` flag and logs progress to the console.

## Where to go next

- [Running benchmarks](guides/running-benchmarks/index.md): deep-dive into the workflow, multi-device execution, and troubleshooting.
- [Configuration reference](reference/configuration.md): every knob explained.
- [Dataset catalog](reference/datasets.md): discover the bundled datasets and download URLs.
- [Tutorials](tutorials/index.md): notebooks that demonstrate data loading, custom analysis, and plotting pipelines.
