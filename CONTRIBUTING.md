# Contributing to CODES Benchmark

Thanks for helping improve CODES Benchmark. Contributions of all sizes are welcome, including bug fixes, new features, documentation updates, and benchmark extensions.

## What to contribute

Common contribution types in this repository are:

- Adding a new dataset
- Adding a new surrogate model
- Adding or improving benchmark evaluations and modalities

For extension-specific implementation details, follow the dedicated guide:

- https://astroai-lab.de/CODES-Benchmark/guides/extending-benchmark.html

## Development setup

Use either `uv` (recommended) or a standard `venv` + `pip` setup.

### Option A: uv (recommended)

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
uv sync
source .venv/bin/activate
uv pip install --group dev
```

### Option B: venv + pip

```bash
git clone https://github.com/robin-janssen/CODES-Benchmark.git
cd CODES-Benchmark
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install black isort pytest sphinx
```

## Code style and formatting

This repository uses:

- [Black](https://black.readthedocs.io/en/stable/) for formatting
- [isort](https://pycqa.github.io/isort/) for import sorting

The CI pipeline auto-applies both tools on contributions. Running them locally is still recommended so you get fast feedback before opening or updating a pull request.

Run both locally with:

```bash
black .
isort .
```

If you use pre-commit, install hooks once and run them across the repository:

```bash
pre-commit install
pre-commit run --all-files
```

## Tests and documentation checks

Before submitting a PR, run:

```bash
pytest
sphinx-build -b html docs/source docs/_build/html
```

If your change affects user-facing behavior, update the relevant docs in `docs/source/`.

## Standard contribution workflow

1. Create an issue (bug report, feature request, or extension proposal), or choose an open issue to work on.
2. Create a branch from `main` with a clear name.
3. Implement your changes and include tests/docs updates where relevant.
4. Run formatting, tests, and docs build locally.
5. Open a pull request and link the issue.
6. Address review feedback and keep your branch up to date.

## Pull request expectations

Please keep PRs focused and easy to review.

- Explain what changed and why.
- Link related issues.
- Include tests for behavior changes.
- Include documentation updates for new datasets, surrogates, modalities, or config options.

## Need help?

If you are unsure where to start, open an issue with your proposal or question.
