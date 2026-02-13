# API Reference Overview

The API docs are auto-generated from the `codes` package using Sphinx autodoc. Use the links below to jump directly into each area:

- {doc}`codes.benchmark` — command-line entry points, benchmarking utilities, plotting helpers, and dataset utilities.
- {doc}`codes.surrogates` — shared base classes plus module pages for every surrogate family (MultiONet, FCNN, LatentNeuralODE, LatentPoly, and helpers). Each surrogate package exposes its config dataclass, training loop, and evaluation helpers.
- {doc}`codes.train` — task orchestration, queueing logic, and progress reporting for the training CLI.
- {doc}`codes.tune` — Optuna integration, objective definitions, database utilities, and plotting for study analysis.
- {doc}`codes.utils` — shared utilities for loading datasets, normalization, checkpointing, and miscellaneous helpers reused throughout the framework.

Each page mirrors the Python package tree and lists modules in alphabetical order. Collapse/expand sections in the sidebar to navigate classes, functions, and dataclasses quickly. When extending the framework, start from `codes.surrogates` to see how existing models integrate with `AbstractSurrogateModel`, or browse `codes.benchmark` to discover which helpers power the CLI scripts.
