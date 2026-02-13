# Running Benchmarks

This hub explains how the benchmark orchestrator works end to end. A typical run looks like this:

1. **Hyperparameter tuning** — Optuna samples candidate configs per surrogate and stores the trials in a database. You decide which trials become new defaults (single- or multi-objective).
2. **Training** — `run_training.py` reads your benchmark config, schedules “main” runs + optional modalities, and produces checkpoints under `trained/<training_id>/`.
3. **Evaluation** — `run_eval.py` reloads those checkpoints, applies whichever evaluation suites you enabled, and writes structured metrics/plots.

Two supporting sections describe the **baseline architectures** that ship with CODES and the **modalities** that expand each training run.

```{admonition} TL;DR
Follow the links below chronologically when learning the system; jump directly to modalities or architectures when you need implementation specifics.
```

```{toctree}
:maxdepth: 1

tuning
training
evaluation
modalities
architectures
accuracy-metrics
uncertainty
```
