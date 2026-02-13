# Tutorials

Interactive tutorials live in this folder so they can be rendered directly on the documentation site via `myst-nb`. Use them when you want a guided, executable tour of the benchmark.

## Available tutorials

| Notebook | Description |
| --- | --- |
| [Benchmark Quickstart](benchmark_quickstart.ipynb) | Walks through loading a configuration file, running a short training job, and visualizing KPI tables. |

To execute a notebook locally:

```bash
uv pip install --group dev  # installs docs + notebook deps
jupyter lab docs/source/tutorials/benchmark_quickstart.ipynb
```

When building the docs (`sphinx-build`), notebooks are parsed but not executed. You can enable execution later by setting `nb_execution_mode = "auto"` in `docs/source/conf.py`.

```{toctree}
:maxdepth: 1

benchmark_quickstart
```
