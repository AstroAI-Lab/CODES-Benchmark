# CODES Benchmark

[![codecov](https://codecov.io/github/robin-janssen/CODES-Benchmark/branch/main/graph/badge.svg?token=TNF9ISCAJK)](https://codecov.io/github/robin-janssen/CODES-Benchmark)
![Static Badge](https://img.shields.io/badge/license-GPLv3-blue)
![Static Badge](https://img.shields.io/badge/NeurIPS-2024-green)
[![All Contributors](https://img.shields.io/github/all-contributors/AstroAI-Lab/CODES-Benchmark?color=ee8449&style=flat-square)](#contributors)

🎉 CODES was accepted to the ML4PS workshop @ NeurIPS2024 🎉

## Benchmarking Coupled ODE Surrogates

CODES is a benchmark for coupled ODE surrogate models.

<picture>
  <!-- Dark mode SVG -->
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/file-alt-solid-white.svg">
  <!-- Light mode SVG -->
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/file-alt-solid.svg">
  <!-- Fallback image (light mode by default) -->
  <img width="14" alt="Paper on arXiv" src="docs/_static/book-solid.svg">
</picture> CODES paper on <a href="https://arxiv.org/abs/2410.20886">arXiV</a>. <p></p>

<picture> 
<source srcset="docs/_static/favicon-96x96.png">
<img width="15" alt="CODES Logo" src="docs/_static/favicon-96x96.png">
</picture> The main documentation can be found on the <a href="https://codes-docs.web.app/index.html">CODES website</a>. <p></p>

<picture>
  <!-- Dark mode SVG -->
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/book-solid-white.svg">
  <!-- Light mode SVG -->
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/book-solid.svg">
  <!-- Fallback image (light mode by default) -->
  <img width="14" alt="CODES API Docs" src="docs/_static/book-solid.svg">
</picture>  The technical API documentation is hosted on this <a href="https://robin-janssen.github.io/CODES-Benchmark/">GitHub Page</a>.

## Motivation

There are many efforts to use machine learning models ("surrogates") to replace the costly numerics involved in solving coupled ODEs. But for the end user, it is not obvious how to choose the right surrogate for a given task. Usually, the best choice depends on both the dataset and the target application.

Dataset specifics - how "complex" is the dataset?

- How many samples are there?
- Are the trajectories very dynamic or are the developments rather slow?
- How dense is the distribution of initial conditions?
- Is the data domain of interest well-covered by the domain of the training set?

Task requirements:

- What is the required accuracy?
- How important is inference time? Is the training time limited?
- Are there computational constraints (memory or processing power)?
- Is uncertainty estimation required (e.g. to replace uncertain predictions by numerics)?
- How much predictive flexibility is required? Do we need to interpolate or extrapolate across time?

Besides these practical considerations, one overarching question is always: Does the model only learn the data, or does it "understand" something about the underlying dynamics?

## Goals

This benchmark aims to aid in choosing the best surrogate model for the task at hand and additionally to shed some light on the above questions.

To achieve this, a selection of surrogate models are implemented in this repository. They can be trained on one of the included datasets or a custom dataset and then benchmarked on the corresponding test dataset.

Some **metrics** included in the benchmark (but there is much more!):

- Absolute and relative error of the models.
- Inference time.
- Number of trainable parameters.
- Memory requirements (**WIP**).

Besides this, there are plenty of **plots and visualisations** providing insights into the models behaviour:

- Error distributions - per model, across time or per quantity.
- Insights into interpolation and extrapolation across time.
- Behaviour when training with sparse data or varying batch size.
- Predictions with uncertainty and predictive uncertainty across time.
- Correlations between the either predictive uncertainty or dynamics (gradients) of the data and the prediction error

Some prime **use-cases** of the benchmark are:

- Finding the best-performing surrogate on a dataset. Here, best-performing could mean high accuracy, low inference times or any other metric of interest (e.g. most accurate uncertainty estimates, ...).
- Comparing performance of a novel surrogate architecture against the implemented baseline models.
- Gaining insights into a dataset or comparing datasets using the built-in dataset insights.

## Key Features

<details>
  <summary><b>Baseline Surrogates</b></summary>

The following surrogate models are currently implemented to be benchmarked:

- Fully Connected Neural Network:
  The vanilla neural network a.k.a. multilayer perceptron.
- DeepONet:
  Two fully connected networks whose outputs are combined using a scalar product. In the current implementation, the surrogate comprises of only one DeepONet with multiple outputs (hence the name MultiONet).
- Latent NeuralODE:
  NeuralODE combined with an autoencoder that reduces the dimensionality of the dataset before solving the dynamics in the resulting latent space.
- Latent Polynomial:
  Uses an autoencoder similar to Latent NeuralODE, but fits a polynomial to the trajectories in the resulting latent space.

</details>

<details>
  <summary><b>Baseline Datasets</b></summary>

The following datasets are currently included in the benchmark:

</details>

<details>
  <summary><b>Uncertainty Quantification (UQ)</b></summary>

To give an uncertainty estimate that does not rely too much on the specifics of the surrogate architecture, we use DeepEnsemble for UQ.

</details>

<details>
  <summary><b>Parallel Training</b></summary>

To gain insights into the surrogates behaviour, many models must be trained on varying subsets of the training data. This task is trivially parallelisable. In addition to utilising all specified devices, the benchmark features some nice progress bars to gain insights into the current status of the training.

</details>

<details>
  <summary><b>Plots, Plots, Plots</b></summary>

While hard metrics are crucial to compare the surrogates, performance cannot always be broken down to a set of numbers. Running the benchmark creates many plots that serve to compare performance of surrogates or provide insights into the performance of each surrogate.

</details>

<details>
  <summary><b>Dataset Insights (WIP)</b></summary>

"Know your data" is one of the most important rules in machine learning. To aid in this, the benchmark provides plots and visualisations that should help to understand the dataset better.

</details>

<details>
  <summary><b>Tabular Benchmark Results</b></summary>

At the end of the benchmark, the most important metrics are displayed in a table, additionally, all metrics generated during the benchmark are provided as a csv file.

</details>

<details>
  <summary><b>Reproducibility</b></summary>

Randomness is an important part of machine learning and even required in the context of UQ with DeepEnsemble, but reproducibility is key in benchmarking enterprises. The benchmark uses a custom seed that can be set by the user to ensure full reproducibility.

</details>

<details>
  <summary><b>Custom Datasets and Own Models</b></summary>

To cover a wide variety of use-cases, the benchmark is designed such that adding own datasets and models is explicitly supported.

</details>

## Quickstart

First, clone the [GitHub Repository](https://github.com/robin-janssen/CODES-Benchmark) with

```
git clone ssh://git@github.com/robin-janssen/CODES-Benchmark
```

Optionally, you can set up a [virtual environment](https://docs.python.org/3/library/venv.html) (recommended).

Then, install the required packages with

```
pip install -r requirements.txt
```

The installation is now complete. To be able to run and evaluate the benchmark, you need to first set up a configuration YAML file. There is one provided, but it should be configured. For more information, check the [configuration page](https://robin-janssen.github.io/CODES-Benchmark/documentation.html#config). There, we also offer an interactive Config-Generator tool with some explanations to help you set up your benchmark.

You can also add your own datasets and models to the benchmark to evaluate them against each other or some of our baseline models. For more information on how to do this, please refer to the [documentation](https://robin-janssen.github.io/CODES-Benchmark/documentation.html).


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->