---
title: 'CODES: A benchmark framework for machine-learned surrogates of coupled ordinary differential equations'
tags:
  - Python
  - PyTorch
  - surrogate models
  - ordinary differential equations
  - benchmarking
  - astrochemistry
  - neural operators
authors:
  # TODO: confirm author list, order, ORCIDs and affiliations with all co-authors.
  - name: Robin Janssen
    orcid: 0009-0006-4356-3059
    corresponding: true
    affiliation: "1, 2"
  - name: Immanuel Sulzer
    affiliation: "1, 2"
  - name: Gianluca Möschter
    affiliation: "1, 2"
  - name: Lorenzo Branca
    orcid: 0000-0002-6064-1964
    affiliation: "1, 2"
  - name: Tobias Buck
    orcid: 0000-0003-2027-399X
    affiliation: "1, 2"
affiliations:
  # TODO: check affiliation names against each author's preferred form.
  - name: Interdisciplinary Center for Scientific Computing (IWR), Heidelberg University, Germany
    index: 1
    ror: 038t36y30
  - name: Institut für Theoretische Astrophysik, Zentrum für Astronomie, Heidelberg University, Germany
    index: 2
date: 31 October 2026
bibliography: paper.bib
---

# Summary

Many problems in physics, chemistry and engineering are described by systems of
coupled ordinary differential equations (ODEs). A typical example is a chemical
reaction network: each equation tracks how the abundance of one species changes
over time as it is created and destroyed by reactions with the others. Solving
such systems accurately is often expensive, because the equations are *stiff*:
some quantities change millions of times faster than others, forcing numerical
solvers to take tiny steps. When a simulation of a galaxy, a planet-forming
disk or a combustion engine must solve such a system in every cell and every
time step, the chemistry alone can dominate the total cost.

Machine-learned *surrogates*, neural networks trained to mimic the solver, can
be orders of magnitude faster. Many architectures have been proposed, but they
are rarely compared under the same conditions, so it is hard to know which one
to use for a given problem. `CODES` (Coupled ODE Surrogates) is a Python
package that trains and evaluates such surrogates on equal footing. From a
single configuration file, it tunes, trains and evaluates several surrogate
architectures on the same data. It then reports not only how accurate each one
is, but also how fast it is, how much memory it needs (measured by its
parameter count and memory footprint), how it behaves outside the training
regime, how much data it needs and how reliable its uncertainty estimates are.

# Statement of need

Surrogate models for coupled ODEs are increasingly used to accelerate
simulations in astrophysics and astrochemistry
[@Grassi:2022; @Sulzer:2023; @Branca:2023; @Branca:2024], as well as in other
fields with stiff kinetics. The literature proposes a wide range of
architectures, including fully connected networks, operator networks such as
DeepONet [@Lu:2021], latent neural ODEs [@Chen:2018; @Rubanova:2019] and
autoencoders with simple latent dynamics
[@Grassi:2022; @Sulzer:2023; @Nockolds:2025]. Each of
these is usually published with its own data pipeline, normalisation choices,
training budget and error metric. Results therefore cannot be compared across
papers, and practitioners who need a surrogate for a new chemical network have
no systematic way to choose one.

`CODES` addresses this gap. It is aimed at two audiences:

- **Domain scientists** who have a set of ODE trajectories (for example from a
  chemistry code such as KROME [@Grassi:2014], UCLCHEM [@Holdship:2017] or the
  JAX-based Carbox [@Vermarien:2025]) and
  want to know which surrogate meets their accuracy target within their runtime
  budget.
- **Machine-learning researchers** who develop new surrogate architectures and
  need a fair, reproducible baseline comparison on realistic, stiff,
  high-dimensional data.

In both cases, `CODES` provides the same data splits, preprocessing,
hyperparameter optimisation budget, random seeds and metrics for every model,
so that differences in the results reflect the models rather than the
experimental setup.

# State of the field

Several benchmark suites exist for machine learning on differential equations.
PDEBench [@Takamoto:2022] and The Well [@Ohana:2024] collect datasets for
partial differential equations on spatial grids and focus on architectures such
as Fourier neural operators and U-Nets. These suites do not cover the setting
`CODES` targets: zero-dimensional systems with many coupled, stiff
quantities that span many orders of magnitude, often conditioned on
physical parameters, and evaluated with a view to replacing a solver inside a
larger simulation. Libraries for neural differential equations, such as
`torchode` [@Lienen:2022] or those surveyed by @Kidger:2022, provide building
blocks for individual models, but not a comparison framework.

Within astrochemistry, published surrogates
[@Grassi:2022; @Sulzer:2023; @Branca:2023; @Branca:2024] each come with
model-specific code. We chose to build a dedicated framework rather than extend
one of these codes because a fair comparison requires the training loop,
data handling and evaluation to live *outside* any single model. Retrofitting a
model-specific code base would have tied the benchmark to that model's
assumptions about input layout and time handling. `CODES` instead defines a
common interface that every surrogate implements, so that existing and new
architectures are evaluated by identical code.

# Software design

\autoref{fig:workflow} shows the structure of `CODES`. The central design
decision is the separation between *surrogates* and the *benchmark
infrastructure*. Every surrogate inherits from `AbstractSurrogateModel`, which
fixes a small interface: `prepare_data` builds data loaders, `forward` maps
inputs to predictions, and `fit` implements the training loop. Shared methods
for prediction, validation, checkpointing, normalisation and optimiser and
scheduler construction are inherited. A model registers itself under a name
and then becomes available in every stage of the workflow. The package
currently ships four surrogate families: MultiONet (a multi-output DeepONet
variant), a fully connected network, a latent neural ODE built on `torchode`
[@Lienen:2022] and a latent polynomial model. It also includes residual
variants of the first two. We deliberately allow each model to own its data
layout in `prepare_data`, because operator networks, recurrent latent models
and direct regressors consume trajectories in incompatible ways. Forcing a
single tensor format would have handicapped some architectures. Consistency is
instead enforced at the output: `predict` always returns arrays of the same
shape, and all metrics are computed from these.

![The `CODES` workflow. A single YAML configuration selects the dataset and
surrogates and drives hyperparameter tuning, training, optional rollout
finetuning and evaluation. All surrogates implement the same abstract
interface, so every stage treats them identically.\label{fig:workflow}](figures/workflow.png){ width=100% }

The workflow has three main stages, each with its own command-line entry point
and all driven by YAML configuration files:

1. **Tuning.** Hyperparameters are optimised with Optuna [@Akiba:2019], with
   SQLite or PostgreSQL storage so that studies can run on several GPUs or
   nodes. Studies can optimise accuracy alone or accuracy and inference time
   together (as a multi-objective study), which reflects the practical trade-off when a surrogate is to be
   embedded in a simulation.
2. **Training.** For each requested *modality*, a set of models is trained in
   parallel worker processes with fixed seeds, so that runs are reproducible.
   The modalities are interpolation and extrapolation in time, sparse training
   data, batch-size scaling and deep ensembles for uncertainty quantification
   [@Lakshminarayanan:2017].
3. **Evaluation.** The trained models are evaluated with the same metrics,
   including absolute and relative errors in log and linear space, inference
   time, parameter count and memory footprint, the correlation between errors
   and the local gradients of the data, behaviour under each modality, and the
   correlation between predicted ensemble uncertainty and actual error. Optionally, models are fine-tuned
   for autoregressive rollout and evaluated on long rollouts.

`CODES` writes the results as tables and plots that compare all surrogates in
a single report.

Datasets are stored as HDF5 files with fixed train, validation and test splits,
optional per-trajectory parameters and a shared time grid. A helper function
creates this layout from NumPy arrays, so users can add their own data without
touching the benchmark code. Sixteen curated datasets, from simple
Lotka–Volterra and oscillator systems to stiff astrochemical networks, are available from Zenodo. The software is built on PyTorch
[@Paszke:2019] and supports schedule-free optimisers [@Defazio:2024]. It has
unit and end-to-end tests run in continuous integration, follows semantic
versioning, and records in each saved model and result the `CODES` version and
commit that produced it.

# Research impact statement

`CODES` was first presented at the Machine Learning and the Physical Sciences
workshop at NeurIPS 2024 [@Janssen:2024]. That contribution focused on the
machine-learning results, comparing the accuracy of different surrogate
architectures for astrochemistry, and not on the software itself. The
framework has since been used for a systematic study of surrogate selection
for non-equilibrium astrochemistry, published in *Astronomy & Astrophysics*
[@Janssen:2026]. That study trained and compared all surrogate families on four
astrochemical datasets generated with KROME and derived practical guidance on
which architecture to choose for which kind of problem.

`CODES` is also used beyond these two papers. Gianluca Möschter developed the
autoregressive finetuning and rollout evaluation in his bachelor's thesis at
Heidelberg University.
<!-- TODO: cite G. Möschter's paper once it is on arXiv. -->
Lorenzo Branca uses `CODES` to develop the chemistry emulators of the
astronomix code (Branca et al., in preparation).
<!-- TODO: cite L. Branca's astronomix emulator paper once it is on arXiv.
TODO: add any further external users or citing works (check Google Scholar /
ADS citations of the repository, Janssen et al. 2024 and Janssen et al. 2026). -->

All datasets used in these works are publicly archived on Zenodo and can be
downloaded with a script included in the repository, so the published
comparisons can be reproduced and extended to new architectures. The
repository has been developed openly since May 2024, with tagged releases,
documentation hosted at <https://astroai-lab.de/CODES-Benchmark/> and
contributions from developers outside the core team.

# AI usage disclosure

<!-- TODO: fill in the model name and version, and confirm the list of
AI-assisted pull requests. -->
`CODES` was developed without generative AI assistance. The architecture, the
benchmark design and almost all of the code and documentation, up to and
including release 0.2.0, were written by hand, primarily by Robin Janssen.
Only the most recent pull requests (October 2026, after release 0.2.0) used
an AI assistant, Claude Code (Anthropic; model and version: TODO), for code
review and bug fixing; every AI-suggested change was reviewed, tested and
accepted by the human authors. Claude Code was also used to prepare the first
draft of this paper from the repository contents, which the authors then
revised and verified. The human authors made all design decisions and take
full responsibility for the software and the paper.

# Acknowledgements

<!-- TODO: confirm the DFG grant number and official name of the COSMOS
project; check all funder wording against the funders' current guidance. -->
We thank all contributors to the `CODES` repository. This work is funded by
the Carl-Zeiss-Stiftung through the CZS Nexus programme. We further acknowledge
funding by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) through the COSMOS project (project number TODO) and under
Germany's Excellence Strategy EXC 2181/1 - 390900948 (the Heidelberg
STRUCTURES Excellence Cluster). The authors gratefully acknowledge the
computing time made available to them on the high-performance computer HoreKa
at the NHR Center KIT. This center is jointly supported by the Federal
Ministry of Research, Technology and Space and the Ministry of Science,
Research and the Arts of Baden-Württemberg, as part of the National
High-Performance Computing (NHR) joint funding program
(<https://www.nhr-verein.de/en/our-partners>). HoreKa is partly funded by the
German Research Foundation (DFG).

# References
