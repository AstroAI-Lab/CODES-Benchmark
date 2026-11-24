Primordial
==========

:doc:`Snapshot overview </benchmarks/snapshots/aa-reference-2026-11/index>` · :doc:`All snapshots </benchmarks/snapshots/index>`

**A&A reference configurations (November 2026)**

**Evaluation split** Validation (val) · **Timing GPU** NVIDIA A30 · **Workload** 41,100 predicted states per pass

Direct and ensemble metrics show mean ± sample standard deviation across seeds. Bold marks the best displayed value for each metric, including displayed ties. Numerical rankings describe this snapshot and protocol.

Direct prediction
-----------------

.. list-table::
   :header-rows: 1
   :class: benchmark-results
   :widths: 23 15 15 13 19 15

   * - Model
     - mLAE ↓ (dex)
     - LAE₉₉ ↓ (dex)
     - mRE ↓ (%)
     - Inference ↓ (µs/state)
     - Parameters (k)
   * - FullyConnected
     - **0.097 ± 0.003**
     - **0.712 ± 0.025**
     - **11.91 ± 0.34**
     - 0.342 ± 0.000
     - 895.8
   * - MultiONet
     - 0.193 ± 0.001
     - 1.269 ± 0.003
     - 24.75 ± 0.35
     - 0.232 ± 0.000
     - 498.0
   * - LatentNeuralODE
     - 0.668 ± 0.004
     - 3.018 ± 0.012
     - 80.94 ± 0.52
     - 0.441 ± 0.000
     - 1417.6
   * - LatentPoly
     - 0.884 ± 0.012
     - 3.521 ± 0.138
     - 91.77 ± 1.09
     - **0.110 ± 0.000**
     - 460.7

mLAE is mean absolute log error. LAE₉₉ is the 99th percentile of absolute log error. mRE is calculated as the median relative error within each seed.

Synchronized forward calls after one warm-up pass. Each run's total forward-call time is divided by validation trajectories × stored timesteps. The protocol uses model-specific batch sizes.

Five-model ensembles
--------------------

.. list-table::
   :header-rows: 1
   :class: benchmark-results
   :widths: 36 30 34

   * - Model
     - Ensemble mLAE ↓ (dex)
     - Error–uncertainty Pearson r ↑
   * - FullyConnected
     - **0.073 ± 0.001**
     - 0.471 ± 0.002
   * - MultiONet
     - 0.147 ± 0.001
     - **0.498 ± 0.012**
   * - LatentNeuralODE
     - 0.651 ± 0.005
     - 0.290 ± 0.017
   * - LatentPoly
     - 0.821 ± 0.021
     - 0.278 ± 0.014

Pearson r measures how ensemble spread tracks error magnitude.

Rollout
-------

Ordinary training · stored time grid · 10 stored steps per chunk. Entries are rollout mLAE in dex.

.. list-table::
   :header-rows: 1
   :class: benchmark-results
   :widths: 30 17 17 17 19

   * - Model
     - Seed 42
     - Seed 142
     - Seed 242
     - Median ↓
   * - FullyConnected
     - 0.246
     - 0.251
     - 0.246
     - **0.246**
   * - MultiONet
     - 0.471
     - 0.465
     - 0.481
     - 0.471
   * - LatentNeuralODE
     - 1.250
     - 1.269
     - 1.246
     - 1.250
   * - LatentPoly
     - 1.199
     - 1.422
     - 1.303
     - 1.303

Diagnostic plots
----------------

Absolute log-error distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/primordial/error-distribution.svg
   :alt: Smoothed error density versus absolute log error, with one line per surrogate.

   Each line averages the seed-level smoothed histograms for direct predictions, giving equal weight to each seed. Bins are evenly spaced in log space. The download records exact zeros separately.

Catastrophic-error recall
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/primordial/catastrophic-error-recall.svg
   :alt: Recall of the largest ensemble errors versus the percentage of estimates flagged, with one curve per surrogate and a diagonal baseline for random flagging.

   Flag individual predicted quantities in order of decreasing ensemble spread. For each surrogate and seed, catastrophic errors are at or above the 99th percentile of the ensemble-mean absolute log error. Lines show mean recall, shading shows the minimum and maximum across seeds, and the diagonal shows what random flagging would catch on average.

:download:`Per-seed plot data (JSON) <diagnostics/primordial/plot-data.json>`

Data and provenance
-------------------

:download:`Results bundle (JSON) <results.json>` · :download:`Per-seed metrics (CSV) <replicas.csv>`

`Run code at e3edee2 <https://github.com/AstroAI-Lab/CODES-Benchmark/tree/e3edee24f25247142bacd314fd4f8434b1aa62a5>`_ · `Dataset archive <https://doi.org/10.5281/zenodo.18018572>`_ · `A&A paper <https://doi.org/10.1051/0004-6361/202658937>`_

Timing hardware: NVIDIA A30, 24576 MiB, 620.70.06. PyTorch 2.14.1, CUDA 13.0. The JSON bundle includes the protocol, configuration hashes, evaluation settings, and full-precision seed results.
