Cloud
=====

:doc:`Snapshot overview </benchmarks/snapshots/aa-reference-2026-11/index>` · :doc:`All snapshots </benchmarks/snapshots/index>`

**A&A reference configurations (November 2026)**

**Evaluation split** Validation (val) · **Timing GPU** NVIDIA A30 · **Workload** 163,900 predicted states per pass

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
     - **0.021 ± 0.001**
     - **0.281 ± 0.007**
     - **1.12 ± 0.05**
     - **0.038 ± 0.005**
     - 24.2
   * - MultiONet
     - 0.034 ± 0.003
     - 0.329 ± 0.020
     - 4.03 ± 0.32
     - 0.263 ± 0.004
     - 458.6
   * - LatentNeuralODE
     - 1.914 ± 0.000
     - 4.538 ± 0.001
     - 99.96 ± 0.00
     - 0.249 ± 0.030
     - 286.2
   * - LatentPoly
     - 1.647 ± 0.022
     - 4.810 ± 0.214
     - 99.86 ± 0.01
     - 0.062 ± 0.010
     - 192.6

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
     - **0.015 ± 0.001**
     - **0.728 ± 0.025**
   * - MultiONet
     - 0.024 ± 0.000
     - 0.645 ± 0.011
   * - LatentNeuralODE
     - 1.914 ± 0.000
     - 0.172 ± 0.057
   * - LatentPoly
     - 1.605 ± 0.004
     - 0.059 ± 0.024

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
     - 0.263
     - 132.386
     - 0.417
     - 0.417
   * - MultiONet
     - 0.168
     - 0.172
     - 0.150
     - **0.168**
   * - LatentNeuralODE
     - 1.914
     - 1.914
     - 1.914
     - 1.914
   * - LatentPoly
     - 1.759
     - 1.708
     - 1.770
     - 1.759

.. note:: FullyConnected seed 142 diverged during repeated prediction (132.386 dex). All three FullyConnected seeds also overflowed the real-space rollout error summaries. The log-space results remain visible.

Diagnostic plots
----------------

Absolute log-error distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/cloud/error-distribution.svg
   :alt: Smoothed error density versus absolute log error, with one line per surrogate.

   Each line averages the seed-level smoothed histograms for direct predictions, giving equal weight to each seed. Bins are evenly spaced in log space. The download records exact zeros separately.

Catastrophic-error recall
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/cloud/catastrophic-error-recall.svg
   :alt: Recall of the largest ensemble errors versus the percentage of estimates flagged, with one curve per surrogate and a diagonal baseline for random flagging.

   Flag individual predicted quantities in order of decreasing ensemble spread. For each surrogate and seed, catastrophic errors are at or above the 99th percentile of the ensemble-mean absolute log error. Lines show mean recall, shading shows the minimum and maximum across seeds, and the diagonal shows what random flagging would catch on average.

:download:`Per-seed plot data (JSON) <diagnostics/cloud/plot-data.json>`

Data and provenance
-------------------

:download:`Results bundle (JSON) <results.json>` · :download:`Per-seed metrics (CSV) <replicas.csv>`

`Run code at e3edee2 <https://github.com/AstroAI-Lab/CODES-Benchmark/tree/e3edee24f25247142bacd314fd4f8434b1aa62a5>`_ · `Dataset archive <https://doi.org/10.5281/zenodo.18018572>`_ · `A&A paper <https://doi.org/10.1051/0004-6361/202658937>`_

Timing hardware: NVIDIA A30, 24576 MiB, 620.70.06. PyTorch 2.14.1, CUDA 13.0. The JSON bundle includes the protocol, configuration hashes, evaluation settings, and full-precision seed results.
