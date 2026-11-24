Primordial Parametric
=====================

:doc:`Snapshot overview </benchmarks/snapshots/aa-reference-2026-11/index>` · :doc:`All snapshots </benchmarks/snapshots/index>`

**A&A reference configurations (November 2026)**

**Evaluation split** Validation (val) · **Timing GPU** NVIDIA A30 · **Workload** 82,000 predicted states per pass

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
     - **0.115 ± 0.003**
     - **0.942 ± 0.019**
     - **12.50 ± 0.27**
     - 0.189 ± 0.000
     - 454.0
   * - MultiONet
     - 0.174 ± 0.007
     - 1.221 ± 0.063
     - 21.61 ± 0.92
     - 0.099 ± 0.000
     - 86.4
   * - LatentNeuralODE
     - 0.710 ± 0.019
     - 3.510 ± 0.030
     - 82.60 ± 1.52
     - 0.478 ± 0.005
     - 409.2
   * - LatentPoly
     - 0.845 ± 0.012
     - 3.969 ± 0.166
     - 88.27 ± 0.64
     - **0.041 ± 0.000**
     - 26.6

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
     - **0.090 ± 0.001**
     - 0.493 ± 0.002
   * - MultiONet
     - 0.126 ± 0.002
     - **0.522 ± 0.005**
   * - LatentNeuralODE
     - 0.678 ± 0.012
     - 0.345 ± 0.035
   * - LatentPoly
     - 0.755 ± 0.011
     - 0.401 ± 0.013

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
     - 0.345
     - 0.403
     - 0.333
     - **0.345**
   * - MultiONet
     - 6.563
     - 0.420
     - 0.412
     - 0.420
   * - LatentNeuralODE
     - 1.535
     - 1.548
     - 1.517
     - 1.535
   * - LatentPoly
     - 2.012
     - 1.507
     - 1.751
     - 1.751

.. note:: MultiONet seed 42 diverged during repeated prediction (6.563 dex). Real-space rollout error summaries overflowed for this run and for FullyConnected seed 142.

Diagnostic plots
----------------

Absolute log-error distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/primordial_parametric/error-distribution.svg
   :alt: Smoothed error density versus absolute log error, with one line per surrogate.

   Each line averages the seed-level smoothed histograms for direct predictions, giving equal weight to each seed. Bins are evenly spaced in log space. The download records exact zeros separately.

Catastrophic-error recall
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/primordial_parametric/catastrophic-error-recall.svg
   :alt: Recall of the largest ensemble errors versus the percentage of estimates flagged, with one curve per surrogate and a diagonal baseline for random flagging.

   Flag individual predicted quantities in order of decreasing ensemble spread. For each surrogate and seed, catastrophic errors are at or above the 99th percentile of the ensemble-mean absolute log error. Lines show mean recall, shading shows the minimum and maximum across seeds, and the diagonal shows what random flagging would catch on average.

:download:`Per-seed plot data (JSON) <diagnostics/primordial_parametric/plot-data.json>`

Data and provenance
-------------------

:download:`Results bundle (JSON) <results.json>` · :download:`Per-seed metrics (CSV) <replicas.csv>`

`Run code at e3edee2 <https://github.com/AstroAI-Lab/CODES-Benchmark/tree/e3edee24f25247142bacd314fd4f8434b1aa62a5>`_ · `Dataset archive <https://doi.org/10.5281/zenodo.18018572>`_ · `A&A paper <https://doi.org/10.1051/0004-6361/202658937>`_

Timing hardware: NVIDIA A30, 24576 MiB, 620.70.06. PyTorch 2.14.1, CUDA 13.0. The JSON bundle includes the protocol, configuration hashes, evaluation settings, and full-precision seed results.
