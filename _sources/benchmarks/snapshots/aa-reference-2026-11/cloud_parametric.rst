Cloud Parametric
================

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
     - 0.111 ± 0.003
     - **0.664 ± 0.016**
     - 17.08 ± 0.61
     - 0.176 ± 0.009
     - 360.8
   * - MultiONet
     - **0.101 ± 0.003**
     - 0.844 ± 0.021
     - **10.80 ± 0.38**
     - 0.233 ± 0.002
     - 371.9
   * - LatentNeuralODE
     - 1.522 ± 0.008
     - 4.837 ± 0.050
     - 99.77 ± 0.05
     - 0.300 ± 0.005
     - 233.8
   * - LatentPoly
     - 1.332 ± 0.012
     - 4.527 ± 0.062
     - 99.47 ± 0.01
     - **0.026 ± 0.000**
     - 17.1

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
     - **0.072 ± 0.003**
     - 0.484 ± 0.022
   * - MultiONet
     - 0.078 ± 0.000
     - **0.558 ± 0.005**
   * - LatentNeuralODE
     - 1.510 ± 0.002
     - -0.062 ± 0.026
   * - LatentPoly
     - 1.251 ± 0.005
     - 0.124 ± 0.012

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
     - 0.765
     - 0.703
     - 0.775
     - 0.765
   * - MultiONet
     - 0.616
     - 0.618
     - 0.780
     - **0.618**
   * - LatentNeuralODE
     - 1.878
     - 1.855
     - 1.844
     - 1.855
   * - LatentPoly
     - 2.088
     - 1.881
     - 2.002
     - 2.002

.. note:: The training split was subsampled by a factor of two. Evaluation used the full validation split. All three MultiONet seeds overflowed the real-space rollout error summaries.

Diagnostic plots
----------------

Absolute log-error distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/cloud_parametric/error-distribution.svg
   :alt: Smoothed error density versus absolute log error, with one line per surrogate.

   Each line averages the seed-level smoothed histograms for direct predictions, giving equal weight to each seed. Bins are evenly spaced in log space. The download records exact zeros separately.

Catastrophic-error recall
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diagnostics/cloud_parametric/catastrophic-error-recall.svg
   :alt: Recall of the largest ensemble errors versus the percentage of estimates flagged, with one curve per surrogate and a diagonal baseline for random flagging.

   Flag individual predicted quantities in order of decreasing ensemble spread. For each surrogate and seed, catastrophic errors are at or above the 99th percentile of the ensemble-mean absolute log error. Lines show mean recall, shading shows the minimum and maximum across seeds, and the diagonal shows what random flagging would catch on average.

:download:`Per-seed plot data (JSON) <diagnostics/cloud_parametric/plot-data.json>`

Data and provenance
-------------------

:download:`Results bundle (JSON) <results.json>` · :download:`Per-seed metrics (CSV) <replicas.csv>`

`Run code at e3edee2 <https://github.com/AstroAI-Lab/CODES-Benchmark/tree/e3edee24f25247142bacd314fd4f8434b1aa62a5>`_ · `Dataset archive <https://doi.org/10.5281/zenodo.18018572>`_ · `A&A paper <https://doi.org/10.1051/0004-6361/202658937>`_

Timing hardware: NVIDIA A30, 24576 MiB, 620.70.06. PyTorch 2.14.1, CUDA 13.0. The JSON bundle includes the protocol, configuration hashes, evaluation settings, and full-precision seed results.
