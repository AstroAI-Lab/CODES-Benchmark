# Accuracy Metrics

Chemical-abundance trajectories span many orders of magnitude, which makes standard absolute or relative error metrics misleading. CODES therefore measures accuracy in log space everywhere—during tuning, training, and evaluation.

## Log-space absolute error

- **mLAE** (mean log absolute error): the mean of `|log10(pred) - log10(target)|` across all species, timesteps, and trajectories.
- **LAE99`**: the 99th percentile of the same distribution. This captures worst-case behaviour without being dominated by a handful of outliers.

Because the models are trained on log-transformed abundances, these values directly express errors in orders of magnitude (dex).

## Why not relative error?

- Relative errors explode when dividing by very small abundances and become asymmetric (large overestimations vs. capped underestimations).
- Adding thresholds to stabilize the denominator implicitly weights species without a clear physical justification.

Log-space metrics avoid these pitfalls: they treat over- and under-predictions symmetrically and remain finite even for tiny abundances.

## Where metrics appear

- **Tuning** — single-objective studies minimize LAE99; dual-objective studies minimize both LAE99 and inference time.
- **Training** — loss functions (e.g., Smooth L1, MSE) operate on log-transformed outputs, so optimization aligns with the evaluation metrics.
- **Evaluation** — core reports always include mLAE and LAE99, alongside timing/compute metrics. Optional diagnostics (loss curves, gradient correlations) are derived from the same log-space signals.
