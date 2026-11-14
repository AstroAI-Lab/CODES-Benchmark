# Reference benchmark results

```{toctree}
:maxdepth: 1

running
```

Corrected reference results are forthcoming after a clean rerun. This page deliberately does not reproduce a numerical table until the runs and their provenance have been reviewed.

The reference suite covers `primordial`, `primordial_parametric`, `cloud`, and `cloud_parametric`, with MultiONet, FullyConnected, LatentNeuralODE, and LatentPoly. Each dataset and model has three independent replicas. Each replica uses a five-member ensemble including its main model, for 240 model trainings in total. The iterative evaluation uses the stored time grid and an interval of 10 grid steps; this is not a fixed physical-time step. See [the paper, sections 4.4 and 5.4](https://arxiv.org/html/2603.08567v1) for the reference protocol.

The run protocol preserves the saved training settings supplied by the authors,
including `subset_factor: 2` for `cloud_parametric`. The other datasets use a
factor of 1. All models use the checked-in HPO-refined configurations and ordinary
training. Residual variants and rollout-finetuned models will be evaluated
separately; the repository history records parent-config inheritance for the
residual variants, with no residual-specific tuning configs or results.

| Dataset | Initial conditions | Training subset factor |
| --- | ---: | ---: |
| Primordial | 2048 | 1 |
| Primordial parametric | 4096 | 1 |
| Cloud | 4096 | 1 |
| Cloud parametric | 8192 | 2 |

The cloud-parametric subsampling was chosen to control runtime for the larger
dataset. It preserves the original training budget; it does not redefine the
underlying dataset or the held-out evaluation split.

These are corrected-code reference results, not an exact numerical reproduction
of the published table. In particular, iterative evaluation has changed since
publication. Three independent repetitions add information about training
variability, and synchronized accelerator timing uses a different measurement
method from the older results.

The eventual report will include mean log absolute error (mLAE), the 99th percentile log absolute error (LAE99), mean relative error (mRE), inference time, parameter count, iterative mLAE, mean ensemble uncertainty, ensemble mLAE, and the Pearson correlation of ensemble uncertainty with error. Log errors and uncertainty are in dex; mRE is a percentage; parameters are in thousands. Inference is synchronized forward-call wall time in microseconds per prediction, normalized by the number of validation trajectories times stored timesteps. This is sensitive to the validation workload, hardware, and software; compare timings only under matching conditions. The raw metrics also contain the summed time per complete validation run.

Ensemble uncertainty is the average spread of members within a replica. Published mean ± sample standard deviation describes variation **between the three replicas**, including for inference time; it does not average the timing repetitions' standard deviations.

After all evaluation outputs are available, export a machine-readable bundle and a deterministic Markdown report:

```bash
python -m benchmarks.results --workdir /path/to/run \
  --output reference-results.json --markdown reference-results.md
```

Repeat `--workdir` for additional runs. Each included dataset must have all four models and the same three replica seeds. The exporter checks finite metrics, configuration hashes, and compatible recorded code, protocol, data, and software provenance. Review the bundle, report, configurations, and hardware details together, then update this page and check in the bundle and generated report through a reviewed pull request. Do not silently combine results from different versions, protocols, datasets, or hardware. Retain the individual replica values and provenance alongside any aggregates.

Public exports include only the supported scientific configuration fields and
device/software provenance. YAML comments and arbitrary local metadata are
excluded. Local manifests, execution paths, raw logs and personal scheduler
scripts stay outside the repository. See the [running guide](running.md) for the
separation between local execution files and public result artifacts.

Changes that affect results, add a surrogate, or add a reported metric must include
the applicable reruns and an updated versioned results bundle and report. The
contributor making the change is responsible for that update. Extend the protocol
and exporter when adding a new reference model or metric. Preserve older bundles
as dated historical results and clearly identify the current baseline. Publication
is a manual, reviewed repository update; no benchmark server is required.
