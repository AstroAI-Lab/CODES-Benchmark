# Running the reference benchmark

The reference workflow has no cluster or scheduler dependency. It prepares a
scientific protocol, executes
the existing training and evaluation commands, and exports results for review.
Provisioning machines and submitting scheduler jobs remain outside this workflow.
Keep personal job scripts, allocation accounts, hostnames and filesystem layouts
outside the repository.

## Prepare the environment and inputs

Start from a clean, committed checkout and install its locked environment as
described in [Getting Started](../getting-started.md). Keep the checkout at that
revision for the entire run. Set `REPO` to the checkout and `RUN_ROOT` to an
existing writable directory **outside the checkout**. These shell variables are
local conveniences and are not part of the published protocol. Run the commands
from the checkout, including when invoking them from your own batch script.

Download the datasets before starting compute work:

```bash
cd "$REPO"
uv sync --locked
source .venv/bin/activate
python -c 'from codes.utils.data_utils import download_data; [download_data(name) for name in ("primordial", "primordial_parametric", "cloud", "cloud_parametric")]'
cp configs/benchmark/reference.yaml "$RUN_ROOT/reference-protocol.yaml"
```

The protocol preserves the original dataset-specific batch sizes, 8000–12000
epochs, log transforms, per-species min–max normalization, abundance floors,
best-checkpoint selection and five-member ensembles. These are HPO-refined
settings, with no autoregressive rollout finetuning. `cloud_parametric` retains
`subset_factor: 2` to control runtime for
the larger dataset. The other datasets use 1. Pilot and reference training use
the same configured subset; evaluation uses the held-out split through the
existing evaluation pipeline.

The template selects `devices: ["cuda:0"]`. Adjust the external copy to the
logical device available to the process. Use one device and run model tasks
sequentially to avoid contention within the reference run. Use otherwise idle
hardware for timing. GPU model, memory capacity, driver, CUDA runtime and package
versions are recorded; cluster names and allocation accounts are not required.

## Run a timing pilot

Preparation is lightweight and does not train models. It hashes the data, freezes
the protocol and dataset-specific hyperparameter files, and creates isolated
configs. Data are linked from the checkout; use `--data-root` if datasets are
stored separately under `DATA_ROOT/DATASET_NAME/data.hdf5`.

```bash
python -m benchmarks.reference prepare \
  --repo "$REPO" --protocol "$RUN_ROOT/reference-protocol.yaml" \
  --workdir "$RUN_ROOT/primordial-pilot" --dataset primordial --phase pilot
```

Run the following command **on the intended compute device**, whether directly,
inside an interactive allocation, or through a locally maintained batch script:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export MPLCONFIGDIR="$RUN_ROOT/matplotlib-cache"
python -m benchmarks.reference run --workdir "$RUN_ROOT/primordial-pilot"
python -m benchmarks.reference estimate --workdir "$RUN_ROOT/primordial-pilot"
```

The pilot trains each of the four architectures for two and four epochs and
evaluates each trained model. It uses the production data preprocessing, subset
and batch sizes. Increase the pilot budgets if startup and validation dominate
the measurements. Training estimates include three replicas and all five
ensemble members; ensemble evaluation cost is estimated from the single-model
measurements. These are uncertain extrapolations, so allow a walltime margin.
Repeat the pilot per dataset, and use a fresh workdir when comparing devices.

## Run the reference suite

Prepare one workdir per dataset with `--phase reference`, then execute each
prepared run on its intended device:

```bash
python -m benchmarks.reference prepare \
  --repo "$REPO" --protocol "$RUN_ROOT/reference-protocol.yaml" \
  --workdir "$RUN_ROOT/primordial-reference" --dataset primordial --phase reference
python -m benchmarks.reference run --workdir "$RUN_ROOT/primordial-reference"
```

Repeat for `primordial_parametric`, `cloud` and `cloud_parametric`. Scheduling
each dataset as an independent workload is optional. Use the same selected
device type and software environment when producing one combined reference table.
Each run uses seeds 42, 142 and 242, with disjoint ensemble-member seed ranges.
It trains main and ensemble models and evaluates the metrics described on the
[results page](index.md), including grid-based iterative predictions at interval
10. Interpolation, extrapolation, sparsity, batch scaling and rollout finetuning
are disabled for this initial reference suite.

Rerun the same `run` command to resume after inspecting a failure. Completed
stages are skipped and the training executor retains completed ensemble members.
An interrupted individual model starts training again; this is not an epoch-level
checkpoint resume. Changed source, inputs or runtime identity are rejected.
Two invocations cannot operate on the same workdir concurrently. Logs live under
`logs/`, and training workers retain their own task logs under `trained/`.

## Export public results

```bash
python -m benchmarks.results \
  --workdir "$RUN_ROOT/primordial-reference" \
  --workdir "$RUN_ROOT/primordial_parametric-reference" \
  --workdir "$RUN_ROOT/cloud-reference" \
  --workdir "$RUN_ROOT/cloud_parametric-reference" \
  --output "$RUN_ROOT/reference-results.json" \
  --markdown "$RUN_ROOT/reference-results.md"
```

Review the JSON bundle and generated report before copying the public artifacts
into the repository. The export retains scientific settings, individual results,
hashes and hardware/software provenance. It does not copy local manifests or raw
configuration comments and arbitrary metadata. Repository/workspace/data paths
in `manifest.json` are private execution details and stay in the external run
directory. Keep raw logs and personal scheduler scripts there as well; do not
commit the complete run directory.
