# Extending The Benchmark

Plugging in new datasets or surrogates should not require rewriting orchestration code. This guide captures the supported extension points and the conventions that keep everything interoperable.

## Add a dataset

Use `datasets/_data_generation/make_new_dataset.py` as a working example. It shows how to generate synthetic trajectories, parameters, and timesteps before calling `codes.create_dataset`.

1. **Shape your arrays**  
   - `data`: `(n_samples, n_timesteps, n_quantities)` — the raw trajectories.  
   - `params` (optional): `(n_samples, n_parameters)` — per-trajectory parameters.  
   - `timesteps`: `(n_timesteps,)` — shared timeline for every trajectory.  
   - `labels` (optional): list of quantity names with length `n_quantities`.
2. **Call `create_dataset`**  
   ```python
   from codes import create_dataset

   create_dataset(
       "my_new_dataset",
       data=full_dataset,
       params=full_params,
       timesteps=timesteps,
       labels=labels,
       split=(0.7, 0.1, 0.2),  # train/test/val ratios
   )
   ```
   `create_dataset` writes `datasets/my_new_dataset/data.hdf5` with the following groups: `train`, `test`, `val`, optional `*_params`, and `timesteps`. The helper also ensures consistent shuffling and folder creation.
3. **Register the download link** (optional) in `datasets/data_sources.yaml` so `scripts/download_datasets.py` knows where to fetch the data. The docs pull this file automatically, so your dataset will appear in the catalog without extra work.
4. **Reference the dataset in configs** via `dataset.name`. Log transforms / normalization flags in the config can stay unchanged unless your data needs special treatment.

Once the folder exists, all CLI entry points (`run_training.py`, `run_eval.py`, `run_tuning.py`) will automatically pick it up based on `dataset.name`.

## Add a surrogate model

Every surrogate must inherit from `codes.surrogates.AbstractSurrogate.AbstractSurrogateModel`. This class wires together data preparation, training, logging, checkpointing, and evaluation.

1. **Implement the class** under `codes/surrogates/<YourModel>/<file>.py`.
   - `forward(self, inputs) -> tuple[Tensor, Tensor]`: receives exactly what `prepare_data` emits (for example `(branch_input, trunk_input, targets)` in MultiONet). Return `(predictions, targets)` in that order so the shared training/validation utilities can compute metrics without model-specific branching.
   - `prepare_data(...)`: build train, test, and optional validation dataloaders from the benchmark datasets. Pack batches in the form your `forward` expects.
   - `fit(self, train_loader, test_loader, epochs, position, description, multi_objective=False)`: train the model. This entry point remains required, including for models using the shared loop below.
2. **Reuse shared helpers**  
   - `self.setup_progress_bar(...)` draws status updates without clashing with the concurrent trainer.
   - `self.predict(loader, ...)` is the canonical way to produce predictions/targets during validation and evaluation—this ensures consistent batching, buffer pre-allocation, and shape handling for every surrogate. Test your model with it to guarantee compatibility.
   - `self.validate(...)` computes metrics and handles Optuna pruning and checkpointing. A custom loop should call it once per epoch; it runs metrics at `self.update_epochs` intervals.
   - `save`, `load`, `denormalize`, and `predict` are protected methods. Use the optimizer/scheduler factory for the configured training settings.
3. **Register the surrogate** by appending `AbstractSurrogateModel.register(MySurrogate)` at the end of the file that defines the class. The class name is used to identify the surrogate. Without this hook the CLI cannot instantiate your model.
4. **Expose configuration**  
   - Create a companion config dataclass (e.g., `deeponet_config.py`’s `MultiONetBaseConfig`). Inherit from `AbstractSurrogateBaseConfig` whenever possible so shared hyperparameters (learning rate, optimizer, scheduler, loss, activation) remain documented and gain sensible defaults.
   - Keep model-specific knobs inside that dataclass. Dataset-specific defaults can be supplied in `datasets/<name>/surrogates_config.py` as a dataclass named after the surrogate plus `Config`; they are loaded when `dataset.use_optimal_params` is enabled.
5. **Checkpointing + evaluation**  
   - Checkpoint selection uses the test loader during training. Keep the separate validation split independent for final evaluation. Verify saved models load in `run_eval.py`.
   - `predict` and the evaluation pipeline assume consistent output shapes, so avoid ad-hoc reshaping in downstream scripts—handle it in `forward` or `prepare_data`. This uniform path is what keeps the benchmark fair across models.
6. **Optional: rollout finetuning**
   - To support `run_finetuning.py`, implement `forward_from_state(self, state, times, params=None) -> Tensor`. It receives initial states of shape `(batch, n_quantities)` in normalised units, time inputs of shape `(n_times,)` in the dummy time coordinate the model is trained on, and optional parameters, and returns the predicted states of shape `(batch, n_times, n_quantities)`. Pack the inputs exactly as `prepare_data` does and call `forward`, so the result matches `predict`. See the implementations in `FullyConnected` and `MultiONet`.

### Standard and custom training loops

For models whose `forward` returns predictions and targets, `fit` can delegate to the optional shared loop. Decorate `fit` so saved models include the training duration:

```python
from codes.surrogates.training import fit_standard
from codes.utils import time_execution


@time_execution
def fit(
    self,
    train_loader,
    test_loader,
    epochs,
    position=0,
    description="Training MySurrogate",
    multi_objective=False,
):
    fit_standard(
        self,
        train_loader,
        test_loader,
        epochs,
        position=position,
        description=description,
        multi_objective=multi_objective,
    )
```

The helper uses the configured prediction loss, optimizer and scheduler, reports metrics, and handles checkpoints. For a different batch loss, pass a bound `loss_fn` accepting `(batch, criterion)` and returning a differentiable scalar tensor. For example, inside a model class:

```python
def training_loss(self, batch, criterion):
    predictions, targets = self(batch)
    return criterion(predictions, targets) + 0.01 * predictions.square().mean()


# In fit, pass loss_fn=self.training_loss to fit_standard.
```

The helper applies configured gradient clipping by default. FullyConnected, MultiONet and LatentNeuralODE use it; LatentPoly preserves its existing behavior without clipping. A model may instead implement its own complete `fit`. It should record `train_loss`, `test_loss` and `MAE` histories at the validation interval, set `n_epochs`, handle checkpoint restoration, and retain timing metadata via `@time_execution`.

To profile training, wrap a normal `fit` call rather than maintaining a separate loop:

```python
import torch

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    model.fit(train_loader, test_loader, epochs=1, position=0, description="Profile")
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

`ChemDataset` remains available for existing callers but is deprecated. New sequence loaders can use `FlatSeqBatchIterable`; it emits batches rather than individual samples, so adapt loader and batch handling accordingly.

Before submitting a PR or relying on the surrogate in large runs, train it with the minimal config and confirm that `run_eval.py` + `predict` behave as expected.

## Customize benchmark modes

New benchmark modes (beyond interpolation/extrapolation/sparse/batch-scaling/uncertainty/iterative) follow the same pattern: place an object with `enabled: bool` and the parameters you want to sweep under the top-level config. Detailed documentation for additional modes is coming soon—refer to [Running Benchmarks](running-benchmarks/index.md) for the currently supported modalities and CLI flags.
