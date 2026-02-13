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
   - `prepare_data(self, datasets, metadata, …) -> tuple[DataLoader, DataLoader]`: build the train/validation dataloaders from the benchmark datasets. This is where you slice parameter sets, apply custom transforms, or pack tuples that your `forward` expects.
   - `fit(self, train_loader, val_loader, …)`: implements the training loop. `AbstractSurrogateModel.train_and_save_model` sets up everything (optimizers, schedulers, loggers) before calling `fit`, so you can focus on iterating over the loaders and calling `self.validate`.
2. **Reuse shared helpers**  
   - `self.setup_progress_bar(...)` draws status updates without clashing with the multi-process trainer.  
   - `self.predict(loader, ...)` is the canonical way to produce predictions/targets during validation and evaluation—this ensures consistent batching, buffer pre-allocation, and shape handling for every surrogate. Test your model with it to guarantee compatibility.
   - `self.validate(...)` bundles metric computation, Optuna pruning hooks, checkpointing, and logging. Call it from `fit` (typically once per epoch) after computing validation losses via `self.predict`.
   - Other protected helpers (`save`, `load`, `denormalise`, optimizer/scheduler factories) already encode CODES conventions; override only when a model’s requirements truly differ.
3. **Register the surrogate** by appending `AbstractSurrogateModel.register("MySurrogateName")` at the end of the file that defines the class. Use the string you want users to reference in `config.yaml`. Without this hook the CLI cannot instantiate your model.
4. **Expose configuration**  
   - Create a companion config dataclass (e.g., `deeponet_config.py`’s `MultiONetBaseConfig`). Inherit from `AbstractSurrogateBaseConfig` whenever possible so shared hyperparameters (learning rate, optimizer, scheduler, loss, activation) remain documented and gain sensible defaults.
   - Keep model-specific knobs inside that dataclass rather than the global config; users set them via the `surrogate_configs` section without touching other surrogates. The main benchmark config should stay model-agnostic unless you intentionally expose a cross-surrogate toggle.
5. **Checkpointing + evaluation**  
   - `AbstractSurrogateModel` already serializes weights, optimizer state, and scheduler state, and `run_eval.py` expects that layout. If you modify the format, verify that evaluation still loads checkpoints without custom flags.
   - `predict` and the evaluation pipeline assume consistent output shapes, so avoid ad-hoc reshaping in downstream scripts—handle it in `forward` or `prepare_data`. This uniform path is what keeps the benchmark fair across models.

Before submitting a PR or relying on the surrogate in large runs, train it with the minimal config and confirm that `run_eval.py` + `predict` behave as expected.

## Customize benchmark modes

New benchmark modes (beyond interpolation/extrapolation/sparse/batch-scaling/uncertainty/iterative) follow the same pattern: place an object with `enabled: bool` and the parameters you want to sweep under the top-level config. Detailed documentation for additional modes is coming soon—refer to [Running Benchmarks](running-benchmarks/index.md) for the currently supported modalities and CLI flags.
