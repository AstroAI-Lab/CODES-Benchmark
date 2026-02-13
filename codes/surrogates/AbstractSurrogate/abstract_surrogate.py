import dataclasses
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

import numpy as np
import optuna
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.utils import create_model_dir, parse_hyperparameters


class AbstractSurrogateModel(ABC, nn.Module):
    """
    Abstract base class for surrogate models. This class implements the basic
    structure of a surrogate model and defines the methods that need to be
    implemented by the subclasses for it to be compatible with the benchmarking
    framework. For more information, see
    https://codes-docs.web.app/documentation.html#add_model.

    Args:
        device (str, optional): The device to run the model on. Defaults to None.
        n_quantities (int, optional): The number of quantities. Defaults to 29.
        n_timesteps (int, optional): The number of timesteps. Defaults to 100.
        config (dict, optional): The configuration dictionary. Defaults to {}.

    Attributes:
        train_loss (float): The training loss.
        test_loss (float): The test loss.
        MAE (float): The mean absolute error.
        normalisation (dict): The normalisation parameters.
        train_duration (float): The training duration.
        device (str): The device to run the model on.
        n_quantities (int): The number of quantities.
        n_timesteps (int): The number of timesteps.
        L1 (nn.L1Loss): The L1 loss function.
        config (dict): The configuration dictionary.

    Methods:

        forward(inputs: Any) -> tuple[Tensor, Tensor]:
            Forward pass of the model.

        prepare_data(
            dataset_train: np.ndarray,
            dataset_test: np.ndarray | None,
            dataset_val: np.ndarray | None,
            timesteps: np.ndarray,
            batch_size: int,
            shuffle: bool,
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
            Gets the data loaders for training, testing, and validation.

        fit(
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int | None,
            position: int,
            description: str,
        ) -> None:
            Trains the model on the training data. Sets the train_loss and test_loss attributes.

        predict(data_loader: DataLoader) -> tuple[Tensor, Tensor]:
            Evaluates the model on the given data loader.

        save(
            model_name: str,
            subfolder: str,
            training_id: str,
            data_info: dict,
        ) -> None:
            Saves the model to disk.

        load(training_id: str, surr_name: str, model_identifier: str) -> None:
            Loads a trained surrogate model.

        setup_progress_bar(epochs: int, position: int, description: str) -> tqdm:
            Helper function to set up a progress bar for training.

        denormalize(data: Tensor) -> Tensor:
            Denormalizes the data back to the original scale.
    """

    _registry: list[type["AbstractSurrogateModel"]] = []
    _protected_methods = [
        "predict",
        "save",
        "load",
        "denormalize",
        "setup_progress_bar",
    ]

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        n_parameters: int = 0,
        training_id: str | None = None,
        config: dict | None = None,
    ):
        super().__init__()
        self.train_loss = None
        self.test_loss = None
        self.MAE = None
        self.normalisation = None
        self.device = device
        self.n_quantities = n_quantities
        self.n_timesteps = n_timesteps
        self.n_parameters = n_parameters
        self.L1 = nn.L1Loss()
        self.config = config if config is not None else {}
        self.train_duration = None
        self.optuna_trial = None
        self.update_epochs = 10
        self.n_epochs = 0
        self.training_id = training_id

        # Checkpointing attributes
        self.checkpointing: bool = False
        self.best_test_loss: float | None = None
        self.best_epoch: int | None = None
        self._checkpoint_path: str | None = None

    @classmethod
    def register(cls, surrogate: type["AbstractSurrogateModel"]):
        """Registers a surrogate model class into the registry."""
        if not issubclass(surrogate, cls):
            raise TypeError(
                f"{surrogate.__name__} must be a subclass of AbstractSurrogateModel."
            )

        for method in cls._protected_methods:
            if method in surrogate.__dict__:
                raise AttributeError(
                    f"Method {method} is protected and cannot be overridden."
                )
        cls._registry.append(surrogate)

    @classmethod
    def get_registered_classes(cls) -> list[type["AbstractSurrogateModel"]]:
        """Returns the list of registered surrogate model classes."""
        return cls._registry

    @abstractmethod
    def forward(self, inputs: Any) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (Any): The input data as recieved from the dataloader.

        Returns:
            tuple[Tensor, Tensor]: The model predictions and the targets.
        """
        pass

    @abstractmethod
    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool,
        dummy_timesteps: bool = True,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Prepare the data for training, testing, and validation. This method should
        return the DataLoader objects for the training, testing, and validation data.

        Args:
            dataset_train (np.ndarray): The training dataset.
            dataset_test (np.ndarray): The testing dataset.
            dataset_val (np.ndarray): The validation dataset.
            timesteps (np.ndarray): The timesteps.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            dummy_timesteps (bool): Whether to use dummy timesteps. Defaults to True.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: The DataLoader objects for the
                training, testing, and validation data.
        """
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int,
        description: str,
        multi_objective: bool,
    ) -> None:
        """
        Perform the training of the model. Sets the train_loss and test_loss attributes.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the testing data.
            epochs (int): The number of epochs to train the model for.
            position (int): The position of the progress bar.
            description (str): The description of the progress bar.
            multi_objective (bool): Whether the training is multi-objective.
        """
        pass

    def predict(
        self, data_loader: DataLoader, leave_log: bool = False, leave_norm: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate the model on the given dataloader.

        Args:
            data_loader (DataLoader): The DataLoader object containing the data the
                model is evaluated on.
            leave_log (bool): If True, do not exponentiate the data even if log10_transform is True.
            leave_norm (bool): If True, do not denormalize the data even if normalisation is applied.

        Returns:
            tuple[Tensor, Tensor]: The predictions and targets.
        """
        # infer output size
        with torch.inference_mode():
            dummy_inputs = next(iter(data_loader))
            dummy_outputs, _ = self(dummy_inputs)
            batch_size, out_shape = (
                dummy_outputs.shape[0],
                dummy_outputs.shape[-(dummy_outputs.ndim - 1) :],
            )

        # pre-allocate buffers for predictions and targets
        length = len(data_loader)
        size = (batch_size * length, *out_shape)
        predictions = torch.zeros(size, dtype=dummy_outputs.dtype).to(self.device)
        targets = torch.zeros(size, dtype=dummy_outputs.dtype).to(self.device)

        processed_samples = 0

        with torch.inference_mode():
            for inputs in data_loader:
                inputs = [
                    x.to(self.device, non_blocking=True) if isinstance(x, Tensor) else x
                    for x in inputs
                ]
                preds, targs = self(inputs)
                current_batch_size = preds.shape[0]  # get actual batch size
                predictions[
                    processed_samples : processed_samples + current_batch_size, ...
                ] = preds
                targets[
                    processed_samples : processed_samples + current_batch_size, ...
                ] = targs
                processed_samples += current_batch_size

        # Slice the buffers to include only the processed samples
        predictions = predictions[:processed_samples, ...]
        targets = targets[:processed_samples, ...]

        predictions = self.denormalize(predictions, leave_log, leave_norm)
        targets = self.denormalize(targets, leave_log, leave_norm)

        predictions = predictions.reshape(-1, self.n_timesteps, self.n_quantities)
        targets = targets.reshape(-1, self.n_timesteps, self.n_quantities)

        return predictions, targets

    def save(
        self,
        model_name: str,
        base_dir: str,
        training_id: str,
    ) -> None:
        """
        Save the model to disk.

        Args:
            model_name (str): The name of the model.
            subfolder (str): The subfolder to save the model in.
            training_id (str): The training identifier.
            data_info (dict): The data parameters.
        """

        # Make the model directory
        subfolder = os.path.join(base_dir, training_id, self.__class__.__name__)
        model_dir = create_model_dir(base_dir, subfolder)

        # Load and clean the hyperparameters
        hyperparameters = dataclasses.asdict(self.config)
        remove_keys = ["masses"]
        for key in remove_keys:
            hyperparameters.pop(key, None)
        for key in hyperparameters.keys():
            if isinstance(hyperparameters[key], nn.Module):
                hyperparameters[key] = hyperparameters[key].__class__.__name__

        # Check if the model has some attributes. If so, add them to the hyperparameters
        check_attributes = [
            "n_train_samples",
            "n_timesteps",
        ]
        for attr in check_attributes:
            if hasattr(self, attr):
                hyperparameters[attr] = getattr(self, attr)
                # n_timesteps must not be saved to the model, it will cause problems in the benchmark
                if attr == "n_timesteps":
                    delattr(self, attr)

        # Remove optuna_trial from the class attributes
        if hasattr(self, "optuna_trial"):
            hyperparameters.pop("optuna_trial", None)

        # Add some additional information to the model and hyperparameters
        self.train_duration = (
            self.fit.duration if hasattr(self.fit, "duration") else None
        )
        hyperparameters["train_duration"] = self.train_duration
        hyperparameters["n_epochs"] = self.n_epochs
        hyperparameters["normalisation"] = self.normalisation
        hyperparameters["device"] = self.device
        if "cuda" in self.device:
            devinfo = torch.cuda.get_device_properties(self.device)
            hyperparameters["device_info"] = (
                f"{devinfo.name} ({devinfo.total_memory / 1e9:.2f}GB), CUDA {devinfo.major}.{devinfo.minor}"
            )
        hyperparameters["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Recursively parse hyperparameters to make them yaml-serializable
        clean_hyperparameters = parse_hyperparameters(hyperparameters.copy())

        # Reduce the precision of the losses and accuracy
        for attribute in ["train_loss", "test_loss", "MAE"]:
            value = getattr(self, attribute)
            if value is not None:
                if isinstance(value, Tensor):
                    value = value.cpu().detach().numpy()
                if isinstance(value, np.ndarray):
                    value = value.astype(np.float32)
                setattr(self, attribute, value)

        # Save the hyperparameters as a yaml file
        hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
        with open(hyperparameters_path, "w", encoding="utf-8") as file:
            yaml.dump(clean_hyperparameters, file)

        save_attributes = {
            k: v
            for k, v in self.__dict__.copy().items()
            if k != "state_dict" and not k.startswith("_")
        }
        model_dict = {"state_dict": self.state_dict(), "attributes": save_attributes}

        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model_dict, model_path)

    def load(
        self,
        training_id: str,
        surr_name: str,
        model_identifier: str,
        model_dir: str | None = None,
    ) -> None:
        """
        Load a trained surrogate model.

        Args:
            training_id (str): The training identifier.
            surr_name (str): The name of the surrogate model.
            model_identifier (str): The identifier of the model (e.g., 'main').

        Returns:
            None. The model is loaded in place.
        """
        if model_dir is None:
            model_dict_path = os.path.join(
                os.getcwd(),
                "trained",
                training_id,
                surr_name,
                f"{model_identifier}.pth",
            )
        else:
            model_dict_path = os.path.join(
                model_dir, training_id, surr_name, f"{model_identifier}.pth"
            )
        model_dict = torch.load(
            model_dict_path, map_location=self.device, weights_only=False
        )
        self.load_state_dict(model_dict["state_dict"])
        for key, value in model_dict["attributes"].items():
            # remove self.device from the attributes
            if key == "device":
                continue
            else:
                setattr(self, key, value)
        self.to(self.device)
        self.eval()

    def setup_progress_bar(self, epochs: int, position: int, description: str):
        """
        Helper function to set up a progress bar for training.

        Args:
            epochs (int): The number of epochs.
            position (int): The position of the progress bar.
            description (str): The description of the progress bar.

        Returns:
            tqdm: The progress bar.
        """

        bar_format = "{l_bar}{bar}| {n_fmt:>5}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
        progress_bar = tqdm(
            range(epochs),
            desc=description,
            position=position,
            leave=False,
            bar_format=bar_format,
        )

        # Only used for time_pruning in multi objective optimisation
        self._trial_start_time = time.time()

        return progress_bar

    def denormalize(
        self,
        data: Tensor | np.ndarray,
        leave_log: bool = False,
        leave_norm: bool = False,
    ) -> Tensor | np.ndarray:
        """
        Denormalize the data.

        Args:
            data (Tensor | np.ndarray): The data to denormalize.
            leave_log (bool): If True, do not exponentiate the data even if log10_transform is True.
            leave_norm (bool): If True, do not denormalize the data even if normalisation is applied.

        Returns:
            Tensor | np.ndarray: The denormalized data.
        """
        data_type = None
        if self.normalisation is not None:
            if not leave_norm:
                data_type = data.dtype
                if self.normalisation["mode"] == "disabled":
                    ...
                elif self.normalisation["mode"] == "minmax":
                    dmax = self.normalisation["max"]
                    dmin = self.normalisation["min"]
                    dmax = np.array(dmax) if isinstance(dmax, list) else dmax
                    dmin = np.array(dmin) if isinstance(dmin, list) else dmin
                    if isinstance(data, Tensor) and isinstance(dmax, np.ndarray):
                        dmax = Tensor(dmax).to(data.device)
                        dmin = Tensor(dmin).to(data.device)

                    # data = data.to("cpu")
                    data = (data + 1) * (dmax - dmin) / 2 + dmin
                elif self.normalisation["mode"] == "standardize":
                    mean = self.normalisation["mean"]
                    std = self.normalisation["std"]
                    if isinstance(data, Tensor) and isinstance(mean, np.ndarray):
                        mean = Tensor(mean).to(data.device)
                        std = Tensor(std).to(data.device)
                    data = data * std + mean

            if self.normalisation["log10_transform"] and not leave_log:
                data = 10**data

            # Conserve dtype
            if data_type is not None:
                if isinstance(data, Tensor):
                    return data.to(dtype=data_type)
                if isinstance(data, np.ndarray):
                    return data.astype(data_type)

        return data

    def denormalize_old(self, data: Tensor) -> Tensor:
        """
        Denormalize the data.

        Args:
            data (np.ndarray): The data to denormalize.

        Returns:
            np.ndarray: The denormalized data.
        """
        if self.normalisation is not None:
            if self.normalisation["mode"] == "disabled":
                ...
            elif self.normalisation["mode"] == "minmax":
                dmax = self.normalisation["max"]
                dmin = self.normalisation["min"]
                data = (data + 1) * (dmax - dmin) / 2 + dmin
            elif self.normalisation["mode"] == "standardize":
                mean = self.normalisation["mean"]
                std = self.normalisation["std"]
                data = data * std + mean

            if self.normalisation["log10_transform"]:
                data = 10**data

        return data

    def time_pruning(self, current_epoch: int, total_epochs: int) -> None:
        """
        Determine whether a trial should be pruned based on projected runtime,
        but only after a warmup period (10% of the total epochs).

        Warmup: Do not prune if current_epoch is less than warmup_epochs.
        After warmup, compute the average epoch time, extrapolate the total runtime,
        and retrieve the threshold (runtime_threshold) from the study's user attributes.
        If the projected runtime exceeds the threshold, raise an optuna.TrialPruned exception.

        Args:
            current_epoch (int): The current epoch count.
            total_epochs (int): The planned total number of epochs.

        Raises:
            optuna.TrialPruned: If the projected runtime exceeds the threshold.
        """
        # Define warmup period based on 10% of total epochs.
        warmup_epochs = max(10, int(total_epochs * 0.02))
        if current_epoch < warmup_epochs:
            return

        elapsed = time.time() - self._trial_start_time
        completed_epochs = max(current_epoch, 1)
        average_epoch_time = elapsed / completed_epochs
        projected_total_time = average_epoch_time * total_epochs

        # Retrieve threshold from study's user attributes.
        if self.optuna_trial is not None and hasattr(self.optuna_trial, "study"):
            threshold = self.optuna_trial.study.user_attrs.get(
                "runtime_threshold", None
            )
        else:
            threshold = None

        if threshold is not None:
            if projected_total_time > threshold:
                if self.optuna_trial is not None:
                    tqdm.write(
                        f"[Trial {self.optuna_trial.number}] Projected total time {projected_total_time:.1f}s exceeds threshold {threshold:.1f}s. Pruning trial."
                    )
                    self.optuna_trial.set_user_attr(
                        "prune_reason",
                        f"Projected runtime {projected_total_time:.1f}s exceeds threshold {threshold:.1f}s",
                    )
                raise optuna.TrialPruned(
                    f"Projected total time {projected_total_time:.1f}s exceeds threshold {threshold:.1f}s"
                )

    def setup_checkpoint(self) -> None:
        """
        Prepare everything needed to save the single 'best' checkpoint.
        Must be called before any call to `self.checkpoint(...)`.
        """
        if not self.checkpointing:
            return

        self.best_test_loss = float("inf")
        self.best_epoch = -1

        # Build a small folder under `trained/<training_id>/<ModelClass>/`.
        training_id = self.training_id
        if training_id is None:
            raise RuntimeError(
                "Cannot call setup_checkpoint(): Attribute `self.training_id` is missing."
            )

        model_dir = os.path.join(
            os.getcwd(),
            "trained",
            training_id,
            self.__class__.__name__,
        )
        os.makedirs(model_dir, exist_ok=True)

        # We’ll store exactly one file called "best_checkpoint.pth"
        self._checkpoint_path = os.path.join(model_dir, "best_checkpoint.pth")

    def checkpoint(self, test_loss: float, epoch: int) -> None:
        """
        If save_best is True and test_loss < self.best_test_loss,
        overwrite the single-file checkpoint on disk and update best_test_loss/epoch.
        """
        if not self.checkpointing:
            return

        if self.best_test_loss is None:
            raise RuntimeError(
                "You must call setup_checkpoint() before calling checkpoint()."
            )

        # If this epoch is strictly better than anything before, overwrite:
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_epoch = epoch
            torch.save(self.state_dict(), self._checkpoint_path)

    def get_checkpoint(
        self,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> None:
        """
        After training, compare the current model’s test loss to the best recorded loss.
        If the final model is better, keep it; otherwise load the saved best checkpoint.

        Args:
            test_loader (DataLoader): DataLoader for computing final test loss.
            criterion (nn.Module): Loss function used for evaluation.
        """
        if not self.checkpointing:
            return

        if self.best_epoch is None or self.best_epoch < 0:
            return

        if self._checkpoint_path is None or not os.path.isfile(self._checkpoint_path):
            print(
                f"Warning: no checkpoint file found at {self._checkpoint_path}. "
                "Skipping load of best weights."
            )
            self.best_epoch = -1
            self.best_test_loss = None
            return

        self.eval()
        with torch.inference_mode():
            preds, targets = self.predict(test_loader)
            final_loss = criterion(preds, targets).item()

        if final_loss < self.best_test_loss:
            try:
                os.remove(self._checkpoint_path)
            except Exception:
                pass
            self.best_epoch = self.n_epochs - 1
            self.best_test_loss = final_loss
            return

        checkpoint_state = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=True
        )
        self.load_state_dict(checkpoint_state)

        try:
            os.remove(self._checkpoint_path)
        except Exception:
            pass

    def validate(
        self,
        epoch: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        progress_bar: tqdm,
        total_epochs: int,
        multi_objective: bool,
    ) -> None:
        """
        Shared “validation + checkpoint” logic, to be called once per epoch in each fit().

        Relies on:
          - self.update_epochs (int)
          - self.train_loss  (np.ndarray)
          - self.test_loss   (np.ndarray)
          - self.MAE         (np.ndarray)
          - self.optuna_trial
          - self.L1 (nn.L1Loss)
          - self.predict(...)
          - self.checkpoint(test_loss, epoch)

        Only runs if (epoch % self.update_epochs) == 0.
        Main reporting metric is MAE in log10-space (i.e., Δdex). Additionally, MAE in linear space is computed.
        """

        # If it's not time to check yet, do nothing.
        if epoch % self.update_epochs != 0:
            return

        index = epoch // self.update_epochs

        # Switch into inference/eval mode and compute losses
        with torch.inference_mode():
            self.eval()
            optimizer.eval() if hasattr(optimizer, "eval") else None

            # Compute losses
            preds, targets = self.predict(train_loader, leave_log=True)
            self.train_loss[index] = self.L1(preds, targets).item()
            preds, targets = self.predict(test_loader, leave_log=True)
            self.test_loss[index] = self.L1(preds, targets).item()
            preds, targets = self.predict(test_loader)
            self.MAE[index] = self.L1(preds, targets).item()

            progress_bar.set_postfix(
                {
                    "train_loss": f"{self.train_loss[index]:.2e}",
                    "test_loss": f"{self.test_loss[index]:.2e}",
                }
            )

            if self.optuna_trial is not None:
                if multi_objective:
                    self.time_pruning(current_epoch=epoch, total_epochs=total_epochs)
                else:
                    self.optuna_trial.report(self.test_loss[index], step=epoch)
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                    elif np.isinf(self.test_loss[index]) or np.isnan(
                        self.test_loss[index]
                    ):
                        raise optuna.TrialPruned(
                            "Test loss is NaN or Inf, pruning trial."
                        )

            self.checkpoint(self.test_loss[index], epoch)

            self.train()
            optimizer.train() if hasattr(optimizer, "train") else None

    def setup_optimizer_and_scheduler(
        self,
        epochs: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Set up optimizer and scheduler based on self.config.scheduler and self.config.optimizer.
        Supports "adamw", "sgd" optimizers and "schedulefree", "cosine", "poly" schedulers.
        Patches standard optimizers so that .train() and .eval() exist as no-ops.
        Patches ScheduleFree optimizers to have a no-op scheduler.step().
        For ScheduleFree optimizers, use lr warmup for the first 1% of epochs.
        For Poly scheduler, use a power decay based on self.config.poly_power.
        For Cosine scheduler, use a minimum learning rate defined by self.config.eta_min.

        Args:
            epochs (int): The number of epochs the training will run for.

        Returns:
            tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
                The optimizer and scheduler instances.
        Raises:
            ValueError: If an unknown optimizer or scheduler is specified in the config.
        """
        scheduler_name = self.config.scheduler.lower()
        optimizer_name = self.config.optimizer.lower()

        class DummyScheduler:
            def step(self, *args, **kwargs):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        # create optimizer
        if optimizer_name == "adamw":
            if scheduler_name == "schedulefree":
                from schedulefree import AdamWScheduleFree

                optimizer = AdamWScheduleFree(
                    self.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.regularization_factor,
                    warmup_steps=max(1, epochs // 100),
                )
            else:
                from torch.optim import AdamW

                optimizer = AdamW(
                    self.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.regularization_factor,
                )
        elif optimizer_name == "sgd":
            momentum = self.config.momentum
            if scheduler_name == "schedulefree":
                from schedulefree import SGDScheduleFree

                optimizer = SGDScheduleFree(
                    self.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.regularization_factor,
                    momentum=momentum,
                    warmup_steps=max(1, epochs // 100),
                )
            else:
                from torch.optim import SGD

                optimizer = SGD(
                    self.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.regularization_factor,
                    momentum=momentum,
                )
        else:
            raise ValueError(f"Unknown optimizer '{self.config.optimizer}'")

        # Patch optimizer to have no-op train() and eval(), if not present
        if not hasattr(optimizer, "train"):

            def _opt_train():
                pass

            optimizer.train = _opt_train
        if not hasattr(optimizer, "eval"):

            def _opt_eval():
                pass

            optimizer.eval = _opt_eval

        # create scheduler
        if scheduler_name == "schedulefree":
            scheduler = DummyScheduler()
        elif scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            eta_min = self.config.eta_min
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=eta_min,
            )
        elif scheduler_name == "poly":
            from torch.optim.lr_scheduler import LambdaLR

            power = self.config.poly_power
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda epoch: (1 - epoch / float(epochs)) ** power
            )
        else:
            raise ValueError(f"Unknown scheduler '{self.config.scheduler}'")

        return optimizer, scheduler


SurrogateModel = TypeVar("SurrogateModel", bound=AbstractSurrogateModel)
