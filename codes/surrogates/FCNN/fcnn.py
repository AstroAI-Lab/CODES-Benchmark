import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.surrogates.training import fit_standard, train_epoch
from codes.utils import make_shuffle_generator, time_execution

from .fcnn_config import FCNNBaseConfig


class FCFlatBatchIterable(IterableDataset):
    def __init__(
        self,
        inputs_t: torch.Tensor,
        targets_t: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ):
        self.inputs = inputs_t
        self.targets = targets_t
        self.N = inputs_t.size(0)
        self.bs = batch_size
        self.shuffle = shuffle
        self.generator = make_shuffle_generator() if shuffle else None

    def __iter__(self):
        perm = (
            torch.randperm(self.N, generator=self.generator)
            if self.shuffle
            else torch.arange(self.N)
        )
        for start in range(0, self.N, self.bs):
            idx = perm[start : start + self.bs]
            yield self.inputs[idx], self.targets[idx]

    def __len__(self):
        return math.ceil(self.N / self.bs)  # number of batches


class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_hidden_layers,
        activation=nn.ReLU(),
    ):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FullyConnected(AbstractSurrogateModel):
    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        n_parameters: int = 0,
        training_id: str | None = None,
        config: dict | None = None,
    ):
        """
        Initialize the FullyConnected model with a configuration.

        The configuration must provide the following keys:
        - hidden_size (int)
        - num_hidden_layers (int)
        - learning_rate (float)
        - regularization_factor (float)
        - schedule (bool)
        - activation (nn.Module name or instance)
        """
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            training_id=training_id,
            config=config,
        )
        self.config = FCNNBaseConfig(**self.config)
        self.device = device
        self.N = n_quantities

        # The input is the initial state (n_quantities), plus one for the time,
        # plus n_parameters (if any).
        input_size = self.N + 1 + n_parameters

        self.model = FullyConnectedNet(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=self.N,
            num_hidden_layers=self.config.num_hidden_layers,
            activation=self.config.activation,
        ).to(device)

    def forward(self, inputs: tuple) -> torch.Tensor:
        """
        Forward pass for the FullyConnected model.

        Args:
            inputs (tuple[torch.Tensor, torch.Tensor]):
                (x, targets) - 'targets' is included for a consistent interface
        Returns:
            (outputs, targets)
        """
        inputs = tuple(
            x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
            for x in inputs
        )
        x, targets = inputs
        return self.model(x), targets

    def forward_from_state(
        self,
        state: torch.Tensor,
        times: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the states at several time inputs from one initial state.

        Packs [state, time, params] as create_dataloader does. See
        AbstractSurrogateModel.forward_from_state.
        """
        B, N = state.shape
        T = times.shape[0]
        x = [
            state.unsqueeze(1).expand(B, T, N),
            times.view(1, T, 1).expand(B, T, 1),
        ]
        if params is not None:
            x.append(params.unsqueeze(1).expand(B, T, params.shape[1]))
        x = torch.cat(x, dim=2).reshape(B * T, -1)
        out, _ = self((x, None))
        return out.reshape(B, T, N)

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        dummy_timesteps: bool = True,
        dataset_train_params: np.ndarray | None = None,
        dataset_test_params: np.ndarray | None = None,
        dataset_val_params: np.ndarray | None = None,
    ):
        if dummy_timesteps:
            timesteps = np.linspace(0, 1, dataset_train.shape[1])

        assert timesteps.shape[0] == dataset_train.shape[1], (
            "Number of timesteps in timesteps array and dataset must match."
        )

        nw = getattr(self.config, "num_workers", 0)

        train_loader = self.create_dataloader(
            dataset_train,
            timesteps,
            batch_size,
            shuffle=shuffle,
            dataset_params=dataset_train_params,
            num_workers=nw,
        )

        test_loader = None
        if dataset_test is not None:
            test_loader = self.create_dataloader(
                dataset_test,
                timesteps,
                batch_size,
                shuffle=False,
                dataset_params=dataset_test_params,
                num_workers=nw,
            )

        val_loader = None
        if dataset_val is not None:
            val_loader = self.create_dataloader(
                dataset_val,
                timesteps,
                batch_size,
                shuffle=False,
                dataset_params=dataset_val_params,
                num_workers=nw,
            )

        return train_loader, test_loader, val_loader

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training FullyConnected",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the FullyConnected model.

        Args:
            train_loader (DataLoader): The DataLoader object containing the training data.
            test_loader (DataLoader): The DataLoader object containing the test data.
            epochs (int, optional): The number of epochs to train the model.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).

        Returns:
            None. The training loss, test loss, and MAE are stored in the model.
        """
        # Flat datasets hold one row per (sample, timestep); their length counts batches.
        n_rows = getattr(train_loader.dataset, "N", None)
        self.n_train_samples = None if n_rows is None else n_rows // self.n_timesteps
        fit_standard(
            self,
            train_loader,
            test_loader,
            epochs,
            position,
            description,
            multi_objective,
            epoch_fn=self.epoch,
        )

    def epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        train_epoch(self, data_loader, criterion, optimizer)

    def create_dataloader(
        self,
        data: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool,
        dataset_params: np.ndarray | None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """
        Build CPU tensors once and yield shuffled batches each epoch.
        data: (n_samples, n_timesteps, n_quantities)
        inputs = [initial_state, time, (params?)]
        targets = state_at_time
        """
        n_samples, n_timesteps, n_quantities = data.shape
        total = n_samples * n_timesteps

        if pin_memory and "cuda" not in self.device:
            pin_memory = False

        init_states = data[:, 0, :]  # (n_samples, n_quantities)
        rep_init = np.repeat(
            init_states[:, None, :], n_timesteps, 1
        )  # (n_samples, n_timesteps, n_quantities)
        time_feat = np.tile(
            timesteps.reshape(1, n_timesteps, 1), (n_samples, 1, 1)
        )  # (n_samples, n_timesteps, 1)

        if dataset_params is not None:
            rep_params = np.tile(dataset_params[:, None, :], (1, n_timesteps, 1))
            inputs = np.concatenate([rep_init, time_feat, rep_params], axis=2)
        else:
            inputs = np.concatenate([rep_init, time_feat], axis=2)

        targets = data  # (n_samples, n_timesteps, n_quantities)

        flat_inputs = inputs.reshape(total, inputs.shape[2])
        flat_targets = targets.reshape(total, n_quantities)

        inputs_t = torch.from_numpy(flat_inputs).float()
        targets_t = torch.from_numpy(flat_targets).float()

        ds = FCFlatBatchIterable(inputs_t, targets_t, batch_size, shuffle)
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            # Own generator, so iterating never draws from the shared global RNG.
            generator=torch.Generator(),
        )


AbstractSurrogateModel.register(FullyConnected)


class FullyConnectedResidual(FullyConnected):
    """
    FullyConnected variant that predicts the change from the initial state.

    The network output is added to the initial state, so it only has to learn
    the deviation from it. Everything downstream still sees absolute states.
    """

    def forward(self, inputs: tuple) -> torch.Tensor:
        outputs, targets = super().forward(inputs)
        x = inputs[0].to(self.device, non_blocking=True)
        return outputs + x[:, : self.N], targets


AbstractSurrogateModel.register(FullyConnectedResidual)
