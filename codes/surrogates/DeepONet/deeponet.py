from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.utils import time_execution

from .deeponet_config import MultiONetBaseConfig
from .don_utils import FlatBatchIterable, mass_conservation_loss


class BranchNet(nn.Module):
    """
    Class that defines the branch network for the MultiONet model.

    Args:
        input_size (int): The input size for the network.
        hidden_size (int): The number of hidden units in each layer.
        output_size (int): The number of output units.
        num_hidden_layers (int): The number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the branch network.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """
    Class that defines the trunk network for the MultiONet model.

    Args:
        input_size (int): The input size for the network.
        hidden_size (int): The number of hidden units in each layer.
        output_size (int): The number of output units.
        num_hidden_layers (int): The number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), activation]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the trunk network.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.network(x)


class OperatorNetwork(AbstractSurrogateModel):
    """
    Abstract class for operator networks.
    Child classes must implement a forward method and use the branch_net and trunk_net attributes.

    Args:
        device (str, optional): The device to use for training (e.g., 'cpu', 'cuda:0').
        n_quantities (int, optional): The number of quantities.
        n_timesteps (int, optional): The number of timesteps.

    Raises:
        NotImplementedError: Child classes must implement a forward method.
        NotImplementedError: Child classes must initialize a branch_net and trunk_net.
    """

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        training_id: str | None = None,
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            training_id=training_id,
            config=config,
        )

    def post_init_check(self):
        if not hasattr(self, "branch_net") or not hasattr(self, "trunk_net"):
            raise NotImplementedError(
                "Child classes must initialize a branch_net and trunk_net."
            )
        if not hasattr(self, "forward") or not callable(self.forward):
            raise NotImplementedError("Child classes must implement a forward method.")

    def forward(self, branch_input, trunk_input):
        # Define a generic forward pass or raise an error to enforce child class implementation
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    @staticmethod
    def _calculate_split_sizes(total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits
        return [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]


OperatorNetworkType = TypeVar("OperatorNetworkType", bound=OperatorNetwork)


class MultiONet(OperatorNetwork):
    """
    Class that implements the MultiONet model.
    It differs from a standard DeepONet in that it has multiple outputs, which are obtained by
    splitting the outputs of branch and trunk networks and calculating the scalar product of the splits.

    Args:
        device (str, optional): The device to use for training (e.g., 'cpu', 'cuda:0').
        n_quantities (int, optional): The number of quantities.
        n_timesteps (int, optional): The number of timesteps.
        n_parameters (int, optional): The number of fixed parameters. Defaults to 0.
        config (dict, optional): The configuration for the model.

        The configuration must provide the following information:

        - trunk_input_size (int): The input size for the trunk network.
        - hidden_size (int): The number of hidden units in each layer of the branch and trunk networks.
        - branch_hidden_layers (int): The number of hidden layers in the branch network.
        - trunk_hidden_layers (int): The number of hidden layers in the trunk network.
        - output_factor (int): The factor by which the number of outputs is multiplied.
        - learning_rate (float): The learning rate for the optimizer.
        - schedule (bool): Whether to use a learning rate schedule.
        - regularization_factor (float): The regularization factor for the optimizer.
        - masses (np.ndarray, optional): The masses for mass conservation loss.
        - massloss_factor (float, optional): The factor for the mass conservation loss.
        - params_branch (bool): If True, fixed parameters are concatenated to the branch net;
              if False, to the trunk net.

    Raises:
        TypeError: Invalid configuration for MultiONet model.
    """

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        n_parameters: int = 0,
        training_id: str | None = None,
        config: dict | None = None,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            training_id=training_id,
            config=config,
        )
        self.config = MultiONetBaseConfig(**self.config)
        self.device = device
        self.N = n_quantities  # Number of quantities
        self.outputs = (
            n_quantities * self.config.output_factor
        )  # Number of neurons in the last layer

        # Decide where to feed the fixed parameters based on self.config.params_branch.
        # BranchNet's input is originally:
        #   n_quantities - (self.config.trunk_input_size - 1)
        # TrunkNet's input is originally self.config.trunk_input_size.
        if self.config.params_branch:
            branch_input_size = (
                n_quantities - (self.config.trunk_input_size - 1)
            ) + n_parameters
            trunk_input_size = self.config.trunk_input_size
        else:
            branch_input_size = n_quantities - (self.config.trunk_input_size - 1)
            trunk_input_size = self.config.trunk_input_size + n_parameters

        self.branch_net = BranchNet(
            branch_input_size,
            self.config.hidden_size,
            self.outputs,
            self.config.branch_hidden_layers,
            self.config.activation,
        ).to(device)
        self.trunk_net = TrunkNet(
            trunk_input_size,
            self.config.hidden_size,
            self.outputs,
            self.config.trunk_hidden_layers,
            self.config.activation,
        ).to(device)

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MultiONet model.

        Args:
            inputs (tuple): The input tuple containing branch_input, trunk_input, and targets.

        Returns:
            tuple: The model outputs and the targets.
        """
        inputs = tuple(
            x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
            for x in inputs
        )
        branch_input, trunk_input, targets = inputs
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for b_split, t_split in zip(branch_splits, trunk_splits):
            result.append(torch.sum(b_split * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1), targets

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training DeepONet",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the MultiONet model.

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

        criterion = self.config.loss
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]
        progress_bar = self.setup_progress_bar(epochs, position, description)
        self.train()
        optimizer.train()

        self.setup_checkpoint()

        for epoch in progress_bar:
            self.epoch(train_loader, criterion, optimizer)

            scheduler.step()

            self.validate(
                epoch=epoch,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                progress_bar=progress_bar,
                total_epochs=epochs,
                multi_objective=multi_objective,
            )

        progress_bar.close()
        self.n_epochs = epoch + 1
        # Schedule-free optimizers hold the weights to evaluate and save in eval mode.
        optimizer.eval()
        self.get_checkpoint(test_loader, criterion)

    def setup_criterion(self) -> callable:
        """
        Utility function to set up the loss function for training.

        Returns:
            callable: The loss function.
        """
        crit = nn.MSELoss(reduction="sum")
        if hasattr(self.config, "masses") and self.config.masses is not None:
            weights = (1.0, self.config.massloss_factor)
            crit = mass_conservation_loss(
                self.config.masses, crit, weights, self.device
            )
        return crit

    def epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform one training epoch.

        Args:
            data_loader (DataLoader): The DataLoader object containing the training data.
            criterion (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            float: The total loss for the training step.
        """
        for batch in data_loader:
            branch_input, trunk_input, targets = batch
            optimizer.zero_grad()
            outputs, targets = self((branch_input, trunk_input, targets))
            loss = criterion(outputs, targets)
            loss.backward()
            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
            optimizer.step()

    def create_dataloader(
        self,
        data: np.ndarray,
        timesteps: np.ndarray,
        batch_size: int,
        shuffle: bool,
        dataset_params: np.ndarray | None,
        params_in_branch: bool,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        n_samples, n_timesteps, n_quantities = data.shape

        if pin_memory and "cuda" not in self.device:
            pin_memory = False

        branch = np.repeat(data[:, 0, :], n_timesteps, axis=0)  # (total, n_q)
        trunk = np.tile(timesteps.reshape(1, -1), (n_samples, 1)).reshape(
            -1, 1
        )  # (total, 1)

        if dataset_params is not None:
            rep_params = np.repeat(dataset_params, n_timesteps, axis=0)
            if params_in_branch:
                branch = np.concatenate([branch, rep_params], axis=1)
            else:
                trunk = np.concatenate([trunk, rep_params], axis=1)

        target = data.reshape(-1, n_quantities)

        # one-time NumPy -> Torch (CPU)
        branch_t = torch.from_numpy(branch).float()
        trunk_t = torch.from_numpy(trunk).float()
        target_t = torch.from_numpy(target).float()

        iterable_ds = FlatBatchIterable(
            branch_t, trunk_t, target_t, batch_size, shuffle
        )

        # batch_size=None because dataset is already batching
        loader = DataLoader(
            iterable_ds,
            batch_size=None,
            num_workers=num_workers,  # 0 is fine; workers add little here
            pin_memory=pin_memory,
            persistent_workers=False,
            # Own generator, so iterating never draws from the shared global RNG.
            generator=torch.Generator(),
        )
        return loader

    def forward_from_state(
        self,
        state: torch.Tensor,
        times: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the states at several time inputs from one initial state.

        Packs the branch and trunk inputs as create_dataloader does, with the
        parameters on the side set by params_branch. See
        AbstractSurrogateModel.forward_from_state.
        """
        B, N = state.shape
        T = times.shape[0]
        branch = state.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
        trunk = times.view(1, T, 1).expand(B, T, 1).reshape(B * T, 1)
        if params is not None:
            p = params.unsqueeze(1).expand(B, T, params.shape[1]).reshape(B * T, -1)
            if self.config.params_branch:
                branch = torch.cat([branch, p], dim=1)
            else:
                trunk = torch.cat([trunk, p], dim=1)
        out, _ = self((branch, trunk, None))
        return out.reshape(B, T, N)

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray,
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
            params_in_branch=self.config.params_branch,
            num_workers=nw,
        )

        test_loader = None
        if dataset_test is not None:
            test_loader = self.create_dataloader(
                dataset_test,
                timesteps,
                batch_size,
                False,
                dataset_params=dataset_test_params,
                params_in_branch=self.config.params_branch,
                num_workers=nw,
            )

        val_loader = None
        if dataset_val is not None:
            val_loader = self.create_dataloader(
                dataset_val,
                timesteps,
                batch_size,
                False,
                dataset_params=dataset_val_params,
                params_in_branch=self.config.params_branch,
                num_workers=nw,
            )

        return train_loader, test_loader, val_loader


AbstractSurrogateModel.register(MultiONet)


class MultiONetResidual(MultiONet):
    """
    MultiONet variant that predicts the change from the initial state.

    The branch input starts with the initial state, so adding its first N
    columns back onto the output leaves the network to learn the deviation.
    """

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, targets = super().forward(inputs)
        branch_input = inputs[0].to(self.device, non_blocking=True)
        return outputs + branch_input[:, : self.N], targets


AbstractSurrogateModel.register(MultiONetResidual)
