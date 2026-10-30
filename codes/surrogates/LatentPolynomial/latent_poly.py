import numpy as np
import torch
from torch import Tensor, nn

# from torch.optim import Adam
from torch.utils.data import DataLoader

from codes.surrogates import Decoder as NewDecoder
from codes.surrogates import Encoder as NewEncoder
from codes.surrogates import FlatSeqBatchIterable
from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.utils import time_execution

from .latent_poly_config import LatentPolynomialBaseConfig


class LatentPoly(AbstractSurrogateModel):
    """
    LatentPoly class for training a polynomial model on latent space trajectories.

    This model includes an encoder, decoder, and a learnable polynomial applied on the latent space.
    The architecture is chosen based on the version flag in the configuration.

    Attributes:
        config (LatentPolynomialBaseConfig): The configuration for the model.
        model (PolynomialModelWrapper): The wrapped model (encoder, decoder, polynomial).
        device (str): Device for training.
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
        self.config = LatentPolynomialBaseConfig(**self.config)
        self.config.n_quantities = n_quantities
        # For backward compatibility: if using v1, compute the fixed width list.
        if self.config.model_version == "v1":
            coder_layers_old = [
                4 * self.config.layers_factor,
                2 * self.config.layers_factor,
                1 * self.config.layers_factor,
            ]
            self.config.width_list = coder_layers_old
        # # Update the number of input features based on whether we encode parameters.
        # if self.config.coeff_network:
        #     self.config.in_features = (
        #         n_quantities  # No parameter concatenation to the encoder input.
        #     )
        # else:
        #     self.config.in_features = n_quantities + n_parameters
        self.model = PolynomialModelWrapper(
            config=self.config,
            device=self.device,
            n_parameters=n_parameters,
            n_timesteps=n_timesteps,
        )

    def forward(self, inputs) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            inputs (tuple): Tuple containing the input tensor and timesteps.
                If fixed parameters are provided, the tuple is (data, timesteps, params).

        Returns:
            tuple[Tensor, Tensor]: (Predictions, Targets)
        """
        inputs = tuple(
            x.to(self.device, non_blocking=True) if isinstance(x, Tensor) else x
            for x in inputs
        )
        x, t_range, params = inputs
        return self.model(x, t_range, params), x

    def create_dataloader(
        self,
        data: np.ndarray,  # (N, T, Q)
        timesteps: np.ndarray,  # (T,)
        batch_size: int,
        shuffle: bool,
        dataset_params: np.ndarray | None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        if pin_memory and "cuda" not in self.device:
            pin_memory = False

        data_t = torch.from_numpy(data).float()
        t_t = torch.from_numpy(timesteps).float()
        params_t = (
            torch.from_numpy(dataset_params).float()
            if dataset_params is not None
            else None
        )

        ds = FlatSeqBatchIterable(data_t, t_t, params_t, batch_size, shuffle)

        return DataLoader(
            ds,
            batch_size=None,  # dataset yields full batches
            num_workers=num_workers,  # 0 usually fine
            pin_memory=pin_memory,
            persistent_workers=False,
            # Own generator, so iterating never draws from the shared global RNG.
            generator=torch.Generator(),
        )

    def prepare_data(
        self,
        dataset_train: np.ndarray,
        dataset_test: np.ndarray | None,
        dataset_val: np.ndarray | None,
        timesteps: np.ndarray,
        batch_size: int = 128,
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
        description: str = "Training LatentPoly",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the LatentPoly model.

        Args:
            train_loader (DataLoader): The data loader for the training data.
            test_loader (DataLoader): The data loader for the test data.
            epochs (int | None): The number of epochs to train the model. If None, uses the value from the config.
            position (int): The position of the progress bar.
            description (str): The description for the progress bar.
            multi_objective (bool): Whether multi-objective optimization is used.
                                    If True, trial.report is not used (not supported by Optuna).
        """
        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]
        criterion = self.config.loss

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        self.setup_checkpoint()

        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                x_true, _, params = batch
                optimizer.zero_grad()
                x_pred, x_true = self(batch)
                loss = self.model.total_loss(x_true, x_pred, params, criterion)
                loss.backward()
                optimizer.step()

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


class PolynomialModelWrapper(nn.Module):
    """
    Wraps the Encoder, Decoder, and learnable Polynomial into a single model.

    If config.coeff_network is True, fixed parameters are not concatenated to the encoder input;
    instead, a coefficient network predicts the polynomial coefficients based on the parameters.
    If False, fixed parameters are concatenated to the encoder input and the polynomial coefficients are learned directly.

    Attributes:
        config (LatentPolynomialBaseConfig): Model configuration.
        loss_weights (list[float]): Weights for the loss terms.
        device (str): Device for training.
        encoder (Module): The encoder network.
        decoder (Module): The decoder network.
        poly (Polynomial): The polynomial module.
        coefficient_net (Module | None): The coefficient network (if config.coeff_network is True).
    """

    def __init__(self, config, device, n_parameters: int = 0, n_timesteps: int = 101):
        super().__init__()
        self.config = config
        self.loss_weights = getattr(config, "loss_weights", [100.0, 1.0, 1.0, 1.0])
        self.device = device
        latent_dim = self.config.latent_features
        self.n_timesteps = n_timesteps
        self.l2 = nn.MSELoss()

        # Use coeff_network as the single switch.
        if self.config.coeff_network and n_parameters > 0:
            encoder_in_features = self.config.n_quantities  # remains n_quantities
        else:
            # When not using the coefficient network, the encoder gets concatenated parameters.
            encoder_in_features = self.config.n_quantities + n_parameters

        # Instantiate encoder and decoder according to model_version.
        if self.config.model_version == "v1":
            self.encoder = OldEncoder(
                in_features=encoder_in_features,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
            ).to(self.device)
            # When parameters are concatenated, adjust the output features accordingly.
            # out_feats = (
            #     self.config.in_features
            #     if not self.config.coeff_network
            #     else self.config.in_features
            # )
            self.decoder = OldDecoder(
                out_features=self.config.n_quantities,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
            ).to(self.device)
        else:
            self.encoder = NewEncoder(
                in_features=encoder_in_features,
                latent_features=latent_dim,
                coder_layers=self.config.coder_layers,
                coder_width=self.config.coder_width,
                activation=self.config.activation,
                dtype=torch.float32,
            ).to(self.device)
            # out_feats = (
            #     self.config.in_features
            #     if not self.config.coeff_network
            #     else self.config.in_features
            # )
            self.decoder = NewDecoder(
                out_features=self.config.n_quantities,
                latent_features=latent_dim,
                coder_layers=self.config.coder_layers,
                coder_width=self.config.coder_width,
                activation=self.config.activation,
                dtype=torch.float32,
            ).to(self.device)
        # Instantiate the polynomial module.
        self.poly = Polynomial(degree=self.config.degree, dimension=latent_dim).to(
            self.device
        )
        # Instantiate coefficient network only if coeff_network is True and parameters exist.
        if self.config.coeff_network and n_parameters > 0:
            self.coefficient_net = nn.Sequential(
                nn.Linear(n_parameters, self.config.coeff_width, dtype=torch.float32),
                self.config.activation,
                *[
                    nn.Sequential(
                        nn.Linear(
                            self.config.coeff_width,
                            self.config.coeff_width,
                            dtype=torch.float32,
                        ),
                        self.config.activation,
                    )
                    for _ in range(self.config.coeff_layers - 1)
                ],
                nn.Linear(
                    self.config.coeff_width,
                    latent_dim * self.config.degree,
                    dtype=torch.float32,
                ),
            ).to(self.device)
        else:
            self.coefficient_net = None

    def forward(self, x, t_range, params=None):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch, timesteps, n_quantities).
            t_range (Tensor): Time range tensor.
            params (Tensor | None): Fixed parameters of shape (batch, n_parameters), if provided.

        Returns:
            Tensor: Predicted trajectory.
        """
        current_batch_size = x.shape[0]
        x0 = x[:, 0, :]
        if self.config.coeff_network and params is not None:
            # Scheme using coefficient network:
            encoder_input = x0  # Encoder gets only the raw data.
        else:
            # Otherwise, concatenate fixed parameters to the encoder input (if provided).
            encoder_input = torch.cat([x0, params], dim=1) if params is not None else x0
        z0 = self.encoder(encoder_input)
        t = t_range.unsqueeze(0).repeat(current_batch_size, 1)
        if self.config.coeff_network and (params is not None):
            # Use the coefficient network to predict polynomial coefficients.
            # Compute time basis using the internal _prepare_t method.
            B = self.poly._prepare_t(t)  # shape (batch, timesteps, degree)
            coef_vec = self.coefficient_net(params)  # shape (batch, latent_dim*degree)
            coef = coef_vec.view(current_batch_size, z0.shape[1], self.config.degree)
            # Multiply basis with predicted coefficients:
            poly_out = torch.bmm(
                B, coef.transpose(1, 2)
            )  # shape (batch, timesteps, latent_dim)
        else:
            poly_out = self.poly(t)
        z_pred = poly_out + z0.unsqueeze(1)
        return self.decoder(z_pred)

    def total_loss(
        self,
        x_true: Tensor,
        x_pred: Tensor,
        params: Tensor = None,
        criterion: nn.Module = nn.MSELoss(),
    ):
        """
        Total loss: weighted sum of trajectory reconstruction, identity, first derivative,
        and second derivative losses. All terms remain in the computation graph.
        """
        w0, w1, w2, w3 = (
            self.loss_weights
        )  # assume these are set in config and are floats

        # primary trajectory loss
        traj_loss = criterion(x_pred, x_true)

        # identity loss (reconstruct x0)
        identity = self.identity_loss(x_true, params)

        # derivative losses: compute once
        d_pred = self.first_derivative(x_pred)
        d_true = self.first_derivative(x_true)
        deriv_loss = self.l2(d_pred, d_true)

        d2_pred = self.second_derivative(x_pred)
        d2_true = self.second_derivative(x_true)
        deriv2_loss = self.l2(d2_pred, d2_true)

        return w0 * traj_loss + w1 * identity + w2 * deriv_loss + w3 * deriv2_loss

    def first_derivative(self, x: Tensor):
        # x: [B, T, F]
        h = 1.0 / self.n_timesteps
        # central differences for interior
        d_center = (x[:, 2:, :] - x[:, :-2, :]) / (2 * h)  # [B, T-2, F]
        # forward/backward for boundaries
        d_first = (x[:, 1:2, :] - x[:, :1, :]) / h  # [B,1,F]
        d_last = (x[:, -1:, :] - x[:, -2:-1, :]) / h  # [B,1,F]
        derivative = torch.cat([d_first, d_center, d_last], dim=1)  # [B,T,F]
        return derivative

    def second_derivative(self, x: Tensor):
        # x: [B, T, F]
        h = 1.0 / self.n_timesteps
        # standard second derivative central
        d2_center = (x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]) / (
            h * h
        )  # [B, T-2, F]
        # one-sided approximations at ends (second-order):
        # at t0: f''(t0) ≈ (2 f0 - 5 f1 + 4 f2 - f3) / h^2
        d2_first = (
            2 * x[:, :1, :] - 5 * x[:, 1:2, :] + 4 * x[:, 2:3, :] - x[:, 3:4, :]
        ) / (h * h)
        # at t_{N-1}: symmetric formula
        d2_last = (
            2 * x[:, -1:, :] - 5 * x[:, -2:-1, :] + 4 * x[:, -3:-2, :] - x[:, -4:-3, :]
        ) / (h * h)
        d2 = torch.cat([d2_first, d2_center, d2_last], dim=1)  # [B,T,F]
        return d2

    def identity_loss(self, x_true: Tensor, params: Tensor = None):
        """
        Calculate the identity loss (Encoder -> Decoder) on the initial state x0.

        Args:
            x_true (Tensor): The full trajectory (batch, timesteps, features).
            params (Tensor | None): Fixed parameters (batch, n_parameters).
        Returns:
            Tensor: The identity loss on x0.
        """
        # only reconstruct the initial state
        x0 = x_true[:, 0, :]
        if not self.config.coeff_network and params is not None:
            params = params.to(self.device)
            enc_input = torch.cat([x0, params], dim=1)
        else:
            enc_input = x0

        # encode-decode
        z0 = self.encoder(enc_input)
        x0_hat = self.decoder(z0)
        return self.l2(x0, x0_hat)


class Polynomial(nn.Module):
    """
    Learnable polynomial model.

    Attributes:
        degree (int): Degree of the polynomial.
        dimension (int): Dimension of the in- and output.
        coef (nn.Linear): Linear layer representing polynomial coefficients.
        t_matrix (Tensor): Time matrix for polynomial evaluation.
    """

    def __init__(self, degree: int, dimension: int):
        super().__init__()
        self.degree = degree
        self.dimension = dimension
        self.coef = nn.Linear(
            in_features=degree, out_features=dimension, bias=False, dtype=torch.float32
        )
        self.t_matrix = None

    def forward(self, t: Tensor):
        """
        Evaluate the polynomial at given timesteps.

        Args:
            t (Tensor): Time tensor.

        Returns:
            Tensor: Evaluated polynomial.
        """
        return self.coef(self._prepare_t(t))

    def _prepare_t(self, t: Tensor):
        """
        Prepare time values in a matrix form.

        Args:
            t (Tensor): Time tensor.

        Returns:
            Tensor: Prepared time matrix.
        """
        t = t[:, None]
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)


class OldEncoder(nn.Module):
    """
    Old encoder network implementing a fixed 4–2–1 structure scaled by layers_factor.

    Args:
        in_features (int): Number of input features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths: [4, 2, 1] scaled by layers_factor.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        layers = []
        layers.append(nn.Linear(in_features, coder_layers[0], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], latent_features, dtype=torch.float32))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the old encoder."""
        return self.mlp(x)


class OldDecoder(nn.Module):
    """
    Old decoder network corresponding to the fixed 4–2–1 encoder.

    Args:
        out_features (int): Number of output features.
        latent_features (int): Dimension of the latent space.
        layers_factor (int): Factor to scale the base widths [4, 2, 1].
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        # Compute the hidden layer widths and reverse the order for decoding.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        coder_layers.reverse()
        layers = []
        layers.append(nn.Linear(latent_features, coder_layers[0], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=torch.float32))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], out_features, dtype=torch.float32))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the old decoder."""
        return self.mlp(x)


AbstractSurrogateModel.register(LatentPoly)
