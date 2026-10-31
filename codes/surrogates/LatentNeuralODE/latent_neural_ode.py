import contextlib

import numpy as np
import torch
import torchode as to
from torch import Tensor, nn
from torch.utils.data import DataLoader

from codes.surrogates.AbstractSurrogate import AbstractSurrogateModel
from codes.utils import time_execution

from .latent_neural_ode_config import LatentNeuralODEBaseConfig
from .utilities import FlatSeqBatchIterable


class LatentNeuralODE(AbstractSurrogateModel):
    """
    LatentNeuralODE represents a latent neural ODE model.
    It includes an encoder, decoder, and neural ODE. Fixed parameters can be injected either
    into the encoder or later into the ODE network, controlled by config.encode_params.

    Args:
        device (str | None): Device for training (e.g. 'cpu', 'cuda:0').
        n_quantities (int): Number of quantities.
        n_timesteps (int): Number of timesteps.
        n_parameters (int): Number of fixed parameters (default 0).
        config (dict | None): Configuration for the model.
    """

    def __init__(
        self,
        device: str | None = None,
        n_quantities: int = 29,
        n_timesteps: int = 100,
        n_parameters: int = 0,
        training_id: str | None = None,
        config: dict | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            device=device,
            n_quantities=n_quantities,
            n_timesteps=n_timesteps,
            training_id=training_id,
            config=config,
        )
        self.config = LatentNeuralODEBaseConfig(**self.config)
        self.n_parameters = n_parameters
        # Instantiate the model wrapper with the additional n_parameters.
        self.model = ModelWrapper(
            config=self.config,
            n_quantities=n_quantities,
            n_parameters=n_parameters,
            n_timesteps=self.n_timesteps,
            dtype=dtype,
        ).to(device)
        self.dtype = dtype

    def forward(self, inputs):
        """
        Forward pass through the model.
        Expects inputs to be either (data, timesteps) or (data, timesteps, params).
        """
        inputs = tuple(
            (
                x.to(self.device, dtype=self.dtype, non_blocking=True)
                if isinstance(x, Tensor)
                else x
            )
            for x in inputs
        )
        x, t_range, params = inputs
        x0 = x[:, 0, :]
        latent_prediction = self.model(x0, t_range, params)
        return latent_prediction, x

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
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:

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
        if pin_memory and "cuda" not in self.device:
            pin_memory = False

        data_t = torch.from_numpy(data).float()
        t_t = torch.from_numpy(timesteps).float()
        if dataset_params is not None:
            params_t = torch.from_numpy(dataset_params).float()
        else:
            params_t = None

        ds = FlatSeqBatchIterable(data_t, t_t, params_t, batch_size, shuffle)

        return DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            # Own generator, so iterating never draws from the shared global RNG.
            generator=torch.Generator(),
        )

    @time_execution
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentNeuralODE",
        multi_objective: bool = False,
    ) -> None:
        """
        Train the LatentNeuralODE model.

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
        criterion = self.config.loss

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        optimizer.train()

        self.setup_checkpoint()

        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                batch = tuple(
                    (
                        x.to(device=self.device, non_blocking=True)
                        if isinstance(x, Tensor)
                        else x
                    )
                    for x in batch
                )
                x_true, t_range, params = batch
                optimizer.zero_grad()

                # forward pass
                x_pred, x_true = self((x_true, t_range, params))

                # total loss now takes params into account for identity term
                loss = self.model.total_loss(x_true, x_pred, params, criterion)
                loss.backward()
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.parameters(), self.config.gradient_clip
                    )
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

    @time_execution
    def fit_profile(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        position: int = 0,
        description: str = "Training LatentNeuralODE",
        multi_objective: bool = False,
    ) -> None:
        from torch.profiler import ProfilerActivity, profile, record_function

        optimizer, scheduler = self.setup_optimizer_and_scheduler(epochs)
        criterion = self.config.loss

        loss_length = (epochs + self.update_epochs - 1) // self.update_epochs
        self.train_loss, self.test_loss, self.MAE = [
            np.zeros(loss_length) for _ in range(3)
        ]

        progress_bar = self.setup_progress_bar(epochs, position, description)

        self.model.train()
        # If optimizer.train() is a no-op or not needed, consider removing it;
        # leave it only if it's a custom wrapper that actually requires it.
        # Standard PyTorch optimizers do not have a .train() method.
        with contextlib.suppress(AttributeError):
            optimizer.train()

        self.setup_checkpoint()

        profiled = False  # flag to do profiling only once
        for epoch in progress_bar:
            for i, batch in enumerate(train_loader):
                batch = tuple(
                    (
                        x.to(device=self.device, non_blocking=True)
                        if isinstance(x, Tensor)
                        else x
                    )
                    for x in batch
                )
                x_true, t_range, params = batch

                # Profile only the first batch of the first epoch
                if not profiled and epoch == 2 and i == 1:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=True,  # optional: deeper stack traces
                        profile_memory=True,  # track memory allocs
                    ) as prof:
                        with record_function("zero_grad"):
                            optimizer.zero_grad()

                        with record_function("model_forward"):
                            x_pred, x_true = self((x_true, t_range, params))

                        with record_function("loss_compute"):
                            loss = self.model.total_loss(
                                x_true, x_pred, params, criterion
                            )

                        with record_function("backward"):
                            loss.backward()

                        with record_function("optimizer_step"):
                            optimizer.step()

                    # Advance scheduler if your original logic expects it here
                    scheduler.step()

                    # Output profiling summary
                    print("=== Profiler summary for epoch 0 batch 0 ===")
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=60
                        )
                    )

                    # Export trace for timeline inspection (e.g., chrome://tracing or TensorBoard)
                    prof.export_chrome_trace(f"prof_trace_epoch{epoch}_batch{i}.json")

                    profiled = True  # don't profile again

                else:
                    # Normal training step
                    optimizer.zero_grad()
                    x_pred, x_true = self((x_true, t_range, params))
                    loss = self.model.total_loss(x_true, x_pred, params, criterion)
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.parameters(), self.config.gradient_clip
                        )
                    optimizer.step()

            if not (profiled and epoch == 0):
                # Only step here if you didn't already step inside profiled block
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


class ModelWrapper(nn.Module):
    """
    Wraps the encoder, decoder, and neural ODE in three distinct modes:

    1. **No parameters** (n_parameters=0)
       - Encoder: input = state_dim
       - ODE: latent_dim -> latent_dim (the solver always evolves the latent state)
       - Decoder: latent_dim -> output dimensions

    2. **encode_params=True**
       - Encoder: input = state_dim + param_dim
       - ODE: latent_dim -> latent_dim
       - Decoder: latent_dim -> output dimensions

    3. **encode_params=False**
       - Encoder: input = state_dim
       - Base ODE: (latent_dim + param_dim) -> latent_dim
       - Wrapped in ODEWithParams so that solver state = latent_dim
       - Decoder: latent_dim -> output dimensions
    """

    def __init__(
        self,
        config,
        n_quantities: int,
        n_parameters: int = 0,
        n_timesteps: int = 101,
        dtype: torch.dtype = torch.float64,
        use_pid: bool = False,  # switch from Integral to PID
        adjoint_type: str = "autodiff",  # "autodiff" | "backsolve" | "joint_backsolve"
    ):
        super().__init__()
        self.config = config
        self.n_parameters = n_parameters
        latent_dim = config.latent_features
        self.loss_weights = getattr(config, "loss_weights", [100.0, 1.0, 1.0, 1.0])
        self.dtype = dtype
        self.n_timesteps = n_timesteps
        self.l2 = nn.MSELoss()

        # --- Build encoder ---
        if n_parameters == 0:
            enc_in = n_quantities
        elif config.encode_params:
            enc_in = n_quantities + n_parameters
        else:
            enc_in = n_quantities
        if self.config.model_version == "v1":
            self.encoder = OldEncoder(
                in_features=enc_in,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
                dtype=dtype,
            )
        elif self.config.model_version == "v2":
            self.encoder = Encoder(
                in_features=enc_in,
                latent_features=latent_dim,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Unknown model version {self.config.model_version}. "
                "Supported versions: 'v1', 'v2'."
            )

        # --- Build ODE ---
        if n_parameters == 0 or config.encode_params:
            # solver state = latent_dim
            ode_net = ODE(
                input_shape=latent_dim,
                output_shape=latent_dim,
                activation=config.activation,
                ode_layers=config.ode_layers,
                ode_width=config.ode_width,
                tanh_reg=config.ode_tanh_reg,
                dtype=dtype,
            )
            ode_module = ode_net
        else:
            # wrap base ODE to inject params: solver sees latent_dim state
            base_ode = ODE(
                input_shape=latent_dim + n_parameters,
                output_shape=latent_dim,
                activation=config.activation,
                ode_layers=config.ode_layers,
                ode_width=config.ode_width,
                tanh_reg=config.ode_tanh_reg,
                dtype=dtype,
            )
            ode_module = ODEWithParams(base_ode, n_parameters, latent_dim)

        # --- ODE and solver setup ---
        self.ode = ode_module
        term = to.ODETerm(self.ode)
        step = to.Tsit5(term=term)  # or expose choice of step method if desired

        # choose controller
        if use_pid:
            # tune pcoeff/icoeff/dcoeff as hyperparams if needed
            controller = to.PIDController(
                atol=config.atol,
                rtol=config.rtol,
                pcoeff=getattr(config, "pid_pcoeff", 0.2),
                icoeff=getattr(config, "pid_icoeff", 0.5),
                dcoeff=getattr(config, "pid_dcoeff", 0.0),
                term=term,
            )
        else:
            controller = to.IntegralController(
                atol=config.atol, rtol=config.rtol, term=term
            )

        # choose adjoint/backprop method
        if adjoint_type == "autodiff":
            self.solver = to.AutoDiffAdjoint(step, controller)
        elif adjoint_type == "backsolve":
            self.solver = to.BacksolveAdjoint(term, step, controller)
        elif adjoint_type == "joint_backsolve":
            self.solver = to.JointBacksolveAdjoint(term, step, controller)
        else:
            raise ValueError(f"Unknown adjoint_type {adjoint_type}")

        # --- Build decoder ---
        if self.config.model_version == "v1":
            self.decoder = OldDecoder(
                out_features=n_quantities,
                latent_features=latent_dim,
                layers_factor=self.config.layers_factor,
                activation=self.config.activation,
                dtype=dtype,
            )
        elif self.config.model_version == "v2":
            self.decoder = Decoder(
                out_features=n_quantities,
                latent_features=latent_dim,
                coder_layers=config.coder_layers,
                coder_width=config.coder_width,
                activation=config.activation,
                dtype=dtype,
            )

    def forward(self, x0: Tensor, t_range: Tensor, params: Tensor = None):
        # encode initial state
        if self.n_parameters > 0 and self.config.encode_params:
            assert params is not None
            enc_in = torch.cat([x0, params], dim=1)
        else:
            enc_in = x0
        z0 = self.encoder(enc_in)  # .to(torch.float64)

        # if using closure to inject params
        if self.n_parameters > 0 and not self.config.encode_params:
            assert params is not None
            self.ode.set_params(params)

        # use float64 for ODE solver
        z0 = z0.to(torch.float64)  # solver state must be double
        t_eval = t_range.repeat(x0.size(0), 1).to(torch.float64)
        assert z0.dtype == torch.float64, f"z0 dtype {z0.dtype}"
        assert t_eval.dtype == torch.float64, f"t_eval dtype {t_eval.dtype}"

        sol = self.solver.solve(to.InitialValueProblem(y0=z0, t_eval=t_eval))

        latent_traj = sol.ys  # [timesteps, batch, latent_dim]

        # decode
        return self.decoder(latent_traj.to(self.dtype))

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
        if self.config.encode_params and params is not None:
            enc_input = torch.cat([x0, params], dim=1)
        else:
            enc_input = x0

        # encode-decode
        z0 = self.encoder(enc_input)
        x0_hat = self.decoder(z0)
        return self.l2(x0, x0_hat)


class ODE(nn.Module):
    """
    Neural ODE module defining the function for latent dynamics.
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: nn.Module,
        ode_layers: int,
        ode_width: int,
        tanh_reg: bool,
        dtype=torch.float64,
    ):
        super().__init__()
        self.tanh_reg = tanh_reg
        self.reg_factor = nn.Parameter(torch.tensor(1.0))
        self.activation = activation
        self.dtype = dtype
        layers = []
        layers.append(nn.Linear(input_shape, ode_width, dtype=dtype))
        layers.append(activation)
        for _ in range(ode_layers):
            layers.append(nn.Linear(ode_width, ode_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(ode_width, output_shape, dtype=dtype))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x):
        # Expect solver to always pass float64 state
        if x.dtype != torch.float64:
            raise RuntimeError(
                f"ODE.forward expected float64 input state, got {x.dtype}"
            )

        # Downcast for MLP computation
        x32 = x.to(torch.float32)
        out32 = self.mlp(x32)  # float32

        if self.tanh_reg:
            reg32 = self.reg_factor.to(torch.float32)
            activated32 = reg32 * torch.tanh(out32 / reg32)
            out64 = activated32.to(torch.float64)
            if out64.dtype != torch.float64:
                raise RuntimeError("Output not float64 after upcast")
            return out64

        out64 = out32.to(torch.float64)
        return out64


class ODEWithParams(nn.Module):
    """
    Wraps a base ODE module so that parameters are injected as a constant.
    The solver sees only the latent state y (dim = latent_dim),
    but ODEWithParams.forward will concatenate y with p to compute dy/dt.
    """

    def __init__(self, base_ode: nn.Module, n_parameters: int, latent_dim: int):
        super().__init__()
        self.base_ode = base_ode
        self.latent_dim = latent_dim
        self.n_parameters = n_parameters
        self.p_const: Tensor | None = None

    def set_params(self, params: Tensor):
        # params: [batch, n_parameters]
        self.p_const = params

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        # y: [batch, latent_dim]
        assert self.p_const is not None, "Call set_params() before solving."
        # concat state with params
        inp = torch.cat([y, self.p_const], dim=1)  # [batch, latent_dim + n_parameters]
        # compute derivative of latent only
        dy = self.base_ode(t, inp)  # [batch, latent_dim]
        return dy


class Encoder(nn.Module):
    """
    Fully connected encoder that maps input features to a latent space.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: nn.Module = nn.ReLU(),
        dtype=torch.float64,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, coder_width, dtype=dtype))
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(nn.Linear(coder_width, coder_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(coder_width, latent_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Decoder(nn.Module):
    """
    Fully connected decoder that maps the latent space back to the output.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        coder_layers: int = 3,
        coder_width: int = 32,
        activation: nn.Module = nn.ReLU(),
        dtype=torch.float64,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_features, coder_width, dtype=dtype))
        layers.append(activation)
        for _ in range(coder_layers - 1):
            layers.append(nn.Linear(coder_width, coder_width, dtype=dtype))
            layers.append(activation)
        layers.append(nn.Linear(coder_width, out_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class OldEncoder(nn.Module):
    """
    Old encoder using a fixed 4-2-1 structure.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        # Example: coder_layers = [4, 2, 1] scaled by layers_factor.
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        layers = []
        layers.append(nn.Linear(in_features, coder_layers[0], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], latent_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class OldDecoder(nn.Module):
    """
    Old decoder corresponding to the old encoder.
    """

    def __init__(
        self,
        out_features: int,
        latent_features: int = 5,
        layers_factor: int = 8,
        activation: nn.Module = nn.ReLU(),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        coder_layers = [4 * layers_factor, 2 * layers_factor, 1 * layers_factor]
        # Reverse the order for the decoder.
        coder_layers.reverse()
        layers = []
        layers.append(nn.Linear(latent_features, coder_layers[0], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[0], coder_layers[1], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[1], coder_layers[2], dtype=dtype))
        layers.append(activation)
        layers.append(nn.Linear(coder_layers[2], out_features, dtype=dtype))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


AbstractSurrogateModel.register(LatentNeuralODE)
