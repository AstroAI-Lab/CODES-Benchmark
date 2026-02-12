from dataclasses import dataclass

from torch import nn


@dataclass
class AbstractSurrogateBaseConfig:
    """
    Base configuration for the AbstractSurrogate model.

    This class defines shared attributes and methods for surrogate models.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        regularization_factor (float): Regularization coefficient, applied as weight decay.
        optimizer (str): Type of optimizer to use. Supported options: adamw, sgd.
        momentum (float): Momentum factor for the optimizer (used only if optimizer == "sgd").
        scheduler (str): Type of learning rate scheduler to use.
            - "schedulefree": Use schedulefree optimizer.
            - "cosine": Use cosine annealing scheduler.
            - "poly": Use polynomial decay scheduler.
        poly_power (float): Power for polynomial decay scheduler (used only if scheduler == "poly").
        eta_min (float): Multiplier for minimum learning rate for cosine annealing scheduler (used only if scheduler == "cosine").
        activation (nn.Module | None): Activation module used in the model.
        loss_function (nn.Module | None): Loss function used for training.
        beta (float): Beta parameter forwarded to SmoothL1Loss.
    """

    learning_rate: float = 3e-4
    regularization_factor: float = 0.0
    optimizer: str = "adamw"  # Options: "adamw", "sgd"
    momentum: float = 0.0  # Used only if optimizer == "sgd"
    scheduler: str = "cosine"  # Options: "schedulefree", "cosine", "poly"
    poly_power: float = 0.9  # Used only if scheduler == "poly"
    eta_min: float = 1e-1  # Used only if scheduler == "cosine"
    activation: nn.Module | None = None
    loss_function: nn.Module | None = None  # Options: nn.MSELoss(), nn.SmoothL1Loss()
    beta: float = 0.0  # Used only if loss_function == nn.SmoothL1Loss()

    def __post_init__(self) -> None:
        if self.activation is None:
            self.activation = nn.ReLU()
        if self.loss_function is None:
            self.loss_function = nn.MSELoss()

    @property
    def loss(self) -> nn.Module:
        """
        Returns the loss function to be used for training.

        If the loss function is nn.SmoothL1Loss, it returns an instance with the specified beta.
        Otherwise, it returns the loss function as is.
        """
        if isinstance(self.loss_function, nn.SmoothL1Loss):
            return nn.SmoothL1Loss(beta=self.beta)
        return self.loss_function
