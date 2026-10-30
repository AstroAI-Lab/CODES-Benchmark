from dataclasses import dataclass

from codes.surrogates.AbstractSurrogate import AbstractSurrogateBaseConfig


@dataclass
class LatentNeuralODEBaseConfig(AbstractSurrogateBaseConfig):
    """
    Configuration for the LatentNeuralODE surrogate model.

    This dataclass defines all hyperparameters required to configure the architecture
    and training of a latent neural ODE. It supports both fixed and flexible encoder/decoder
    structures depending on the `model_version` flag.

    Attributes:
        model_version (str): Indicates model architecture style.
            - "v1": Fixed structure (e.g., 4-2-1 layout scaled by factor).
            - "v2": Fully connected architecture with flexible depth/width.
        latent_features (int): Size of the latent space (z-dimension).
        layers_factor (int): Scaling factor for the number of neurons in the encoder/decoder.
            - Used in "v1" to determine layer widths based on coder_hidden.
        coder_layers (int): Number of hidden layers in both encoder and decoder (used in v2).
        coder_width (int): Number of neurons per hidden layer in encoder/decoder (used in v2).
        ode_layers (int): Number of hidden layers in the ODE module.
        ode_width (int): Number of neurons in each hidden layer of the ODE module.
        ode_tanh_reg (bool): Whether to apply tanh regularization in the ODE output.
        rtol (float): Relative tolerance for the ODE solver.
        atol (float): Absolute tolerance for the ODE solver.
        encode_params (bool): Whether to encode parameters in the encoder.
            - If False, parameters are passed after the encoder, as additional inputs to the ODE network.
        loss_weights (tuple): Weights of the trajectory, identity, first derivative and
            second derivative loss terms.
    """

    model_version: str = "v2"
    latent_features: int = 5
    layers_factor: int = 8
    coder_layers: int = 3
    coder_width: int = 64
    ode_layers: int = 4
    ode_width: int = 64
    ode_tanh_reg: bool = True
    rtol: float = 1e-6
    atol: float = 1e-6
    encode_params: bool = False
    loss_weights: tuple = (100.0, 1.0, 1.0, 1.0)
