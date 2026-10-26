from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the primordial_parametric dataset"""

    # primordial_parametric_final_multionet, trial 18
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.PReLU()
    branch_hidden_layers: int = 5
    hidden_size: int = 50
    output_factor: int = 74
    params_branch: bool = True
    trunk_hidden_layers: int = 1
    # primordial_parametric_final_fine, trial 1
    poly_power: float = 2.57
    learning_rate: float = 0.00519
    regularization_factor: float = 0.00138


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the primordial_parametric dataset"""

    # primordial_parametric_final_latentneuralode, trial 234
    scheduler: str = "cosine"
    optimizer: str = "SGD"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.SiLU()
    coder_layers: bool = True
    coder_width: int = 360
    encode_params: bool = False
    latent_features: int = 10
    ode_layers: int = 8
    ode_tanh_reg: bool = True
    ode_width: int = 220
    # primordial_parametric_final_fine, trial 1
    # Tuned as an absolute learning rate of 5.32e-04; CODES >= 0.2.0 reads eta_min as a
    # fraction of learning_rate.
    eta_min: float = 5.32e-04 / 2.78e-06
    learning_rate: float = 2.78e-06
    momentum: float = 0.331
    regularization_factor: float = 6.36e-05


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the primordial_parametric dataset"""

    # primordial_parametric_final_fullyconnected, trial 1
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.Mish()
    hidden_size: int = 470
    num_hidden_layers: int = 3
    # primordial_parametric_final_fine, trial 1
    poly_power: float = 0.163
    # beta: float = 39.5  # tuned, but CODES < 0.2.0 trained with the default 1.0
    learning_rate: float = 1.29e-04
    regularization_factor: float = 0.0333


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the primordial_parametric dataset"""

    # primordial_parametric_final_latentpoly, trial 16
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.ELU()
    coder_layers: bool = True
    coder_width: int = 700
    coeff_network: bool = False
    degree: int = 3
    latent_features: int = 7
    # primordial_parametric_final_fine
    poly_power: float = 0.264
    learning_rate: float = 5.96e-06
    regularization_factor: float = 0.0510


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     # primordialparams, trial 56

#     branch_hidden_layers: int = 2
#     trunk_hidden_layers: int = 7
#     hidden_size: int = 164
#     output_factor: int = 63
#     learning_rate: float = 4e-5
#     activation: nn.Module = nn.GELU()
#     params_branch: bool = True


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     # primordialparams, trial 75

#     latent_features: int = 5
#     coder_layers: int = 4
#     coder_width: int = 268
#     learning_rate: float = 4e-5
#     ode_layers: int = 9
#     ode_width: int = 125
#     ode_tanh_reg: bool = False
#     activation: nn.Module = nn.Tanh()
#     model_version: str = "v2"
#     encode_params: bool = False


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     # primordialparams, trial 40

#     hidden_size: int = 629
#     num_hidden_layers: int = 1
#     learning_rate: float = 6e-5
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     # primordialparams, trial 32

#     latent_features: int = 10
#     degree: int = 7
#     learning_rate: float = 5e-4
#     coder_layers: int = 2
#     coder_width: int = 267
#     activation: nn.Module = nn.LeakyReLU()
#     coeff_network: bool = False
