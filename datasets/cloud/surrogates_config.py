from dataclasses import dataclass

from torch import nn


@dataclass
class MultiONetConfig:
    """Model config for MultiONet for the cloud dataset"""

    # cloud_final_multionet, trial 171
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.Mish()
    branch_hidden_layers: int = 8
    hidden_size: int = 100
    output_factor: int = 50
    trunk_hidden_layers: int = 1
    # cloud_final_fine, trial 26
    poly_power: float = 0.647
    learning_rate: float = 0.00374
    regularization_factor: float = 4.64e-05


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the cloud dataset"""

    # cloud_final_latentneuralode, trial 237
    scheduler: str = "poly"
    optimizer: str = "SGD"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.SiLU()
    coder_layers: int = 8
    coder_width: int = 80
    latent_features: int = 1
    ode_layers: int = 3
    ode_tanh_reg: bool = False
    ode_width: int = 250
    # cloud_final_fine, trial 14
    poly_power: float = 3.43
    momentum: float = 0.143
    learning_rate: float = 3.88e-05
    regularization_factor: float = 3.41


@dataclass
class FullyConnectedConfig:
    """Model config for FullyConnected for the cloud dataset"""

    # cloud_final_fullyconnected, trial 114
    scheduler: str = "cosine"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.SmoothL1Loss()
    activation: nn.Module = nn.PReLU()
    hidden_size: int = 310
    num_hidden_layers: int = 1
    # cloud_final_fine, trial 37
    # Tuned as an absolute learning rate of 0.00266; CODES >= 0.2.0 reads eta_min as a
    # fraction of learning_rate.
    eta_min: float = 0.00266 / 0.00215
    # beta: float = 8.39  # tuned, but CODES < 0.2.0 trained with the default 1.0
    learning_rate: float = 0.00215
    regularization_factor: float = 3.42e-04


@dataclass
class LatentPolyConfig:
    """Model config for LatentPoly for the cloud dataset"""

    # cloud_final_latentpoly, trial 135
    scheduler: str = "poly"
    optimizer: str = "AdamW"
    loss_function: nn.Module = nn.MSELoss()
    activation: nn.Module = nn.GELU()
    coder_layers: int = 2
    coder_width: int = 290
    degree: int = 6
    latent_features: int = 2
    # cloud_final_fine, trial 2
    poly_power: float = 5.18
    learning_rate: float = 9.25e-05
    regularization_factor: float = 2.05e-04


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     loss_function: nn.Module = nn.SmoothL1Loss()
#     optimizer: str = "adamw"
#     scheduler: str = "schedulefree"
#     # other params from cloud_tuning_fine, trial 81:
#     beta: float = 1.78
#     branch_hidden_layers: int = 6
#     hidden_size = 170
#     output_factor: int = 80
#     trunk_hidden_layers: int = 7
#     learning_rate: float = 8.3e-4
#     regularization_factor: float = 9.6e-06
#     activation: nn.Module = nn.Tanh()


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     loss_function: nn.Module = nn.MSELoss()
#     optimizer: str = "adamw"
#     scheduler: str = "schedulefree"
#     ode_tanh_reg: bool = False
#     # other params from cloud_tuning_fine, trial 35:
#     latent_features: int = 14
#     coder_layers: int = 4
#     coder_width: int = 70
#     ode_width: int = 30
#     ode_layers: int = 2
#     learning_rate: float = 7.1e-03
#     regularization_factor: float = 3.4e-03
#     activation: nn.Module = nn.Tanh()


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     loss_function: nn.Module = nn.SmoothL1Loss()
#     optimizer: str = "adamw"
#     scheduler: str = "schedulefree"
#     # other params from cloud_tuning_fine, trial 77
#     beta = 0.817
#     hidden_size: int = 380
#     num_hidden_layers: int = 2
#     learning_rate: float = 6.8e-03
#     regularization_factor: float = 8.7e-06
#     activation: nn.Module = nn.ReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     loss_function: nn.Module = nn.MSELoss()
#     optimizer: str = "adamw"
#     scheduler: str = "schedulefree"
#     # other params from cloud_tuning_fine, trial 89:
#     degree: int = 3
#     latent_features: int = 9
#     coder_layers: int = 3
#     coder_width: int = 210
#     learning_rate: float = 2.2e-5
#     regularization_factor: float = 1.7e-03


# @dataclass
# class MultiONetConfig:
#     """Model config for MultiONet for the simple_ode dataset"""

#     # cloud, trial 69
#     branch_hidden_layers: int = 1
#     trunk_hidden_layers: int = 9
#     hidden_size: int = 225
#     output_factor: int = 63
#     learning_rate: float = 4e-5  # optimal for ~4000 epochs
#     activation: nn.Module = nn.Tanh()


# @dataclass
# class LatentNeuralODEConfig:
#     """Model config for LatentNeuralODE for the simple_ode dataset"""

#     # cloud, trial 63
#     latent_features: int = 6
#     coder_layers: int = 2
#     coder_width: int = 103
#     learning_rate: float = 3e-4
#     ode_layers: int = 4
#     ode_width: int = 197
#     ode_tanh_reg: bool = True
#     activation: nn.Module = nn.SiLU()
#     model_version: str = "v2"


# @dataclass
# class FullyConnectedConfig:
#     """Model config for FullyConnected for the simple_ode dataset"""

#     # cloud, trial 44
#     hidden_size: int = 261
#     num_hidden_layers: int = 1
#     learning_rate: float = 1e-4
#     activation: nn.Module = nn.LeakyReLU()


# @dataclass
# class LatentPolyConfig:
#     """Model config for LatentPoly for the simple_ode dataset"""

#     # cloud, trial 92
#     latent_features: int = 9
#     degree: int = 5
#     learning_rate: float = 3e-4
#     coder_layers: int = 1
#     coder_width: int = 86
#     activation: nn.Module = nn.Mish()
