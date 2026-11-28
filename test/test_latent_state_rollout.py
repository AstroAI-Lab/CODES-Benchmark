import pytest
import torch

from codes.surrogates.LatentNeuralODE.latent_neural_ode import LatentNeuralODE
from codes.surrogates.LatentPolynomial.latent_poly import LatentPoly


@pytest.mark.parametrize("encode_params", [False, True])
def test_latent_neural_ode_state_rollout_matches_forward(encode_params):
    model = LatentNeuralODE(
        device="cpu",
        n_quantities=3,
        n_timesteps=3,
        n_parameters=2,
        config={
            "latent_features": 4,
            "coder_layers": 1,
            "coder_width": 8,
            "ode_layers": 1,
            "ode_width": 8,
            "encode_params": encode_params,
        },
    ).eval()
    state = torch.randn(2, 3)
    params = torch.randn(2, 2)
    times = torch.tensor([0.0, 0.4, 1.0])
    with torch.no_grad():
        expected = model.model(state, times, params)
        actual = model.forward_from_state(state, times, params)
        endpoint = model.forward_from_state(state, times[-1:], params)
        endpoint_expected = model.model(state, torch.tensor([0.0, 1.0]), params)[:, -1:]
    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(endpoint, endpoint_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("coeff_network", [False, True])
def test_latent_poly_state_rollout_matches_forward(coeff_network):
    model = LatentPoly(
        device="cpu",
        n_quantities=3,
        n_timesteps=3,
        n_parameters=2,
        config={
            "latent_features": 4,
            "coder_layers": 1,
            "coder_width": 8,
            "degree": 2,
            "coeff_network": coeff_network,
        },
    ).eval()
    state = torch.randn(2, 3)
    params = torch.randn(2, 2)
    times = torch.tensor([0.0, 0.4, 1.0])
    with torch.no_grad():
        expected = model.model(state.unsqueeze(1), times, params)
        actual = model.forward_from_state(state, times, params)
    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, expected)


def test_latent_state_rollouts_accept_single_nonzero_time_and_zero():
    state = torch.randn(2, 3)
    for model in (
        LatentNeuralODE(
            device="cpu",
            n_quantities=3,
            config={
                "coder_layers": 1,
                "coder_width": 8,
                "ode_layers": 1,
                "ode_width": 8,
            },
        ),
        LatentPoly(
            device="cpu", n_quantities=3, config={"coder_layers": 1, "coder_width": 8}
        ),
    ):
        model.eval()
        with torch.no_grad():
            zero = model.forward_from_state(state, torch.tensor([0.0]))
            endpoint = model.forward_from_state(state, torch.tensor([0.7]))
            expected_zero = model.model(
                state if isinstance(model, LatentNeuralODE) else state.unsqueeze(1),
                torch.tensor([0.0]),
            )
            expected_endpoint = model.model(
                state if isinstance(model, LatentNeuralODE) else state.unsqueeze(1),
                torch.tensor([0.0, 0.7]),
            )[:, -1:]
        assert zero.shape == endpoint.shape == (2, 1, 3)
        assert torch.isfinite(zero).all() and torch.isfinite(endpoint).all()
        torch.testing.assert_close(zero, expected_zero)
        torch.testing.assert_close(endpoint, expected_endpoint, rtol=1e-5, atol=1e-6)
