"""Registry-wide training contracts and focused shared-loop behavior checks."""

import numpy as np
import optuna
import pytest
import torch
from torch import nn

from codes.surrogates import FullyConnected, LatentNeuralODE, MultiONet
from codes.surrogates.training import train_epoch


def _data():
    times = np.linspace(0, 1, 8, dtype=np.float32)
    data = np.stack(
        [
            np.stack(
                (0.2 + 0.08 * i + 0.1 * times, 0.6 - 0.04 * i + 0.06 * times), axis=-1
            )
            for i in range(4)
        ]
    ).astype(np.float32)
    return data, times


def test_epoch_uses_configured_smooth_l1_and_clips_gradients():
    class LinearSurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
            self.config = type("Config", (), {"gradient_clip": 0.25})()

        def forward(self, batch):
            x, y = batch
            return self.weight * x, y

    model = LinearSurrogate()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.SmoothL1Loss(beta=4.0)
    train_epoch(
        model, [(torch.tensor([2.0]), torch.tensor([1.0]))], criterion, optimizer
    )
    # Unclipped gradient is -0.5 for beta=4; clipping to 0.25 gives +0.025.
    assert model.weight.item() == pytest.approx(0.025, abs=1e-6)


def test_fit_validation_scheduler_and_schedulefree_checkpoint_order(monkeypatch):
    torch.manual_seed(10)
    data, times = _data()
    model = FullyConnected(
        device="cpu",
        n_quantities=2,
        n_timesteps=8,
        config={
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "scheduler": "schedulefree",
            "loss_function": nn.SmoothL1Loss(),
            "beta": 0.3,
        },
    )
    model.update_epochs = 2
    model.checkpointing = True
    train, test, _ = model.prepare_data(
        data, data[:2], None, times, batch_size=2, shuffle=False
    )
    events = []
    setup = model.setup_optimizer_and_scheduler
    validate = model.validate
    checkpoint = model.get_checkpoint
    save_checkpoint = model.checkpoint
    checkpoint_epochs = []

    def save_spy(loss, epoch):
        checkpoint_epochs.append(epoch)
        return save_checkpoint(loss, epoch)

    monkeypatch.setattr(model, "checkpoint", save_spy)

    def setup_spy(epochs):
        optimizer, scheduler = setup(epochs)
        original_step = scheduler.step

        def step(*args, **kwargs):
            events.append(("scheduler",))
            return original_step(*args, **kwargs)

        scheduler.step = step
        return optimizer, scheduler

    def validate_spy(**kwargs):
        events.append(("validate", kwargs["epoch"], kwargs["multi_objective"]))
        return validate(**kwargs)

    def checkpoint_spy(loader, criterion):
        # Schedule-free optimizers require evaluation mode to expose checkpoint weights.
        events.append(
            (
                "checkpoint",
                optimizer_ref[0].param_groups[0]["train_mode"],
                criterion.beta,
            )
        )
        return checkpoint(loader, criterion)

    optimizer_ref = []

    def setup_capture(epochs):
        optimizer, scheduler = setup_spy(epochs)
        optimizer_ref.append(optimizer)
        return optimizer, scheduler

    monkeypatch.setattr(model, "setup_optimizer_and_scheduler", setup_capture)
    monkeypatch.setattr(model, "validate", validate_spy)
    monkeypatch.setattr(model, "get_checkpoint", checkpoint_spy)
    model.fit(train, test, 3, multi_objective=True)
    assert events == [
        ("scheduler",),
        ("validate", 0, True),
        ("scheduler",),
        ("validate", 1, True),
        ("scheduler",),
        ("validate", 2, True),
        ("checkpoint", False, 0.3),
    ]
    assert checkpoint_epochs == [0, 2]
    assert model.n_epochs == 3
    assert len(model.train_loss) == len(model.test_loss) == len(model.MAE) == 2
    assert np.isfinite(model.train_loss).all() and np.isfinite(model.test_loss).all()
    assert model.best_epoch in (0, 2)
    assert np.isfinite(model.best_test_loss)


def test_registered_surrogate_small_training_smoke(small_surrogate_factory):
    """Check integration without pinning convergence or requiring shared-loop use."""
    torch.manual_seed(22)
    data, times = _data()
    model = small_surrogate_factory()
    model.update_epochs = 1
    model.checkpointing = True
    train, test, _ = model.prepare_data(
        data[:2], data[2:], None, times, batch_size=2, shuffle=False
    )
    initial = [p.detach().clone() for p in model.parameters()]
    model.fit(train, test, 2)

    assert model.n_epochs == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, model.parameters())
    )
    assert np.isfinite(model.fit_duration) and model.fit_duration >= 0
    for history in (model.train_loss, model.test_loss, model.MAE):
        assert len(history) == 2
        assert np.isfinite(history).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.best_epoch in (0, 1) and np.isfinite(model.best_test_loss)
    predictions, targets = model.predict(test)
    assert predictions.shape == targets.shape == data[2:].shape
    assert torch.isfinite(predictions).all()
    assert not predictions.requires_grad


def test_latent_neural_ode_parameterized_training():
    """Exercise LNODE's parameter transfer and auxiliary loss path explicitly."""
    torch.manual_seed(22)
    data, times = _data()
    model = LatentNeuralODE(
        device="cpu",
        n_quantities=2,
        n_timesteps=8,
        n_parameters=1,
        config={
            "latent_features": 2,
            "coder_layers": 2,
            "coder_width": 8,
            "ode_layers": 2,
            "ode_width": 8,
            "gradient_clip": 1.0,
        },
    )
    model.update_epochs = 1
    model.checkpointing = True
    train, test, _ = model.prepare_data(
        data[:2],
        data[2:],
        None,
        times,
        batch_size=2,
        shuffle=False,
        dataset_train_params=np.array([[0.2], [0.4]], dtype=np.float32),
        dataset_test_params=np.array([[0.6], [0.8]], dtype=np.float32),
    )
    initial = [p.detach().clone() for p in model.parameters()]
    model.fit(train, test, 2)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, model.parameters())
    )
    assert np.isfinite(model.fit_duration) and model.fit_duration >= 0
    assert np.isfinite(model.MAE).all()
    assert np.isfinite(model.train_loss).all() and np.isfinite(model.test_loss).all()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.best_epoch in (0, 1) and np.isfinite(model.best_test_loss)


@pytest.mark.parametrize(
    "model_class, config",
    [
        (FullyConnected, {"hidden_size": 8, "num_hidden_layers": 2}),
        (
            MultiONet,
            {
                "hidden_size": 8,
                "branch_hidden_layers": 2,
                "trunk_hidden_layers": 2,
                "output_factor": 2,
            },
        ),
    ],
)
def test_fit_preserves_custom_epoch_override(monkeypatch, model_class, config):
    torch.manual_seed(9)
    data, times = _data()
    model = model_class(device="cpu", n_quantities=2, n_timesteps=8, config=config)
    train, test, _ = model.prepare_data(
        data, data[:2], None, times, batch_size=2, shuffle=False
    )
    original_epoch = model.epoch
    calls = []

    def custom_epoch(loader, criterion, optimizer):
        calls.append(loader)
        return original_epoch(loader, criterion, optimizer)

    monkeypatch.setattr(model, "epoch", custom_epoch)
    model.fit(train, test, 2)
    assert calls == [train, train]


@pytest.mark.parametrize("epochs", [0, -1, True, 1.0])
def test_invalid_epochs_rejected_before_optimizer_setup(monkeypatch, epochs):
    model = FullyConnected(
        device="cpu",
        n_quantities=2,
        n_timesteps=8,
        config={"hidden_size": 8, "num_hidden_layers": 2},
    )

    def unexpected_setup(*args):
        pytest.fail("optimizer should not be initialized for invalid epochs")

    monkeypatch.setattr(model, "setup_optimizer_and_scheduler", unexpected_setup)
    data, times = _data()
    train, test, _ = model.prepare_data(
        data, data[:2], None, times, batch_size=2, shuffle=False
    )
    with pytest.raises(ValueError, match="epochs"):
        model.fit(train, test, epochs)


@pytest.mark.parametrize("failure", ["prune", "loss"])
def test_fit_closes_progress_on_training_failure(monkeypatch, failure):
    class FailingLoss(nn.Module):
        def forward(self, predictions, targets):
            raise RuntimeError("loss failed")

    class PruningTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True

    data, times = _data()
    config = {"hidden_size": 8, "num_hidden_layers": 2}
    if failure == "loss":
        config["loss_function"] = FailingLoss()
    model = FullyConnected(device="cpu", n_quantities=2, n_timesteps=8, config=config)
    if failure == "prune":
        model.optuna_trial = PruningTrial()
    train, test, _ = model.prepare_data(
        data, data[:2], None, times, batch_size=2, shuffle=False
    )
    setup = model.setup_progress_bar
    closed = []

    def setup_progress(*args):
        progress = setup(*args)
        original_close = progress.close

        def close():
            closed.append(True)
            original_close()

        progress.close = close
        return progress

    monkeypatch.setattr(model, "setup_progress_bar", setup_progress)
    with pytest.raises(optuna.TrialPruned if failure == "prune" else RuntimeError):
        model.fit(train, test, 2)
    assert closed


@pytest.mark.parametrize(
    "overrides", [{"loss_fn": lambda batch, criterion: None}, {"gradient_clip": 0.0}]
)
def test_incompatible_epoch_options_fail_before_model_setup(overrides):
    from codes.surrogates.training import fit_standard

    with pytest.raises(ValueError, match="epoch_fn"):
        fit_standard(None, None, None, 1, epoch_fn=lambda *args: None, **overrides)
