"""Shared training loop for the standard surrogate models."""

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

Batch: TypeAlias = tuple[Tensor | None, ...] | list[Tensor | None]
Criterion: TypeAlias = Callable[..., Tensor]
BatchLoss: TypeAlias = Callable[[Batch, Criterion], Tensor]
EpochCallback: TypeAlias = Callable[
    [DataLoader, Criterion, torch.optim.Optimizer], None
]


def train_epoch(
    model: Any,
    data_loader: DataLoader,
    criterion: Criterion,
    optimizer: torch.optim.Optimizer,
    loss_fn: BatchLoss | None = None,
    *,
    gradient_clip: float | None = None,
) -> None:
    """Train for one epoch, using a custom batch loss when supplied.

    ``loss_fn(batch, criterion)`` must return a differentiable scalar. By default,
    the model returns predictions and targets and the criterion is applied to them.
    A gradient_clip override of zero disables clipping; None uses the config.
    """
    clip = model.config.gradient_clip if gradient_clip is None else gradient_clip
    for batch in data_loader:
        optimizer.zero_grad()
        if loss_fn is None:
            predictions, targets = model(batch)
            loss = criterion(predictions, targets)
        else:
            loss = loss_fn(batch, criterion)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def fit_standard(
    model: Any,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    position: int = 0,
    description: str = "Training",
    multi_objective: bool = False,
    loss_fn: BatchLoss | None = None,
    *,
    gradient_clip: float | None = None,
    epoch_fn: EpochCallback | None = None,
) -> None:
    """Train and restore the best checkpoint for a standard surrogate.

    The model supplies config.loss, config.gradient_clip, update_epochs, and the
    standard optimizer, progress, checkpoint and validation methods. The optimizer
    returned by setup_optimizer_and_scheduler must provide train() and eval().
    ``loss_fn`` may customize the batch loss as described by :func:`train_epoch`;
    ``gradient_clip`` overrides the configured limit. ``epoch_fn``, if provided,
    receives (train_loader, criterion, optimizer) and owns the whole epoch. It
    cannot be combined with a batch loss or gradient clipping override.

    Scheduler steps and validation checks occur once per epoch; metrics are
    computed at the model's update_epochs interval. The model's loss histories
    and n_epochs are populated, and its checkpoint is restored on success.
    Concrete fit methods can use time_execution to record fit_duration.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if epoch_fn is not None and (loss_fn is not None or gradient_clip is not None):
        raise ValueError("epoch_fn cannot be combined with loss_fn or gradient_clip")

    criterion = model.config.loss
    optimizer, scheduler = model.setup_optimizer_and_scheduler(epochs)
    loss_length = (epochs + model.update_epochs - 1) // model.update_epochs
    model.train_loss, model.test_loss, model.MAE = [
        np.zeros(loss_length) for _ in range(3)
    ]
    progress_bar = model.setup_progress_bar(epochs, position, description)
    try:
        model.train()
        optimizer.train()
        model.setup_checkpoint()
        for epoch in progress_bar:
            if epoch_fn is None:
                train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    loss_fn,
                    gradient_clip=gradient_clip,
                )
            else:
                epoch_fn(train_loader, criterion, optimizer)
            scheduler.step()
            model.validate(
                epoch=epoch,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                progress_bar=progress_bar,
                total_epochs=epochs,
                multi_objective=multi_objective,
            )
            epoch_callback = getattr(model, "epoch_callback", None)
            if epoch_callback is not None:
                epoch_callback(epoch + 1, optimizer)
        model.n_epochs = epoch + 1
        # Schedule-free optimizers expose evaluation weights in eval mode.
        optimizer.eval()
        model.get_checkpoint(test_loader, criterion)
    finally:
        progress_bar.close()
