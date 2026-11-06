from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import IterableDataset

from codes.utils import make_shuffle_generator


class FlatBatchIterable(IterableDataset):
    def __init__(
        self,
        branch_t: torch.Tensor,
        trunk_t: torch.Tensor,
        target_t: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ):
        self.branch = branch_t
        self.trunk = trunk_t
        self.target = target_t
        self.N = branch_t.size(0)
        self.bs = batch_size
        self.shuffle = shuffle
        self.generator = make_shuffle_generator() if shuffle else None

    def __iter__(self):
        # new permutation (or sequential) each epoch/iterator
        if self.shuffle:
            perm = torch.randperm(self.N, generator=self.generator)
        else:
            perm = torch.arange(self.N)

        for start in range(0, self.N, self.bs):
            idxs = perm[start : start + self.bs]
            yield (self.branch[idxs], self.trunk[idxs], self.target[idxs])

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


def mass_conservation_loss(
    masses: list,
    criterion=nn.MSELoss(reduction="sum"),
    weights: tuple = (1, 1),
    device: torch.device = torch.device("cpu"),
):
    """
    Replaces the standard MSE loss with a sum of the standard MSE loss and a mass conservation loss.

    :param masses: A list of masses for the quantities.
    :param criterion: The loss function to use for the standard loss.
    :param weights: A 2-tuple of weights for the standard loss and the mass conservation loss.
    :param device: The device to use for the loss function.

    :return: A new loss function that includes the mass conservation loss.
    """
    masses = torch.tensor(masses, dtype=torch.float32, device=device)

    def loss(outputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Loss function that includes the mass conservation loss.

        :param outputs: The predicted values.
        :param targets: The ground truth values.
        """
        standard_loss = criterion(outputs, targets)

        # Calculate the weighted sum of each quantity for predicted and ground truth,
        # resulting in the total predicted mass and ground truth mass for each sample in the batch
        predicted_mass = torch.sum(outputs * masses, dim=1)
        true_mass = torch.sum(targets * masses, dim=1)

        # Calculate the mass conservation loss as the MSE of the predicted mass vs. true mass
        mass_loss = torch.abs(predicted_mass - true_mass).sum()
        # Sum up the standard MSE loss and the mass conservation loss
        total_loss = weights[0] * standard_loss + weights[1] * mass_loss

        # print(f"Standard loss: {standard_loss.item()}, Mass loss: {mass_loss.item()}")

        return total_loss

    return loss
