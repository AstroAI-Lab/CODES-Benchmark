from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np
import torch

from codes.utils import make_shuffle_generator


class ChemDataset(torch.utils.data.Dataset):
    """
    Dataset class for the latent neural ODE model.
    Returns each sample along with its timesteps and (optionally) fixed parameters.

    Deprecated: use FlatSeqBatchIterable for pre-batched sequence data.
    """

    def __init__(
        self,
        raw_data: np.ndarray | torch.Tensor,
        timesteps: np.ndarray | torch.Tensor,
        device: str | torch.device,
        parameters: np.ndarray | torch.Tensor | None,
    ) -> None:
        warnings.warn(
            "ChemDataset is deprecated and will be removed in a future release; "
            "use FlatSeqBatchIterable for pre-batched sequence data.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(raw_data, torch.Tensor):
            raw_data = torch.tensor(raw_data, dtype=torch.float32)
        self.data = raw_data.to(device)
        self.length = self.data.shape[0]
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float32)
        self.timesteps = timesteps.to(device)
        if parameters is not None:
            if not isinstance(parameters, torch.Tensor):
                parameters = torch.tensor(parameters, dtype=torch.float32)
            self.parameters = parameters.to(device)
        else:
            self.parameters = None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.parameters is not None:
            return self.data[index, :, :], self.timesteps, self.parameters[index]
        else:
            return self.data[index, :, :], self.timesteps

    def __len__(self) -> int:
        return self.length


class FlatSeqBatchIterable(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_t: torch.Tensor,
        timesteps_t: torch.Tensor,
        params_t: torch.Tensor | None,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.data = data_t
        self.t = timesteps_t
        self.params = params_t
        self.N = data_t.size(0)
        self.bs = batch_size
        self.shuffle = shuffle
        self.generator = make_shuffle_generator() if shuffle else None

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        perm = (
            torch.randperm(self.N, generator=self.generator)
            if self.shuffle
            else torch.arange(self.N)
        )
        for start in range(0, self.N, self.bs):
            idx = perm[start : start + self.bs]
            yield (
                self.data[idx],
                self.t,
                None if self.params is None else self.params[idx],
            )

    def __len__(self) -> int:
        return (self.N + self.bs - 1) // self.bs
