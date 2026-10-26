import torch

from codes.utils import make_shuffle_generator


class ChemDataset(torch.utils.data.Dataset):
    """
    Dataset class for the latent neural ODE model.
    Returns each sample along with its timesteps and (optionally) fixed parameters.
    """

    def __init__(self, raw_data, timesteps, device, parameters):
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

    def __getitem__(self, index):
        if self.parameters is not None:
            return self.data[index, :, :], self.timesteps, self.parameters[index]
        else:
            return self.data[index, :, :], self.timesteps

    def __len__(self):
        return self.length


class FlatSeqBatchIterable(torch.utils.data.IterableDataset):
    def __init__(self, data_t, timesteps_t, params_t, batch_size, shuffle: bool):
        self.data = data_t
        self.t = timesteps_t
        self.params = params_t
        self.N = data_t.size(0)
        self.bs = batch_size
        self.shuffle = shuffle
        self.generator = make_shuffle_generator() if shuffle else None

    def __iter__(self):
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

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs
