import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co

logger = logging.getLogger(__name__)

def generate_regression_data(train: bool, size: int, batch_size: int, shuffle: bool, seed: int = None, type: int = 1) -> DataLoader:
    """Returns dataloader of a toy regression dataset.
    """
    if train:
        return DataLoader(
            RegressionDataset(size=size, l_lim=0.0, u_lim=0.5, type=type),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )
    else:
        return DataLoader(
            RegressionDataset(size=size, l_lim=-0.2, u_lim=1.4, type=type),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )


class RegressionDataset(Dataset):
    def __init__(
        self,
        size: int,
        l_lim: float = 0,
        u_lim: float = 1,
        train: bool = False,
        random: bool = False,
        seed: Union[int, None] = 0,
        type: Union[int, None] = 1

    ) -> None:
        """Creating the regression dataset.
        Depending on type, values ashere to the following equations:

        Type 1: y = x + 0.3sin(2π(x+ɛ)) + 0.3sin(4π(x+ɛ)) + ɛ; ɛ~𝒩(0,0.02).
        Type 2: y = 2sin(5x) + 3|x|*ɛ; ɛ ~ U[0,1].
        Type 3: y = x^3 + ɛ; ɛ ~𝒩(0,9).  // Ober's paper regression

        :param size: size of vector to generate
        :type size: int
        :param l_lim: lower x-range limit to generate data over
        :type l_lim: float
        :param u_lim: upper x-range limit to generate data over
        :type u_lim: float
        :param seed: random seed to be used, defaults to 0
        :type seed: Union[int, None], optional
        """
        super().__init__()

        self.size = size
        self.seed = seed

        if self.seed is not None: torch.manual_seed(self.seed)

        if random:
            # random on [l_lim, u_lim]
            self.x = torch.unsqueeze(torch.rand(self.size)*(u_lim - l_lim) + l_lim, dim=1)
        else:
            # linear
            self.x = torch.unsqueeze(torch.linspace(l_lim, u_lim, self.size, requires_grad=False), dim=1) 
            

        assert type in [1, 2, 3]
        if type == 1:
            epsilon = torch.randn(self.x.size()) * 0.02
            self.y = self.x + 0.3*torch.sin(2*np.pi*(self.x + epsilon)) + 0.3*torch.sin(4*np.pi*(self.x + epsilon)) + epsilon
        elif type == 2: 
            self.y = 2* torch.sin(5*self.x) + 3*torch.abs(self.x) * torch.rand(self.x.size())
        elif type == 3:

            self.x = torch.zeros(size, 1)
            self.x[:int(size/2), :] = torch.rand(int(size/2), 1)*2. - 4.
            self.x[int(size/2):, :] = torch.rand(int(size/2), 1)*2. + 2.

            self.y = self.x**3. + 3*torch.randn(size, 1)

            # Rescale the outputs to have unit variance
            scale = self.y.std()
            self.y /= scale

        else:
            raise NotImplementedError


    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def all(self):
        """ Be careful of loading the entire dataset depending on its size """
        return self[:]