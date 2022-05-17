import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from config import HyperParamConfig, DataLoaderConfig

logger = logging.getLogger(__name__)

REGRESSION_CONFIG = HyperParamConfig(
    name = "Regression",

    # Dimensions
    batch_size=40,
    epochs = 1000,
    input_dim = 1,
    output_dim = 1,
    hidden_units = 100,

    # Training
    elbo_samples = 5,
    inference_samples = 10,
    regression_likelihood_noise = 0.1,

    # Optimiser
    lr = 1e-3,
    step_size = 100, 
    gamma = 0.10,
    shuffle = True
)


class RegressionDataset(Dataset):
    def __init__(
        self,
        size: int,
        l_lim: float = 0,
        u_lim: float = 1,
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

        assert type in [1, 2, 3], "Type must be one of {1, 2, 3}"

        self.size = size
        self.seed = seed

        if self.seed is not None: torch.manual_seed(self.seed)

        if random:
            # random on [l_lim, u_lim]
            self.x = torch.unsqueeze(torch.rand(self.size)*(u_lim - l_lim) + l_lim, dim=1)
        else:
            # linear
            self.x = torch.unsqueeze(torch.linspace(l_lim, u_lim, self.size, requires_grad=False), dim=1) 
            
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