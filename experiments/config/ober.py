from __future__ import annotations

import sys
import os
from dataclasses import dataclass

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

from priors import Prior
from dgp import DGP
import lab as B
import torch
from gi.server import SequentialServer, Server, SynchronousServer
from kl import KL

from .config import Config, set_experiment_name


# The default config settings below follow Ober et al.'s toy regression experiment details

@dataclass
class OberConfig(Config):
    location = os.path.basename(__file__)
    posterior_type: str = "pvi"
    dgp: DGP = DGP.ober_regression

    
    # Learning rates
    sep_lr: bool = False  # True => use seperate learning rates
    lr_global = 0.05
    lr_nz: float = 0.05
    lr_client_z: float = 0.05
    lr_yz: float = 0.05

    # Communication settings
    global_iters: int = 1  # server iterations
    local_iters: int = 2000  # client-local iterations

    deterministic: bool = False  # deterministic client training
    random_z: bool = False  # random inducing point initialization
    linspace_yz: bool = False  # True => use linspace(-1, 1) for yz initialization
    
    # Ober fixes ll variance to 3/scale(x_tr)
    fix_ll: bool = True  # true => fix ll variance
    # ll_var: float = 0.10  # likelihood variance

    N: int = 40  # Num total training data pts, not the number of data pts per client.
    M: int = 40
    batch_size: int = 40
    
    # Model architecture
    S: int = 10
    I: int = 100
    dims = [1, 50, 50, 1]
    
    # Server & clients
    server_type: Server = SequentialServer
    num_clients: int = 1

    def __post_init__(self):
        self.name = set_experiment_name(self)

        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [B.exp(-4) for _ in range(len(self.dims) - 1)]
        # self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1

        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [float(1 / self.num_clients) for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}


@dataclass
class MFVI_OberConfig(OberConfig):
    """ We keep hypers equal as much as possible for MFVI.
    We only change the learning rate and weight precisions.
    """
    posterior_type: str = "mfvi"

    sep_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.10
    lr_yz: float = 0.10

    global_iters: int = 1
    local_iters: int = 2000

    # Model architecture
    S: int = 10
    I: int = 100
    dims = [1, 50, 50, 1]

    # Initialize weight layer mean from N(0,1)
    random_mean_init: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Weight variances per Ober et al.
        self.nz_inits: list[float] = [1e3 - (self.dims[i] + 1) for i in range(len(self.dims) - 1)]

