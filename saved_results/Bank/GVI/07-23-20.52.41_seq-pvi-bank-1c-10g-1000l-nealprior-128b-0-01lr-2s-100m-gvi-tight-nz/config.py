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
from gi.mfvi import MFVI_Classification
from gi.gibnn import GIBNN_Classification

from .config import Config, set_experiment_name


@dataclass
class GI_BankConfig(Config):
    posterior_type: str = "pvi_bank"
    location = os.path.basename(__file__)
    dgp: DGP = DGP.uci_bank
    model_type = GIBNN_Classification

    prior: Prior = Prior.NealPrior

    # GI settings
    deterministic: bool = False  # deterministic client training
    random_z: bool = False  # random inducing point initialization
    linspace_yz: bool = False  # True => use linspace(-1, 1) for yz initialization

    # Model architecture
    N: int = 0.8  # train_split
    M: int = 100
    S: int = 2
    I: int = 50
    dims = [51, 50, 50, 2]

    batch_size: int = 128  # None => full batch

    # PVI architecture - server & clients
    server_type: Server = SequentialServer
    num_clients: int = 1
    global_iters: int = 10  # shared/global server iterations
    local_iters: int = 1000  # client-local iterations

    # Learning rates
    sep_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.03
    lr_nz: float = 0.05  # CIFAR from Ober uses log_prec_lr 3 factor
    lr_client_z: float = 0.01
    lr_yz: float = 0.01

    def __post_init__(self):
        self.name = set_experiment_name(self)

        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [float(1 / self.num_clients) for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}

        # Precisions of the inducing points per layer
        # self.nz_inits: list[float] = [B.exp(-4) / 3 for _ in range(len(self.dims) - 1)]
        self.nz_inits: list[float] = [1 for _ in range(len(self.dims) - 1)]

        # self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1


@dataclass
class MFVI_BankConfig(Config):
    posterior_type: str = "mfvi_bank"
    location = os.path.basename(__file__)
    dgp: DGP = DGP.uci_bank
    model_type = MFVI_Classification

    prior: Prior = Prior.StandardPrior

    # MFVI settings
    deterministic: bool = False  # deterministic client training
    random_mean_init: bool = False  # True => Initialize weight layer mean from N(0,1)

    # Model architecture
    N: int = 0.8  # train_split
    S: int = 2
    I: int = 50
    dims = [51, 50, 50, 2]

    batch_size: int = 128  # None => full batch

    # PVI settings
    server_type: Server = SequentialServer
    num_clients: int = 1
    global_iters: int = 10  # shared/global server iterations
    local_iters: int = 1000  # client-local iterations

    # Learning rates
    sep_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.05
    lr_yz: float = 0.01

    def __post_init__(self):
        self.name = set_experiment_name(self)

        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [float(1 / self.num_clients) for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}

        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [1e3 - (self.dims[i] + 1) for i in range(len(self.dims) - 1)]
