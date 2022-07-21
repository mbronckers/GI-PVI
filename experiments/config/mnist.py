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

@dataclass
class MNISTConfig(Config):
    posterior_type: str = "pvi_mnist"
    location = os.path.basename(__file__)
    data: DGP = DGP.mnist

    # MFVI settings
    deterministic: bool = False  # deterministic client training
    random_mean_init: bool = False   # True => Initialize weight layer mean from N(0,1)

    # Model architecture
    N: int = None
    M: int = 50
    S: int = 1
    I: int = 5
    dims = [(28 * 28), 50, 50, 10]
    batch_size: int = 1024

    # Communication settings
    global_iters: int = 10  # shared/global server iterations
    local_iters: int = 2000  # client-local iterations

    # PVI architecture - server & clients
    server_type: Server = SequentialServer
    num_clients: int = 1
    global_iters: int = 1  # server iterations
    local_iters: int = 2000  # client-local iterations
 
    # Learning rates
    separate_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.05  # CIFAR from Ober uses log_prec_lr 3 factor
    lr_client_z: float = 0.01
    lr_yz: float = 0.01

    def __post_init__(self):
        self.name = set_experiment_name(self)
        
        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [float(1 / self.num_clients) for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}

        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [B.exp(-4) / 3 for _ in range(len(self.dims) - 1)]
        self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1
