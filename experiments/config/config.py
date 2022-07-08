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


@dataclass
class Config:
    name: str = "global-vi"
    seed: int = 0
    plot: bool = True
    deterministic: bool = False

    epochs: int = 1000

    N: int = 40  # Number of training data pts
    M: int = 40  # Number of inducing points
    S: int = 10  # Number of training weight samples
    I: int = 100  # Number of inference samples

    batch_size: int = 40

    nz_init: float = B.exp(-4)  # precision
    ll_var: float = 1e-2  # likelihood variance
    fix_ll: bool = True  # fix ll variance

    # Learning rates
    separate_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.001
    lr_output_var: float = 1e-3
    lr_client_z: float = 0.01
    lr_yz: float = 0.01

    prior: Prior = Prior.StandardPrior
    dgp: DGP = DGP.ober_regression
    optimizer: str = "Adam"

    random_z: bool = False
    bias: bool = True

    dims = [1, 50, 50, 1]

    load: str = None

    log_step: int = 20

    start = None
    start_time = None
    results_dir = None
    wd = None
    plot_dir = None
    metrics_dir = None
    model_dir = None
    training_plot_dir = None

    # Clients
    num_clients: int = 1

    def __post_init__(self):
        self.client_splits: list[float] = [1.0]
        self.optimizer_params: dict = {"lr": self.lr_global}


################################################################

# The default config settings follow Ober et al.'s toy regression experiment details


@dataclass
class PVIConfig(Config):
    log_step: int = 50

    deterministic: bool = False  # deterministic client training data and likelihood variance (3/scale)
    linspace_yz: bool = False  # True => use linspace(-1, 1) for yz initialization

    # Communication settings
    iters: int = 10  # server iterations
    epochs: int = 2000  # client-local epochs

    # Note: number of test points is also equal to N
    N: int = 40  # Num total training data pts, not the number of data pts per client.
    M: int = 20  # Number of inducing points per client
    # batch_size: int = 30

    num_clients: int = 2
    server_type: Server = SynchronousServer

    prior: Prior = Prior.NealPrior
    kl: KL = KL.Analytical

    def __post_init__(self):
        # Directory name
        if self.server_type == SequentialServer:
            self.name = "seq_pvi"
        elif self.server_type == SynchronousServer:
            self.name = "sync_pvi"
        else:
            self.name = "pvi"
        self.name += f"_{self.num_clients}c_{self.iters}i_{self.epochs}e_{self.N}N_{self.M}M_{str(self.kl)}_KL"

        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [B.exp(-4) for _ in range(len(self.dims) - 1)]
        # self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1

        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [1 / self.num_clients for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}


@dataclass
class ClassificationConfig(PVIConfig):
    name: str = "classification"


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
