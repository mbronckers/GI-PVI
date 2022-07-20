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
    seed: int = 0
    plot: bool = True

    deterministic: bool = False  # deterministic client training
    linspace_yz: bool = False  # True => use linspace(-1, 1) for yz initialization

    S: int = 10  # Number of training weight samples
    I: int = 100  # Number of inference samples

    # ll_var: float = 1e-2  # likelihood variance
    fix_ll: bool = True  # fix ll variance
    random_z: bool = False  # random inducing point initialization

    # Learning rates
    sep_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.05
    lr_client_z: float = 0.05
    lr_yz: float = 0.05

    prior: Prior = Prior.NealPrior
    kl: KL = KL.Analytical
    optimizer: str = "Adam"

    bias: bool = True

    load: str = None
    log_step: int = 50

    start = None
    start_time = None
    results_dir = None
    wd = None
    plot_dir = None
    metrics_dir = None
    model_dir = None
    training_plot_dir = None

    def __post_init__(self):
        # Directory name
        if self.server_type == SequentialServer:
            self.name = "seq"
        elif self.server_type == SynchronousServer:
            self.name = "sync"


################################################################

# The default config settings above follows Ober et al.'s toy regression experiment details


@dataclass
class PVIConfig(Config):
    posterior_type: str = "pvi"
    dgp: DGP = DGP.ober_regression

    # Communication settings
    global_iters: int = 1  # server iterations
    local_iters: int = 2000  # client-local iterations

    # deterministic: bool = True  # deterministic client training
    # linspace_yz: bool = True  # True => use linspace(-1, 1) for yz initialization

    N: int = 40  # Num total training data pts, not the number of data pts per client.
    M: int = 40
    batch_size: int = 40
    S: int = 10
    dims = [1, 50, 50, 1]

    # Server & clients
    server_type: Server = SequentialServer
    num_clients: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.name += f"_{self.posterior_type}"
        self.name += f"_{self.num_clients}c_{self.global_iters}g_{self.local_iters}l_{self.N}N_{self.M}M"

        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [B.exp(-4) for _ in range(len(self.dims) - 1)]
        # self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1

        # Homogeneous, equal-sized split.
        self.client_splits: list[float] = [float(1 / self.num_clients) for _ in range(self.num_clients)]
        self.optimizer_params: dict = {"lr": self.lr_global}


@dataclass
class MFVIConfig(PVIConfig):
    posterior_type: str = "mfvi"

    sep_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.10
    lr_nz: float = 0.10
    lr_yz: float = 0.10

    global_iters: int = 1
    local_iters: int = 2000

    dims = [1, 50, 50, 1]

    # Initialize weight layer mean from N(0,1)
    random_mean_init: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Weight variances per Ober et al.
        self.nz_inits: list[float] = [1e3 - (self.dims[i] + 1) for i in range(len(self.dims) - 1)]


@dataclass
class ProteinConfig(PVIConfig):
    posterior_type: str = "pvi_protein"
    dgp: DGP = DGP.uci_protein

    global_iters: int = 1  # server iterations
    local_iters: int = 1000  # client-local iterations

    # UCI Protein config
    N: int = 0.8  # Fraction training pts vs test
    M: int = 10  # Number of inducing points per client
    dims = [9, 50, 50, 1]
    batch_size: int = 10000
    fix_ll: bool = False  # true => fix ll variance
    ll_var: float = 0.10  # likelihood variance

    def __post_init__(self):
        super().__post_init__()
        self.name = f"_{self.posterior_type}"
        self.name += f"_{self.num_clients}c_{self.global_iters}g_{self.local_iters}l_{self.N}N_{self.M}M"


@dataclass
class MFVI_ProteinConfig(MFVIConfig):
    posterior_type: str = "pvi_protein"
    dgp: DGP = DGP.uci_protein

    fix_ll: bool = False  # true => fix ll variance
    global_iters: int = 1  # server iterations
    local_iters: int = 1000  # client-local iterations

    # UCI Protein config
    N: int = 0.8  # Fraction training pts (total)
    M: int = 100  # Number of inducing points per client
    dims = [9, 50, 50, 1]
    batch_size: int = 10000
    fix_ll: bool = False
    ll_var: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        self.name = f"_{self.posterior_type}"
        self.name += f"_{self.num_clients}c_{self.global_iters}g_{self.local_iters}l_{self.N}N_{self.M}M"


@dataclass
class ClassificationConfig(PVIConfig):
    posterior_type: str = "pvi_class"

    data: DGP = DGP.mnist
    dims = [(28 * 28), 100, 100, 10]

    # Communication settings
    global_iters: int = 10  # shared/global server iterations
    local_iters: int = 2000  # client-local iterations

    # Note: number of test points is also equal to N
    N: int = 60000
    M: int = 100  # Number of inducing points per client
    S: int = 1
    I: int = 5
    batch_size: int = 1024

    num_clients: int = 1
    server_type: Server = SynchronousServer

    prior: Prior = Prior.NealPrior
    kl: KL = KL.Analytical
    log_step: int = 1

    # Learning rates
    separate_lr: bool = False  # True => use seperate learning rates
    lr_global: float = 0.05
    lr_nz: float = 0.05  # CIFAR from Ober uses log_prec_lr 3 factor
    lr_client_z: float = 0.01
    lr_yz: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        # Precisions of the inducing points per layer
        self.nz_inits: list[float] = [B.exp(-4) / 3 for _ in range(len(self.dims) - 1)]
        self.nz_inits[-1] = 1.0  # According to paper, last layer precision gets initialized to 1
