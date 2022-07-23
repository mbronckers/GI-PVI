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
    """Any settings that are common to all experiments and do not change."""

    seed: int = 0
    plot: bool = True

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


################################################################


def set_experiment_name(config: Config):
    if config.server_type == SequentialServer:
        name = "seq"
    elif config.server_type == SynchronousServer:
        name = "sync"
    name += f"_{config.posterior_type}"
    name += f"_{config.num_clients}c_{config.global_iters}g_{config.local_iters}l_{config.prior.name.lower()}"

    if config.N > 1:
        name += f"_{config.N}N"
    name += f"_{config.batch_size}b" if config.batch_size else f"_full_b"
    name += f"_{config.lr_global}lr_{config.S}S"
    if "M" in config.__annotations__.keys():
        name += f"_{config.M}M"
    return name
