from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from typing import Callable

from matplotlib import pyplot as plt

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import argparse
import logging
import logging.config
from pathlib import Path

import gi
import lab as B
import lab.torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from gi.server import SequentialServer, SynchronousServer

from gi.utils.plotting import (line_plot, plot_confidence, plot_predictions, scatter_plot)
from slugify import slugify
from varz import Vars, namespace
from wbml import experiment, out, plot

from config.config import Color, PVIConfig
from dgp import DGP, generate_data, split_data, split_data_clients
from priors import build_prior, parse_prior_arg
from utils.gif import make_gif
from utils.metrics import rmse


def estimate_local_vfe(
        key: B.RandomState, 
        model: gi.GIBNN, 
        likelihood: Callable,
        client: gi.client.Client,
        x,
        y,
        ps: dict[str, gi.NaturalNormal], 
        ts: dict[str, dict[str, gi.NormalPseudoObservation]], 
        zs: dict[str, B.Numeric], 
        S: int, 
        N: int
    ):
    # Cavity distribution is defined by all the approximate likelihoods 
    # (and corresponding inducing locations) except for those indexed by 
    # client.name.
    ts_cav = dict(ts)
    zs_cav = dict(zs)
    del ts_cav[client.name]
    del zs_cav[client.name]
    key, _cache = model.sample_posterior(key, ps, ts, zs, ts_cav, zs_cav, S)
    out = model.propagate(x) # out : [S x N x Dout]
    
    # Compute KL divergence.
    kl = 0.
    for layer_name, layer_cache in _cache.items(): # stored KL in cache already
        kl += layer_cache["kl"] 
        
    # Compute the expected log-likelihood.
    exp_ll = likelihood(out).logpdf(y)
    exp_ll = exp_ll.mean(0).sum()       # take mean across inference samples and sum
    kl = kl.mean()                      # across inference samples
    error = y - out.mean(0)               # error of mean prediction
    rmse = B.sqrt(B.mean(error**2))
    
    # Mini-batching estimator of ELBO (N / batch_size)
    elbo = ((N / len(x)) * exp_ll) - kl
    
    return key, elbo, exp_ll, kl, rmse

def main(args):
    # Set up dataset.

    # Construct clients etc.
    
    # Construct server.

    # Perform PVI.
    server = SequentialServer(clients)
    for i in range(args.iters):
        print(f"Iteration: {i}")
        # Get next client(s).
        curr_clients = next(server)
        for client in curr_clients:
            # Construct optimiser by adding clients parameters.
            opt = torch.optim.Adam(client.vs.get_vars(), lr=args.lr)
            
            for epoch in range(args.epochs):
                # Construct i-th minibatch {x, y} training data
                inds = (B.range(args.batch_size) + args.batch_size*i) % len(client.x)
                x_mb = B.take(client.x, inds) # take() is for JAX
                y_mb = B.take(client.y, inds)
                
                # Need to make sure t[client.name] and z[client.name] take the values
                # that are being optimised (i.e. those in client.vs).
                
                key, local_vfe, exp_ll, kl, error = estimate_local_vfe(key, model, likelihood, client, x_mb, y_mb, ps, ts, zs, S, N)
                local_vfe.backward()
                opt.step()
