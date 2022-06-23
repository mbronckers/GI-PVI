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
from gi.distributions import NormalPseudoObservation

from slugify import slugify
from varz import Vars, namespace
from wbml import experiment, out, plot

from config.config import Color, PVIConfig
from dgp import DGP, generate_data, split_data, split_data_clients
from priors import build_prior, parse_prior_arg
from utils.gif import make_gif
from utils.metrics import rmse
from utils.optimization import rebuild, add_zs, add_ts, get_vs_state, load_vs

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
        S: B.Int, 
        N: B.Int
    ):
    # Cavity distribution is defined by all the approximate likelihoods (and corresponding inducing locations) except for those indexed by client.name.
    # Create cavity distributions.
    
    # If they are leaf node, there is "requires_grad=True" and is not "grad_fn=SliceBackward" or "grad_fn=CopySlices". I guess that non-leaf node has grad_fn, which is used to propagate gradients.
    
    # zs_cav = deepcopy(zs)
    zs_cav = {}
    for client_name, _client_z in zs.items():
        zs_cav[client_name] = _client_z
    del zs_cav[client.name]   
    
    # Construct cavity from scratch (to avoid linked copies)
    ts_cav = {}
    for layer_name, _t in ts.items():
        ts_cav[layer_name] = {}
        for client_name, client_t in _t.items():
            if client_name != client.name: 
                # ts_cav[layer_name][client_name] = NormalPseudoObservation(yz=client_t.yz.detach().clone(), nz=client_t.nz.detach().clone())
                ts_cav[layer_name][client_name] = NormalPseudoObservation(yz=client_t.yz, nz=client_t.nz)
        
        # del ts_cav[layer_name][client.name]

    key, _cache = model.sample_posterior(key, ps, ts, zs, ts_cav, zs_cav, S)
    out = model.propagate(x) # out : [S x N x Dout]
    
    # Compute KL divergence.
    kl = 0.
    for layer_name, layer_cache in _cache.items(): # stored KL in cache already
        kl += layer_cache["kl"] 
        
    # Compute the expected log-likelihood.
    exp_ll = likelihood(out).logpdf(y)
    exp_ll = exp_ll.mean(0).sum()         # take mean across inference samples and sum
    kl = kl.mean()                        # across inference samples
    error = y - out.mean(0)               # error of mean prediction
    rmse = B.sqrt(B.mean(error**2))
    
    # Mini-batching estimator of ELBO (N / batch_size)
    elbo = ((N / len(x)) * exp_ll) - kl
    
    return key, elbo, exp_ll, kl, rmse

def main(args, config, logger):
    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    
    # Setup regression dataset.
    N = args.N  # num training points
    key, x, y, scale = generate_data(key, args.dgp, N, xmin=-4., xmax=4.)
    x_tr, y_tr, x_te, y_te = split_data(x, y)

    # Define model
    model = gi.GIBNN(nn.functional.relu, args.bias)

    # Build prior
    M = args.M # number of inducing points
    dims = config.dims
    ps = build_prior(*dims, prior=args.prior, bias=args.bias)
    
    # Deal with client split
    if args.num_clients != len(config.client_splits): 
        # Equal split if different number of clients specified via CLI
        config.client_splits = [1/(args.num_clients) for _ in range(args.num_clients)]
    logger.info(f"{Color.WHITE}Client splits: {config.client_splits}{Color.END}")

    # We can only fix the likelihood.
    likelihood = gi.likelihoods.NormalLikelihood(args.ll_var)
    
    # Build clients
    clients = {}
    for i, (client_x_tr, client_y_tr) in enumerate(split_data_clients(x_tr, y_tr, config.client_splits)):
        
        _vs = Vars(B.default_dtype) # Initialize variable container for each client
        
        key, z, yz = gi.client.build_z(key, M, client_x_tr, client_y_tr, args.random_z)
        
        t = gi.client.build_ts(key, M, yz, *dims, nz_init=args.nz_init)
        
        clients[f"client{i}"] = gi.Client(f"client{i}", client_x_tr, client_y_tr, z, t, _vs)

    # Plot initial inducing points
    if args.plot:
        fig, _ax = plt.subplots(1, 1, figsize=(10,10))
        scatterplot = plot.patch(sns.scatterplot)
        _ax.set_title(f"Inducing points and training data - all clients")
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        for i, (name, c) in enumerate(clients.items()):
            scatterplot(c.x, c.y, label=f"Training data - {name}", ax=_ax)
            scatterplot(c.z, c.get_final_yz(), label=f"Initial inducing points - {name}", ax=_ax)

        plot.tweak(_ax)
        plt.savefig(os.path.join(_plot_dir, "init_zs.png"), pad_inches=0.2, bbox_inches='tight')
        
    # Collect clients
    ts: dict[str, dict[str, gi.NormalPseudoObservation]] = {}
    zs: dict[str, B.Numeric] = {}
    for client_name, client in clients.items():
        _t = client.t
        for layer_name, layer_t in _t.items():
            if layer_name not in ts: ts[layer_name] = {}
            ts[layer_name][client_name] = layer_t
        zs[client_name] = client.z
    
    # Set requirement for gradients    
    # vs.requires_grad(True, *vs.names) # By default, no variable requires a gradient in Varz
    # output_var = vs.positive(args.ll_var, name="output_var")
    # rebuild(vs, likelihood, clients)

    # Optimizer parameters
    batch_size = min(args.batch_size, N)
    S = args.training_samples # number of inference samples
    epochs = args.epochs

    # Construct server.
    server = SequentialServer(clients)

    # Perform PVI.
    iters = args.iters
    for i in range(iters):

        # Get next client(s).
        curr_clients = next(server)
        
        logger.info(f"SERVER - iter {i}/{iters}: optimizing clients {curr_clients}")
        
        for client in curr_clients:
            # Construct optimiser of only clients parameters.
            opt = getattr(torch.optim, config.optimizer)(client.get_params(), **config.optimizer_params)
            
            logger.info(f"Client {client.name}: starting optimization of {client.vs.names}")
            
            epochs = args.epochs
            for epoch in range(epochs):
                # Construct i-th minibatch {x, y} training data
                inds = (B.range(args.batch_size) + args.batch_size*i) % len(client.x)
                x_mb = B.take(client.x, inds) # take() is for JAX
                y_mb = B.take(client.y, inds)
                
                # Need to make sure t[client.name] and z[client.name] take the values
                # that are being optimised (i.e. those in client.vs).
                
                key, local_vfe, exp_ll, kl, error = estimate_local_vfe(key, model, likelihood, client, x_mb, y_mb, ps, ts, zs, S, N)
                local_vfe.backward()
                opt.step()
                # check whether client t / z are updated
                # client.update_nz()
                opt.zero_grad()
                
                logger.info(f"[{epoch:4}/{epochs:4}] - local vfe: {round(local_vfe.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(error.item(), 5):8.5f}")

            # Communicate back new t (not delta_t)
            zs[client.name] = client.z
            for layer_name in ts.keys():
                ts[layer_name][client.name] = client.t[layer_name]



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    config = PVIConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, help='seed', nargs='?', default=config.seed)
    parser.add_argument('--epochs', '-e', type=int, help='client epochs', default=config.epochs)
    parser.add_argument('--iters', '-i', type=int, help='server iters', default=config.epochs)
    parser.add_argument('--plot', '-p', action='store_true', help='Plot results', default=config.plot)
    parser.add_argument('--name', '-n', type=str, help='Experiment name', default=config.name)
    parser.add_argument('--M', '-M', type=int, help='number of inducing points', default=config.M)
    parser.add_argument('--N', '-N', type=int, help='number of training points', default=config.N)
    parser.add_argument('--training_samples', '-S', type=int, help='number of training weight samples', default=config.S)
    parser.add_argument('--inference_samples', '-I', type=int, help='number of inference weight samples', default=config.I)
    parser.add_argument('--nz_init', type=float, help='Client likelihood factor noise initial value', default=config.nz_init)
    parser.add_argument('--lr', type=float, help='learning rate', default=config.lr_global)
    parser.add_argument('--ll_var', type=float, help='likelihood var', default=config.ll_var)
    parser.add_argument('--batch_size', '-b', type=int, help='training batch size', default=config.batch_size)
    parser.add_argument('--dgp', '-d', type=int, help='dgp/dataset type', default=config.dgp)
    parser.add_argument('--load', '-l', type=str, help='model directory to load (e.g. experiment_name)', default=config.load)
    parser.add_argument('--random_z', '-z', action='store_true', help='Randomly initializes global inducing points z', default=config.random_z)
    parser.add_argument('--prior', '-P', type=str, help='prior type', default=config.prior)
    parser.add_argument('--bias', help='Use bias vectors in BNN', default=config.bias)
    parser.add_argument('--sep_lr', help='Use separate LRs for parameters (see config)', default=config.separate_lr)
    parser.add_argument('--num_clients', '-nc', help='Number of clients (implicit equal split)', default=config.num_clients)
    args = parser.parse_args()

    # Create experiment directories
    _start = datetime.utcnow()
    _time = _start.strftime("%Y-%m-%d-%H.%M.%S")
    _results_dir_name = "results"
    _results_dir = os.path.join(_root_dir, _results_dir_name, f"{_time}_{slugify(args.name)}")
    _wd = experiment.WorkingDirectory(_results_dir, observe=True, seed=args.seed)
    _plot_dir = os.path.join(_results_dir, "plots")
    _metrics_dir = os.path.join(_results_dir, "metrics")
    _model_dir = os.path.join(_results_dir, "model")
    _training_plot_dir = os.path.join(_plot_dir, "training")
    Path(_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_training_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_model_dir).mkdir(parents=True, exist_ok=True)
    Path(_metrics_dir).mkdir(parents=True, exist_ok=True)

    config.start = _start
    config.start_time = _time
    config.results_dir = _results_dir
    config.wd = _wd
    config.plot_dir = _plot_dir
    config.metrics_dir = _metrics_dir
    config.model_dir = _model_dir
    config.training_plot_dir = _training_plot_dir

    # Save script
    if os.path.exists(os.path.abspath(sys.argv[0])):
        shutil.copy(os.path.abspath(sys.argv[0]), _wd.file("script.py"))
        shutil.copy(os.path.join(_root_dir, "experiments/config/config.py"), _wd.file("config.py"))

    else:
        out("Could not save calling script.")    

    #### Logging ####
    logging.config.fileConfig(os.path.join(file_dir, 'config/logging.conf'), defaults={'logfilepath': _results_dir})
    logger = logging.getLogger()
    np.set_printoptions(linewidth=np.inf)

    # Log program info
    logger.debug(f"Call: {sys.argv}")
    logger.debug(f"Root: {_results_dir}")
    logger.debug(f"Time: {_time}")
    logger.debug(f"Seed: {args.seed}")
    logger.info(f"{Color.WHITE}Args: {args}{Color.END}")

    main(args, config, logger)
