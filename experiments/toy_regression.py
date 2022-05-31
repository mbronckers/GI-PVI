from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable

file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

import argparse
import logging
from pathlib import Path

import gi
import lab as B
import lab.torch
import numpy as np
import slugify
import torch
import torch.nn as nn
from gi.utils.plotting import plot
from varz import Vars, namespace
from wbml import experiment, out

logger = logging.getLogger(__name__)

def generate_data(key, size):
    """ Toy regression dataset from paper """
    x = B.zeros(B.default_dtype, size, 1)
    
    key, x[:int(size / 2), :] = B.rand(key, B.default_dtype, int(size / 2), 1)
    x[:int(size / 2)] = x[:int(size / 2)] * 2. - 4.
    key, x[int(size / 2):] = B.rand(key, B.default_dtype, int(size / 2), 1)
    x[int(size / 2):] = x[int(size / 2):] * 2. + 2.
    
    key, eps = B.randn(key, B.default_dtype, size, 1)
    y = x ** 3. + 3*eps

    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y = y/scale
    
    return key, x, y

def generate_test_data(key, size):
    """ Toy (test) regression dataset from paper """
    x = B.zeros(B.default_dtype, size)
    
    key, x = B.rand(key, B.default_dtype, int(size), 1)
    x = x * 4. - 2.
    
    # Paper specifies ε from N(0, 9), but this does not align with their plots. I suspect (3*ε) from U[0,1]
    key, eps = B.rand(key, B.default_dtype, int(size), 1)
    y = x ** 3. + 3*eps

    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y = y/scale
    
    return key, x, y

def generate_data2(key, size, xmin, xmax):
    
    key, eps1 = B.rand(key, B.default_dtype, int(size), 1)
    key, eps2 = B.rand(key, B.default_dtype, int(size), 1)

    eps1, eps2 = eps1.squeeze(), eps2.squeeze()
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1).squeeze()
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    return key, x, y, eps1, eps2

def build_prior(*dims: B.Int):
    """
    :param dims: BNN dimensionality [Din x *D_latents x Dout]
    """
    ps = {}
    for i in range(len(dims) - 1):
        mean = B.zeros(B.default_dtype, dims[i + 1], dims[i], 1) # Dout x Din x 1
        var = B.eye(B.default_dtype, dims[i])
        var = B.tile(var, dims[i + 1], 1, 1)
        ps[f"layer{i}"] = gi.NaturalNormal.from_normal(gi.Normal(mean, var))
        
    return ps

def build_z(key: B.RandomState, M: B.Int, x, y):
    """
    Build M inducing points from data (x, y)

    :param zs: inducing inputs
    :param yz: pseudo (inducing) outputs for final layer

    :returns: key, z, y
    """
    if M <= len(x):
        key, idx = B.randperm(key, B.default_dtype, M)
        z = x[idx]
        yz = y[idx]
    else:
        z = x
        yz = y
        key, z_ = B.randn(key, B.default_dtype, M - len(x), *x.shape[1:])
        key, yz_ = B.randn(key, B.default_dtype, M - len(x), *y.shape[1:])
        z = B.concat(z, z_)
        yz = B.concat(yz, yz_)
        
    return key, z, y
    

def build_ts(key, M, yz, *dims: B.Int, nz_init=1e-3):
    """
    Builds likelihood factors per layer for one client

    :return ts: Dictionary of likelihood factors for each layer.
    :rtype: dict<k=layer_name, v=NormalPseudoObservation>
    """
    ts = {}
    for i in range(len(dims) - 1):
        if i == len(dims) - 1: # final layer
            _nz = B.ones(dims[i + 1], M) * nz_init
            t = gi.NormalPseudoObservation(yz, _nz)
        else:
            _nz = B.ones(dims[i + 1], M) * nz_init
            key, _yz = B.randn(key, B.default_dtype, M, dims[i + 1]) # draw yz ~ N(0, 1)
            t = gi.NormalPseudoObservation(_yz, _nz)
            
        ts[f"layer{i}"] = t
        
    return ts

def estimate_elbo(key: B.RandomState, model: gi.GIBNN, likelihood: Callable,
            x: B.Numeric, y: B.Numeric, 
            ps: dict[str, gi.NaturalNormal], 
            ts: dict[str, dict[str, gi.NormalPseudoObservation]], 
            zs: dict[str, B.Numeric], S, N):
    
    key, _cache = model.sample_posterior(key, ps, ts, zs, S)
    out = model.propagate(x) # out : [S x N x Dout]
    
    # Compute KL divergence.
    kl = 0.
    for layer_name, layer_cache in _cache.items(): # stored KL in cache already
        kl += layer_cache["kl"] 
        
    # Compute the expected log-likelihood.
    exp_ll = likelihood(out).logpdf(y)
    exp_ll = exp_ll.mean(0).sum()       # take mean across inference samples and sum
    kl = kl.mean()                      # across inference samples
    
    # Mini-batching estimator of ELBO
    elbo = ((N / len(x)) * exp_ll) - kl
    
    logger.debug(f"elbo: {round(elbo.item(), 0):10.1f}, ll: {round(exp_ll.item(), 0):10.1f}, kl: {round(kl.item(), 1):10.1f}")

    return key, elbo

@namespace("zs")
def add_zs(vs, zs):
    """ Add client inducing points to optimizable params in vs """
    # TODO: for some reason, the (client-local) inducing points do not change across epochs
    for client_name, client_z in zs.items():
        vs.unbounded(client_z, name=f"{client_name}_z")

@namespace("ts")
def add_ts(vs, ts):
    """ Add client likelihood factors to optimizable params in vs """
    for client_name, client_dict in ts.items():
        for layer_name, t in client_dict.items():
            vs.unbounded(t.yz, name=f"{client_name}_{layer_name}_yz")
            vs.positive(t.nz, name=f"{client_name}_{layer_name}_nz")

def track_change(opt, vs, var_names, i, epoch, olds):
    """ Steps optimizer and reports delta for variables when iteration%epoch == 0"""
    _olds = olds

    opt.step()

    if i%epoch==0:
        # Report deltas
        for var_name in var_names:
            if _olds != {}:
                _delta = vs[var_name].detach().cpu().squeeze() - _olds[var_name]
            else:
                _delta = vs[var_name].detach().cpu().squeeze()

            logger.debug(f"Δ{var_name}: {np.array(_delta)}")

        # Update olds
        for var_name in var_names:
            _old = vs[var_name].detach().cpu().squeeze()
            _olds[var_name] = _old

    return opt, olds

def track(opt, vs, var_names, i, epoch):
    opt.step()

    for var_name in var_names:
        _v = vs[var_name].detach().cpu().squeeze()
        if i%epoch == 0: logger.debug(f"{var_name}: {np.array(_v)}")

    return opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Sets the log level to DEBUG')
    parser.add_argument('--seed', type=int, help='seed', nargs='?', default=0)
    parser.add_argument('--epochs', '-e', type=int, help='epochs', default=1000)
    parser.add_argument('--plot', '-p', action='store_true', help='Plot results')
    parser.add_argument('--name', '-n', type=str, help='Experiment name', default="")
    args = parser.parse_args()

    # Logging settings
    if args.verbose: 
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    np.set_printoptions(linewidth=np.inf)
    start_time = datetime.utcnow().strftime("%Y-%m-%d-%H.%M.%S")
    _start = datetime.utcnow().strftime("%Y-%m-%d-%H.%M")
    
    # Create directory for saving plots
    if args.plot:        
        name = f"{_start}_{slugify.slugify(args.name)}" if args.name != "" else _start
        file_dir = os.path.dirname(__file__)
        save_dir = os.path.join(file_dir, "../plots/")
        fdir = os.path.join(save_dir, name)
        Path(fdir).mkdir(parents=True, exist_ok=True)

    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    
    # Generate regression data
    key, x_tr, y_tr = generate_data(key, size=100)
    key, x_te, y_te = generate_test_data(key, size=50)
    
    # Define model (i.e. define prior).
    model = gi.GIBNN(nn.functional.relu)

    # Build one client
    M = 100 # number of inducing points
    ps = build_prior(1, 50, 50, 1)
    key, z, yz = build_z(key, M, x_tr, y_tr)
    t = build_ts(key, M, yz, 1, 50, 50, 1, nz_init=1e-3)

    # Collect clients
    ts = {k: {"client0": v} for k, v in t.items()} # flip order bc ts = dict<k=layer, v=pseudo obs>
    zs = {"client0": z}

    # Initialize variable containers for optimizable params with appropriate constraints
    vs = Vars(B.default_dtype)
    
    # Define likelihood.
    output_var = vs.positive(1e-3, name="output_var")
    likelihood = gi.likelihoods.NormalLikelihood(output_var)
    
    # Add zs, ts to optimizable containers
    add_zs(vs, zs)
    add_ts(vs, ts)
    
    # Set requirement for gradients    
    vs.requires_grad(True, *vs.names) # By default, no variable requires a gradient in Varz
    
    # Optimizer parameters
    opt = torch.optim.Adam(vs.get_vars(), lr=1e-2)
    N = len(x_tr)
    batch_size = min(100, N)
    S = 2 # number of inference samples
    mb_idx = 0 # minibatch idx
    epochs = args.epochs
    
    # Logging
    log_step = 100 # report change in param values every <log_step> epochs
    olds = {} # to keep track of old parameter values
    
    plot(fdir=fdir, fname=f"regression_data",
    x1=x_tr, y1=y_tr, x2=x_te, y2=y_te,
    desc1="Training data", desc2="Testing data",
    xlabel="x", ylabel="y", title=f"Regression data")

    for i in range(epochs):
        
        # Construct i-th minibatch {x, y} training data
        inds = B.range(batch_size) + mb_idx
        x_mb = B.take(x_tr, inds) # take() is for JAX
        y_mb = B.take(y_tr, inds)
        mb_idx = (mb_idx + batch_size) % len(x_tr)
        
        key, elbo = estimate_elbo(key, model, likelihood, x_mb, y_mb, ps, ts, zs, S, N)
        loss = -elbo
        loss.backward()

        if i%100 == 0: 
            logger.info(f"Epoch {i} - loss (-elbo): {int(loss.item())}")

            if args.plot:
                _inducing_inputs = zs["client0"]
                _pseudo_outputs = ts[list(ts.keys())[-1]]["client0"].yz # final layer yz

                plot(fdir=fdir, fname=f"{start_time}_{i}",
                    x1=_inducing_inputs, y1=_pseudo_outputs, x2=x_tr, y2=y_tr, 
                    desc1="Variational params", desc2="Training data", 
                    xlabel="x", ylabel="y", title=f"Epoch {i}")
        
        # opt, olds = track_change(opt, vs, ['ts.layer2_client0_yz', 'ts.layer0_client0_nz'], i, 100, olds)
        opt.step()
        opt.zero_grad()

