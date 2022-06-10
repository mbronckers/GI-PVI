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
import slugify
import torch
import torch.nn as nn
from gi.utils.plotting import line_plot, plot_confidence, scatter_plot
from varz import Vars, namespace
from wbml import experiment, out

from priors import Prior, build_prior, parse_prior_arg

from dgp import generate_data, generate_data2, split_data

def build_z(key: B.RandomState, M: B.Int, x, y, random: bool=False):
    """
    Build M inducing points from data (x, y).
    - If M < len(x), select a random M-sized subset from x
    - If M > len(x), init len(x) points to x, then randomly sample from N(0,1)

    :param zs: inducing inputs
    :param yz: pseudo (inducing) outputs for final layer
    :param random: if true, we entirely init z to random samples from N(0,1)

    :returns: key, z, y
    """
    if random:
        key, z = B.randn(key, B.default_dtype, M, *x.shape[1:]) # [M x input_dim]
        key, yz = B.randn(key, B.default_dtype, M, *y.shape[1:]) # [M x output_dim]
        return key, z, yz

    if M <= len(x):
        # Select random subset of size M of training points x
        key, perm = B.randperm(key, B.default_dtype, len(x))
        idx = perm[:M]
        z, yz = x[idx], y[idx]
    else:
        z, yz = x, y
        key, z_ = B.randn(key, B.default_dtype, M - len(x), *x.shape[1:]) # Generate z_, yz_ 
        key, yz_ = B.randn(key, B.default_dtype, M - len(x), *y.shape[1:])
        z = B.concat(z, z_)
        yz = B.concat(yz, yz_)
        
    return key, z, yz

def build_ts(key, M, yz, *dims: B.Int, nz_init=1e-3):
    """
    Builds likelihood factors per layer for one client

    For the final layer, the pseudo observations are init to the passed <yz> (usually, the training data output y)
    For non-final layers, the pseudo obersvations <_yz> ~ N(0, 1)

    :return ts: Dictionary of likelihood factors for each layer.
    :rtype: dict<k=layer_name, v=NormalPseudoObservation>
    """
    ts = {}
    num_layers = len(dims) - 1
    for i in range(num_layers):   
        _nz = B.ones(dims[i + 1], M) * nz_init                       # [Dout x M]

        if i < num_layers - 1: 
            key, _yz = B.randn(key, B.default_dtype, M, dims[i + 1]) # [M x Dout]
            t = gi.NormalPseudoObservation(_yz, _nz)
        else: 
            t = gi.NormalPseudoObservation(yz, _nz) # final layer
            
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
    error = y-out.mean(0)               # error of mean prediction
    mse = B.mean(error**2)

    # Mini-batching estimator of ELBO (N / batch_size)
    elbo = ((N / len(x)) * exp_ll) - kl
    
    return key, elbo, exp_ll, kl, mse

@namespace("zs")
def add_zs(vs, zs):
    """ Add client inducing points to optimizable params in vs """
    for client_name, client_z in zs.items():
        vs.unbounded(client_z, name=f"{client_name}_z")

@namespace("ts")
def add_ts(vs, ts):
    """ Add client likelihood factors to optimizable params in vs """
    for client_name, client_dict in ts.items():
        for layer_name, t in client_dict.items():
            vs.unbounded(t.yz, name=f"{client_name}_{layer_name}_yz")
            vs.positive(t.nz, name=f"{client_name}_{layer_name}_nz")

def rebuild(vs, likelihood, clients):
    """
    For positive (constrained) variables in vs, 
        we need to re-initialize the values of the objects 
        to the latest vars in vs for gradient purposes
    
    :param likelihood: update the output variance
    :param clients: update the pseudo precision
    """

    _idx = vs.name_to_index["output_var"] 
    likelihood.var = vs.transforms[_idx](vs.get_vars()[_idx])
    for client_name, client in clients.items():
        client.update_nz(vs)

def load_vs(fpath):
    """ Load saved vs state dict into new Vars container object"""
    assert os.path.exists(fpath)
    _vs_state_dict = torch.load(fpath)

    vs: Vars = vars(B.default_dtype)
    for idx, name in enumerate(_vs_state_dict.keys()):
        if name.__contains__("output_var") or \
            name.__contains__("nz"):
            vs.positive(_vs_state_dict[name], name=name)
        else:
            vs.unbounded(_vs_state_dict[name], name=name)

    return vs

def eval_logging(x, y, y_pred, mse, pred_var, data_name, _results_dir, _fname, plot: bool):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        y_pred (_type_): _description_
        mse (_type_): _description_
        pred_var (_type_): _description_
        data_name (str): type of (x,y) dataset, e.g. "test", "train", "eval", "all"
        _results_dir (_type_): results directory to save plots
        _fname (_type_): plot file name
        plot (bool): save plot figure
    """
    _S = y_pred.shape[0] # number of inference samples

    # Log test error and variance
    logger.info(f"{data_name} error (MSE): {round(mse.item(), 1):3}, {data_name} var: {round(y_pred.var().item(), 1):3}")

    # Save model predictions
    _results_eval = pd.DataFrame({
        'x_eval': x.squeeze().detach().cpu(),
        'y_eval': y.squeeze().detach().cpu(),
        'pred_errors': (y - y_pred.mean(0)).squeeze().detach().cpu(),
        'pred_var': pred_var.squeeze().detach().cpu(),
        'y_pred_mean': y_pred.mean(0).squeeze().detach().cpu()
    })
    
    for num_sample in range(_S): 
        _results_eval[f'preds_{num_sample}'] = y_pred[num_sample].squeeze().detach().cpu()
    
    _results_eval.to_csv(os.path.join(_results_dir, f"{_fname}.csv"), index=False)

    # Plot model predictions
    if plot:
        _ax = scatter_plot(x, y, x, y_pred.mean(0), f"{data_name.capitalize()} data", "Model predictions", "x", "y", f"Model predictions on {data_name} data ({_S} samples)")

        _preds_idx = [f'preds_{i}' for i in range(_S)]
        quartiles = np.quantile(_results_eval[_preds_idx], np.array((0.05,0.25,0.75,0.95)), axis=1) # [num quartiles x num preds]
        _ax = plot_confidence(_ax, x.squeeze().detach().cpu(), quartiles, all=False)
        _ax.legend()
        plt.savefig(os.path.join(_plot_dir, f'eval_{data_name}_preds.png'), pad_inches=0.2, bbox_inches='tight')


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, help='seed', nargs='?', default=0)
    parser.add_argument('--epochs', '-e', type=int, help='epochs', default=1000)
    parser.add_argument('--plot', '-p', action='store_true', help='Plot results')
    parser.add_argument('--name', '-n', type=str, help='Experiment name', default="")
    parser.add_argument('--M', '-M', type=int, help='number of inducing points', default=100)
    parser.add_argument('--N', '-N', type=int, help='number of training points', default=1000)
    parser.add_argument('--training_samples', '-S', type=int, help='number of training weight samples', default=2)
    parser.add_argument('--inference_samples', '-I', type=int, help='number of inference weight samples', default=10)
    parser.add_argument('--nz_init', type=float, help='Client likelihood factor noise initial value', default=1e-3)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-2)
    parser.add_argument('--ll_var', type=float, help='likelihood var', default=1e-3)
    parser.add_argument('--batch_size', '-b', type=int, help='training batch size', default=100)
    parser.add_argument('--dgp', '-d', type=int, help='dgp/dataset type', default=1)
    # parser.add_argument('--load', '-l', type=str, help='model directory to load (e.g. experiment_name/model)', default=None)
    parser.add_argument('--random_z', '-z', action='store_true', help='Randomly initializes global inducing points z')
    parser.add_argument('--prior', '-P', type=str, help='prior type', default="NealPrior")
    args = parser.parse_args()

    # Create experiment directories
    _time = datetime.utcnow().strftime("%Y-%m-%d-%H.%M.%S")
    _results_dir_name = "results"
    _results_dir = os.path.join(_root_dir, _results_dir_name, f"{_time}_{slugify.slugify(args.name)}")
    _wd = experiment.WorkingDirectory(_results_dir, observe=True, seed=args.seed)
    _plot_dir = os.path.join(_results_dir, "plots")
    _metrics_dir = os.path.join(_results_dir, "metrics")
    _model_dir = os.path.join(_results_dir, "model")
    _training_plot_dir = os.path.join(_plot_dir, "training")
    Path(_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_training_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_model_dir).mkdir(parents=True, exist_ok=True)
    Path(_metrics_dir).mkdir(parents=True, exist_ok=True)

    # Save script
    if os.path.exists(os.path.abspath(sys.argv[0])):
        shutil.copy(os.path.abspath(sys.argv[0]), _wd.file("script.py"))
    else:
        out("Could not save calling script.")

    #### Logging ####
    logging.config.fileConfig(os.path.join(file_dir, 'logging.conf'), defaults={'logfilepath': _results_dir})
    logger = logging.getLogger()
    np.set_printoptions(linewidth=np.inf)

    # Log program info
    logger.debug(f"Call: {sys.argv}")
    logger.debug(f"Root: {_results_dir}")
    logger.debug(f"Time: {_time}")
    logger.debug(f"Seed: {args.seed}")

    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    
    # Generate regression data
    N = args.N      # number of training points
    if args.dgp == 1:
        key, x, y = generate_data(key, size=N, xmin=-6, xmax=6)
    else:
        key, x, y, _, _ = generate_data2(key, size=N, xmin=-4, xmax=4)
    x_tr, y_tr, x_te, y_te = split_data(x, y)
    
    # Define model
    model = gi.GIBNN(nn.functional.relu)

    # Build one client
    M = args.M # number of inducing points
    in_features = 1
    dims = [in_features, 50, 50, 1]
    prior = parse_prior_arg(args.prior)
    ps = build_prior(*dims, prior=prior)
    key, z, yz = build_z(key, M, x_tr, y_tr, args.random_z)
    t = build_ts(key, M, yz, *dims, nz_init=args.nz_init)
    clients = {"client0": gi.Client("client0", (x_tr, y_tr), z, t)}
    
    # Plot initial inducing points
    if args.plot:
        _ax = scatter_plot(x1=x_tr, y1=y_tr, x2=z, y2=yz,
                    desc1="Training data", desc2="Initial inducing points",
                    xlabel="x", ylabel="y", title=f"Inducing points and training data")
        plt.savefig(os.path.join(_plot_dir, "init_zs.png"), pad_inches=0.2, bbox_inches='tight')
        

    # Collect clients
    ts = {}
    zs = {}
    for client_name, client in clients.items():
        _t = client.t
        for layer_name, layer_t in _t.items():
            if layer_name not in ts: ts[layer_name] = {}
            ts[layer_name][client_name] = layer_t
        zs[client_name] = client.z
        
    # Initialize variable containers for optimizable params with appropriate constraints
    vs = Vars(B.default_dtype)
    
    # Define likelihood.
    output_var = vs.positive(args.ll_var, name="output_var")
    likelihood = gi.likelihoods.NormalLikelihood(output_var)
    
    # Add zs, ts to optimizable containers
    add_zs(vs, zs)
    add_ts(vs, ts)
    
    # Set requirement for gradients    
    vs.requires_grad(True, *vs.names) # By default, no variable requires a gradient in Varz
    rebuild(vs, likelihood, clients)

    # Optimizer parameters
    lr = args.lr
    opt = torch.optim.Adam(vs.get_vars(), lr=lr)
    batch_size = min(args.batch_size, N)
    S = args.training_samples # number of inference samples
    epochs = args.epochs
    
    # Logging
    log_step = 100 # report every <log_step> epochs
    olds = {} # to keep track of old parameter values
    elbos = []
    errors = []
    kls = []
    lls = []

    # Plot data
    if args.plot:
        _ax = scatter_plot(x1=x_tr, y1=y_tr, x2=x_te, y2=y_te,
                    desc1="Training data", desc2="Testing data",
                    xlabel="x", ylabel="y", title=f"Regression data")
        plt.savefig(os.path.join(_plot_dir, 'regression_data.png'), pad_inches=0.2, bbox_inches='tight')
    
    # Run training
    for i in range(epochs):
        
        # Construct i-th minibatch {x, y} training data
        inds = (B.range(batch_size) + batch_size*i) % len(x_tr)
        x_mb = B.take(x_tr, inds) # take() is for JAX
        y_mb = B.take(y_tr, inds)
        
        key, elbo, exp_ll, kl, error = estimate_elbo(key, model, likelihood, x_mb, y_mb, ps, ts, zs, S, N)

        logger.debug(f"[{i:4}/{epochs:4}] - elbo: {round(elbo.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(error.item(), 1):8.1f}")
        
        elbos.append(elbo.detach().cpu().item())
        errors.append(error.item())
        kls.append(kl.detach().cpu().item())
        lls.append(exp_ll.detach().cpu().item())

        loss = -elbo
        loss.backward()

        if i%log_step == 0 or i == (epochs-1):
            logger.info(f"Epoch {i:5} - elbo: {round(elbo.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(error.item(), 1):8.1f}")

            if args.plot:
                _inducing_inputs = zs["client0"]
                _pseudo_outputs = ts[list(ts.keys())[-1]]["client0"].yz # final layer yz

                scatter_plot(x1=x_tr, y1=y_tr, x2=_inducing_inputs, y2=_pseudo_outputs, 
                    desc1="Training data", desc2="Variational params",
                    xlabel="x", ylabel="y", title=f"Epoch {i}")
                plt.savefig(os.path.join(_plot_dir, f'training/{_time}_{i}.png'), pad_inches=0.2, bbox_inches='tight')

        opt.step()
        rebuild(vs, likelihood, clients) # Rebuild clients & likelihood
        opt.zero_grad()

    if args.plot: 
        _ax = line_plot(x=[i for i in range(len(elbos))], y=elbos, desc="Training", xlabel="Epoch", ylabel="ELBO", title="ELBO convergence")
        plt.savefig(os.path.join(_plot_dir, 'elbo.png'), pad_inches=0.2, bbox_inches='tight')

    # Save metrics and parameter state
    _vs_state_dict = dict(zip(vs.names, [vs[_name] for _name in vs.names]))
    torch.save(_vs_state_dict, os.path.join(_results_dir, 'model/_vs.pt'))
    _results_training = pd.DataFrame({'lls': lls, 'kls': kls, 'elbos': elbos, 'errors': errors})
    _results_training.index.name = "epoch"
    _results_training.to_csv(os.path.join(_results_dir, "metrics/training.csv"))

    # Run eval on test dataset
    with torch.no_grad():
        
        # Resample <_S> inference weights
        key, _ = model.sample_posterior(key, ps, ts, zs, args.inference_samples)

        # Get <_S> predictions, calculate average MSE, variance
        y_pred = model.propagate(x_te)
        mse = B.mean((y_te-y_pred.mean(0))**2)
        pred_var = y_pred.var(0)

        # Log and plot results
        eval_logging(x_te, y_te, y_pred, mse, pred_var, "test", _results_dir, "model/eval_test", args.plot)
    
    # Run eval on entire dataset
    with torch.no_grad():
    
        # Resample <_S> inference weights
        key, _ = model.sample_posterior(key, ps, ts, zs, args.inference_samples)

        # Get <_S> predictions, calculate average RMSE, variance
        y_pred = model.propagate(x)

        mse = B.mean((y-y_pred.mean(0))**2)
        pred_var = y_pred.var(0)

        # Log and plot results
        eval_logging(x, y, y_pred, mse, pred_var, "all", _results_dir, "model/eval_all", args.plot)

