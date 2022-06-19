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
from slugify import slugify
import torch
import torch.nn as nn
import seaborn as sns
from varz import Vars, namespace
from wbml import experiment, out, plot
from gi.utils.plotting import line_plot, plot_confidence, scatter_plot, plot_predictions

from priors import build_prior, parse_prior_arg
from dgp import generate_data, split_data, split_data_clients, DGP
from config.config import Config, Color
from utils.gif import make_gif
from utils.metrics import rmse
from utils.optimization import rebuild, add_zs, add_ts, get_vs_state, load_vs

def estimate_elbo(key: B.RandomState, model: gi.GIBNN, likelihood: Callable,
            x: B.Numeric, y: B.Numeric, 
            ps: dict[str, gi.NaturalNormal], 
            ts: dict[str, dict[str, gi.NormalPseudoObservation]], 
            zs: dict[str, B.Numeric], 
            S: int, N: int):
    
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
    rmse = B.sqrt(B.mean(error**2))
    
    # Mini-batching estimator of ELBO (N / batch_size)
    elbo = ((N / len(x)) * exp_ll) - kl
    
    return key, elbo, exp_ll, kl, rmse

def eval_logging(x, y, x_tr, y_tr, y_pred, error, pred_var, data_name, _results_dir, _fname, _plot: bool):
    """ Logs the model inference results and saves plots

    Args:
        x (_type_): eval input locations
        y (_type_): eval labels
        x_tr (_type_): training data
        y_tr (_type_): training labels
        y_pred (_type_): model predictions (S x Dout)
        error (_type_): (y - y_pred)
        pred_var (_type_): y_pred.var
        data_name (str): type of (x,y) dataset, e.g. "test", "train", "eval", "all"
        _results_dir (_type_): results directory to save plots
        _fname (_type_): plot file name
        _plot (bool): save plot figure
    """
    _S = y_pred.shape[0] # number of inference samples

    # Log test error and variance
    logger.info(f"{Color.WHITE} {data_name} error (RMSE): {round(error.item(), 3):3}, var: {round(y_pred.var().item(), 3):3}{Color.END}")

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
    
    _results_eval.to_csv(os.path.join(_results_dir, f"model/{_fname}.csv"), index=False)

    # Plot model predictions
    if _plot:
    
        # Plot eval data, training data, and model predictions (in that order)
        _ax = scatter_plot(None, x, y, x_tr, y_tr, f"{data_name.capitalize()} data", "Training data", "x", "y", f"Model predictions on {data_name.lower()} data ({_S} samples)")
        scatterplot = plot.patch(sns.scatterplot)
        scatterplot(ax=_ax, y=y_pred.mean(0), x=x, label="Model predictions", color=gi.utils.plotting.colors[3])

        # Plot confidence bounds
        _preds_idx = [f'preds_{i}' for i in range(_S)]
        quartiles = np.quantile(_results_eval[_preds_idx], np.array((0.05,0.25,0.75,0.95)), axis=1) # [num quartiles x num preds]
        _ax = plot_confidence(_ax, x.squeeze().detach().cpu(), quartiles, all=True)
        # _ax.legend(loc='upper right', prop={'size': 12})
        plot.tweak(_ax)
        plt.savefig(os.path.join(_plot_dir, f'{_fname}.png'), pad_inches=0.2, bbox_inches='tight')
        
        # if data_name.__contains__("domain"): _ax.set(ylim=(-4., 4))  # limit domain plot to -4, 4 to be comparable with Ober
        # plt.savefig(os.path.join(_plot_dir, f'{_fname}_ylim.png'), pad_inches=0.2, bbox_inches='tight')

        # Plot all sampled functions
        ax = plot_predictions(None, x, y_pred, "Model predictions", "x", "y", f"Model predictions on {data_name.lower()} data ({_S} samples)")
        
        _sampled_funcs_dir = os.path.join(_plot_dir, "sampled_funcs")
        Path(_sampled_funcs_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(_sampled_funcs_dir, f'{_fname}_samples.png'), pad_inches=0.2, bbox_inches='tight')
        
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, help='seed', nargs='?', default=config.seed)
    parser.add_argument('--epochs', '-e', type=int, help='epochs', default=config.epochs)
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

    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    
    # Generate regression data
    N = args.N      # number of training points
    key, x, y, scale = generate_data(key, args.dgp, N, xmin=-4., xmax=4.)
    x_tr, y_tr, x_te, y_te = split_data(x, y)

    # Define model
    model = gi.GIBNN(nn.functional.relu, args.bias)

    # Build prior
    M = args.M # number of inducing points
    dims = config.dims
    ps = build_prior(*dims, prior=args.prior, bias=args.bias)
    
    # Equal split if different number of clients specified via CLI 
    if args.num_clients != len(config.client_splits): 
        config.client_splits = [1/(args.num_clients) for _ in range(args.num_clients)]
    logger.info(f"{Color.WHITE}Client splits: {config.client_splits}{Color.END}")
    
    # Build clients
    clients = {}
    for i, (client_x_tr, client_y_tr) in enumerate(split_data_clients(x_tr, y_tr, config.client_splits)):
        data = {'x': client_x_tr, 'y': client_y_tr}
        key, z, yz = gi.client.build_z(key, M, client_x_tr, client_y_tr, args.random_z)
        t = gi.client.build_ts(key, M, yz, *dims, nz_init=args.nz_init)
        clients[f"client{i}"] = gi.Client(f"client{i}", data, z, t)
    
    # Plot initial inducing points
    if args.plot:
        fig, _ax = plt.subplots(1, 1, figsize=(10,10))
        scatterplot = plot.patch(sns.scatterplot)
        _ax.set_title(f"Inducing points and training data - all clients")
        _ax.set_xlabel("x")
        _ax.set_ylabel("y")
        for i, (name, c) in enumerate(clients.items()):
            scatterplot(c.x, c.y, label=f"Training data - {i}", ax=_ax)
            scatterplot(c.z, c.get_final_yz(), label=f"Initial inducing points - {i}", ax=_ax)

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
    if args.sep_lr:
        opt = getattr(torch.optim, config.optimizer)(
            [
                {"params": vs.get_latent_vars("*nz"), "lr": config.lr_nz},
                {"params": vs.get_latent_vars("output_var"), "lr": config.lr_output_var}, # ll var
                {"params": vs.get_latent_vars("*client*_z"), "lr": lr}, # inducing 
                {"params": vs.get_latent_vars("*yz"), "lr": lr}, # pseudo obs
            ],
            **config.optimizer_params)
    else:
        opt = getattr(torch.optim, config.optimizer)(vs.get_vars(), **config.optimizer_params)

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
        _ax = scatter_plot(None, x1=x_tr, y1=y_tr, x2=x_te, y2=y_te,
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

        logger.debug(f"[{i:4}/{epochs:4}] - elbo: {round(elbo.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(error.item(), 5):8.5f}")
        
        elbos.append(elbo.detach().cpu().item())
        errors.append(error.item())
        kls.append(kl.detach().cpu().item())
        lls.append(exp_ll.detach().cpu().item())

        loss = -elbo
        loss.backward()

        if i%log_step == 0 or i == (epochs-1):
            logger.info(f"Epoch {i:5} - elbo: {round(elbo.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(error.item(), 5):8.5f}")

            if args.plot:
                _inducing_inputs = zs["client0"]
                _pseudo_outputs = ts[list(ts.keys())[-1]]["client0"].yz # final layer yz

                scatter_plot(ax=None, x1=x_tr, y1=y_tr, 
                    x2=_inducing_inputs, y2=_pseudo_outputs, 
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

    # GIF the inducing point updates
    if args.plot: make_gif(_plot_dir)

    # Run eval on test dataset
    with torch.no_grad():
        
        # Resample <_S> inference weights
        key, _ = model.sample_posterior(key, ps, ts, zs, args.inference_samples)

        # Get <_S> predictions, calculate average RMSE, variance
        y_pred = model.propagate(x_te)

        # Log and plot results
        eval_logging(x_te, y_te, x_tr, y_tr, y_pred, rmse(y_te, y_pred), y_pred.var(0), "Test set", _results_dir, "eval_test_preds", args.plot)
    
        # Run eval on entire dataset
        y_pred = model.propagate(x)
        eval_logging(x, y, x_tr, y_tr, y_pred, rmse(y, y_pred), y_pred.var(0), "Both train/test set", _results_dir, "eval_all_preds", args.plot)

        # Run eval on entire domain (linspace)
        num_pts = 100
        x_domain = B.linspace(-6, 6, num_pts)[..., None]
        key, eps = B.randn(key, B.default_dtype, int(num_pts), 1)
        y_domain = x_domain**3. + 3*eps
        y_domain = y_domain/scale   # scale with train datasets
        y_pred = model.propagate(x_domain)
        eval_logging(x_domain, y_domain, x_tr, y_tr, y_pred, rmse(y_domain, y_pred), y_pred.var(0), "Entire domain", _results_dir, "eval_domain_preds", args.plot)

    logger.info(f"Total time: {(datetime.utcnow() - _start)} (H:MM:SS:ms)")