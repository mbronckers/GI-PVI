from __future__ import annotations
from copy import copy

import os
import shutil
import sys
from datetime import datetime
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

from gi.client import MFVI_Client

from slugify import slugify
from varz import Vars, namespace
from wbml import experiment, out, plot

from config.config import Color, PVIConfig
from dgp import DGP, generate_data, split_data_clients
from priors import build_prior
from utils.gif import make_gif
from utils.metrics import rmse
from utils.optimization import collect_frozen_vp, construct_optimizer, collect_vp, estimate_local_vfe
from utils.log import eval_logging, plot_client_vp, plot_all_inducing_pts


def main(args, config, logger):
    # Lab variable initialization
    B.default_dtype = torch.float64
    B.epsilon = 0.0
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    torch.set_printoptions(precision=10, sci_mode=False)

    # Setup regression dataset.
    N = args.N  # num training points
    key, x, y, x_tr, y_tr, x_te, y_te, scale = generate_data(key, args.data, N, xmin=-4.0, xmax=4.0)
    logger.info(f"Scale: {scale}")

    # Build prior
    M = args.M  # number of inducing points
    dims = config.dims
    ps = build_prior(*dims, prior=args.prior, bias=args.bias)

    # Deal with client split
    if args.num_clients != len(config.client_splits):
        raise ValueError("Number of clients specified by --num-clients does not match number of client splits in config file.")
    logger.info(f"{Color.WHITE}Client splits: {config.client_splits}{Color.END}")

    # Likelihood variance is fixed in PVI.
    if config.deterministic:
        likelihood = gi.likelihoods.NormalLikelihood(3 / scale)
    else:
        likelihood = gi.likelihoods.NormalLikelihood(3 / scale)  # (args.ll_var)
    logger.info(f"Likelihood variance: {likelihood.var}")

    # Define model
    model = gi.MFVI_Regression(nn.functional.relu, args.bias, config.kl, likelihood)

    # Build clients
    clients: dict[str, MFVI_Client] = {}

    if config.deterministic and args.num_clients > 1:
        raise ValueError("Deterministic mode is not supported with multiple clients.")
    if config.deterministic and args.num_clients == 1:
        _client = gi.Client(key, f"client0", x_tr, y_tr, M, *dims, random_z=args.random_z, nz_inits=config.nz_inits, linspace_yz=config.linspace_yz)
        key = _client.key
        clients[f"client0"] = _client
    else:
        # We use a separate key here to create consistent keys with deterministic (i.e. not calling split_data) runs of PVI.
        # otherwise, replace _tmp_key with key
        _tmp_key = B.create_random_state(B.default_dtype, seed=1)
        _tmp_key, splits = split_data_clients(_tmp_key, x_tr, y_tr, config.client_splits)
        for client_i, (client_x_tr, client_y_tr) in enumerate(splits):
            _client = gi.Client(key, f"client{client_i}", client_x_tr, client_y_tr, M, *dims, random_z=args.random_z, nz_inits=config.nz_inits, linspace_yz=config.linspace_yz)
            key = _client.key
            clients[f"client{client_i}"] = _client

    # Plot initial inducing points
    if args.data == DGP.ober_regression:
        plot_all_inducing_pts(clients, config.plot_dir)

    # Optimizer parameters
    S = args.training_samples  # number of training inference samples
    epochs = args.epochs
    log_step = config.log_step

    # Construct server.
    server = config.server_type(clients)
    if isinstance(server, SequentialServer):
        # Loop over all clients <iters> times.
        iters = args.iters * config.num_clients
    else:
        iters = args.iters

    # Perform PVI.
    for iter in range(iters):

        # Get next client(s).
        curr_clients = next(server)

        logger.info(f"SERVER - {server.name} - iter [{iter+1:2}/{iters}] - optimizing {curr_clients}")

        # Construct frozen zs, ts by iterating over all the clients. Automatically links back the previously updated clients' t & z.
        frozen_ts, frozen_zs = collect_vp(clients)

        # Log global/server model.
        with torch.no_grad():
            # Resample <_S> inference weights
            key, _ = model.sample_posterior(key, ps, frozen_ts, frozen_zs, S=args.inference_samples, cavity_client=None)

            # Run eval on entire dataset
            y_pred = model.propagate(x)
            eval_logging(
                x,
                y,
                x_tr,
                y_tr,
                y_pred,
                rmse(y, y_pred),
                y_pred.var(0),
                f"SERVER - global model - iter {iter} - train/test set",
                config.results_dir,
                f"server_all_preds_iter_{iter}",
                config.server_dir,
                plot_samples=False,
            )

        num_clients = len(curr_clients)
        for idx, curr_client in enumerate(curr_clients):

            # Construct optimiser of only client's parameters.
            opt = construct_optimizer(args, config, curr_client, pvi=True)

            logger.info(f"SERVER - {server.name} - iter [{iter+1:2}/{iters}] - {idx+1}/{num_clients} client - starting optimization of {curr_client.name}")

            # Communicated posterior communicated to client in 1st iter is the prior
            if iter == 0:
                tmp_ts = {k: {curr_client.name: curr_client.t[k]} for k, _ in frozen_ts.items()}
                tmp_zs = {curr_client.name: curr_client.z}
            else:
                # Construct the posterior communicated to client.
                tmp_ts, tmp_zs = collect_frozen_vp(frozen_ts, frozen_zs, curr_client)  # All detached except current client.

            # Run client-local optimization
            epochs = args.epochs
            client_data_size = curr_client.x.shape[0]
            batch_size = min(len(curr_client.x), min(args.batch_size, N))
            for epoch in range(epochs):

                # Construct epoch-th minibatch {x, y} training data
                inds = (B.range(batch_size) + batch_size * epoch) % client_data_size
                x_mb = B.take(curr_client.x, inds)
                y_mb = B.take(curr_client.y, inds)

                key, local_vfe, exp_ll, kl, error = estimate_local_vfe(key, model, curr_client, x_mb, y_mb, ps, tmp_ts, tmp_zs, S, N=client_data_size)
                loss = -local_vfe
                loss.backward()
                opt.step()
                curr_client.update_nz()
                opt.zero_grad()

                if epoch == 0 or (epoch + 1) % log_step == 0 or (epoch + 1) == epochs:
                    logger.info(
                        f"CLIENT - {curr_client.name} - iter {iter+1:2}/{iters} - epoch [{epoch+1:4}/{epochs:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )
                    # Only plot every <log_step> epoch
                    if args.plot and ((epoch + 1) % log_step == 0):
                        plot_client_vp(config, curr_client, iter, epoch)
                else:
                    logger.debug(
                        f"CLIENT - {curr_client.name} - iter {iter+1:2}/{iters} - epoch [{epoch+1:4}/{epochs:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )

    # Log global/server model.
    with torch.no_grad():
        frozen_ts, frozen_zs = collect_vp(clients)
        # Resample <_S> inference weights
        key, _ = model.sample_posterior(key, ps, frozen_ts, frozen_zs, S=args.inference_samples, cavity_client=None)

        # Run eval on entire dataset
        y_pred = model.propagate(x)
        eval_logging(
            x,
            y,
            x_tr,
            y_tr,
            y_pred,
            rmse(y, y_pred),
            y_pred.var(0),
            f"SERVER - global model - post training - train/test set",
            config.results_dir,
            f"server_all_preds_post_training",
            config.server_dir,
            plot_samples=False,
        )

    # Save var state
    _global_vs_state_dict = {}
    for _, _c in clients.items():
        _vs_state_dict = dict(zip(_c.vs.names, [_c.vs[_name] for _name in _c.vs.names]))
        _global_vs_state_dict.update(_vs_state_dict)
    torch.save(_global_vs_state_dict, os.path.join(_results_dir, "model/_vs.pt"))

    if args.plot:
        for c_name in clients.keys():
            make_gif(config.plot_dir, c_name)

    model_eval(args, config, key, x, y, x_tr, y_tr, x_te, y_te, scale, model, ps, clients)

    logger.info(f"Total time: {(datetime.utcnow() - config.start)} (H:MM:SS:ms)")


def model_eval(args, config, key, x, y, x_tr, y_tr, x_te, y_te, scale, model, ps, clients):
    with torch.no_grad():
        ts, zs = collect_vp(clients)

        # Resample <_S> inference weights
        # key, _ = model.sample_posterior(key, ps, ts, zs, ts_p=None, zs_p=None, S=args.inference_samples)
        key, _ = model.sample_posterior(key, ps, ts, zs, S=args.inference_samples, cavity_client=None)

        # Get <_S> predictions, calculate average RMSE, variance
        y_pred = model.propagate(x_te)

        # Log and plot results
        eval_logging(
            x_te,
            y_te,
            x_tr,
            y_tr,
            y_pred,
            rmse(y_te, y_pred),
            y_pred.var(0),
            "Test set",
            config.results_dir,
            "eval_test_preds",
            config.plot_dir,
        )

        # Run eval on entire dataset
        y_pred = model.propagate(x)
        eval_logging(
            x,
            y,
            x_tr,
            y_tr,
            y_pred,
            rmse(y, y_pred),
            y_pred.var(0),
            "Both train/test set",
            config.results_dir,
            "eval_all_preds",
            config.plot_dir,
        )

        # Run eval on entire domain (linspace)
        num_pts = 100
        x_domain = B.linspace(-6, 6, num_pts)[..., None]
        key, eps = B.randn(key, B.default_dtype, int(num_pts), 1)
        y_domain = x_domain**3.0 + 3 * eps
        y_domain = y_domain / scale  # scale with train datasets
        y_pred = model.propagate(x_domain)
        eval_logging(
            x_domain,
            y_domain,
            x_tr,
            y_tr,
            y_pred,
            rmse(y_domain, y_pred),
            y_pred.var(0),
            "Entire domain",
            config.results_dir,
            "eval_domain_preds",
            config.plot_dir,
        )

        # Run eval on entire domain (linspace)
        num_pts = 1000
        x_domain = B.linspace(-6, 6, num_pts)[..., None]
        key, eps = B.randn(key, B.default_dtype, int(num_pts), 1)
        y_domain = x_domain**3.0 + 3 * eps
        y_domain = y_domain / scale  # scale with train datasets
        y_pred = model.propagate(x_domain)
        eval_logging(
            x_domain,
            y_domain,
            x_tr,
            y_tr,
            y_pred,
            rmse(y_domain, y_pred),
            y_pred.var(0),
            "Entire domain",
            config.results_dir,
            "eval_domain_preds_fix_ylim",
            config.plot_dir,
            ylim=(-4, 4),
        )

        # Ober's plot
        mean_ys = y_pred.mean(0)
        std_ys = y_pred.std(0)
        ax = plt.gca()
        # plt.yticks(np.arange(B.min(ys), B.max(ys)+1, 1.0))
        plt.fill_between(x_domain[:, 0], mean_ys[:, 0] - 2 * std_ys[:, 0], mean_ys[:, 0] + 2 * std_ys[:, 0], alpha=0.5)
        plt.plot(x_domain, mean_ys)
        plt.scatter(x_tr, y_tr, c="r")
        ax.set_axisbelow(True)  # Show grid lines below other elements.
        ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
        plt.savefig(os.path.join(config.plot_dir, f"ober.png"), pad_inches=0.2, bbox_inches="tight")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    config = PVIConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="seed", nargs="?", default=config.seed)
    parser.add_argument("--epochs", "-e", type=int, help="client epochs", default=config.epochs)
    parser.add_argument("--iters", "-i", type=int, help="server iters (running over all clients <iters> times)", default=config.iters)
    parser.add_argument("--plot", "-p", action="store_true", help="Plot results", default=config.plot)
    parser.add_argument("--no_plot", action="store_true", help="Do not plot results")
    parser.add_argument("--name", "-n", type=str, help="Experiment name", default="")
    parser.add_argument("--M", "-M", type=int, help="number of inducing points", default=config.M)
    parser.add_argument("--N", "-N", type=int, help="number of training points", default=config.N)
    parser.add_argument("--det", action="store_true", help="Deterministic training data split and ll variance", default=config.deterministic)
    parser.add_argument(
        "--training_samples",
        "-S",
        type=int,
        help="number of training weight samples",
        default=config.S,
    )
    parser.add_argument(
        "--inference_samples",
        "-I",
        type=int,
        help="number of inference weight samples",
        default=config.I,
    )
    parser.add_argument(
        "--nz_init",
        type=float,
        help="Initial value of client's likelihood precision",
        default=config.nz_init,
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=config.lr_global)
    parser.add_argument("--ll_var", type=float, help="likelihood var", default=config.ll_var)
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        help="training batch size",
        default=config.batch_size,
    )
    parser.add_argument("--data", "-d", type=int, help="dgp/dataset type", default=config.dgp)
    parser.add_argument(
        "--load",
        "-l",
        type=str,
        help="model directory to load (e.g. experiment_name)",
        default=config.load,
    )
    parser.add_argument(
        "--random_z",
        "-z",
        action="store_true",
        help="Randomly initializes global inducing points z",
        default=config.random_z,
    )
    parser.add_argument("--prior", "-P", type=str, help="prior type", default=config.prior)
    parser.add_argument("--bias", help="Use bias vectors in BNN", default=config.bias)
    parser.add_argument(
        "--sep_lr",
        help="Use separate LRs for parameters (see config)",
        default=config.separate_lr,
    )
    parser.add_argument(
        "--num_clients",
        "-nc",
        help="Number of clients (implicit equal split)",
        default=config.num_clients,
    )
    args = parser.parse_args()

    # Create experiment directories
    config.name += f"_{args.name}"
    _start = datetime.utcnow()
    _time = _start.strftime("%m-%d-%H.%M.%S")
    _results_dir_name = "results"
    _results_dir = os.path.join(_root_dir, _results_dir_name, f"{_time}_{slugify(config.name)}")
    _wd = experiment.WorkingDirectory(_results_dir, observe=True, seed=args.seed)
    _plot_dir = os.path.join(_results_dir, "plots")
    _metrics_dir = os.path.join(_results_dir, "metrics")
    _model_dir = os.path.join(_results_dir, "model")
    _training_plot_dir = os.path.join(_plot_dir, "training")
    _server_dir = os.path.join(_plot_dir, "server")
    Path(_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_training_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(_model_dir).mkdir(parents=True, exist_ok=True)
    Path(_metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(_server_dir).mkdir(parents=True, exist_ok=True)

    if args.no_plot:
        config.plot = False
        args.plot = False
    config.start = _start
    config.start_time = _time
    config.results_dir = _results_dir
    config.wd = _wd
    config.plot_dir = _plot_dir
    config.metrics_dir = _metrics_dir
    config.model_dir = _model_dir
    config.training_plot_dir = _training_plot_dir
    config.server_dir = _server_dir

    # Save script
    if os.path.exists(os.path.abspath(sys.argv[0])):
        shutil.copy(os.path.abspath(sys.argv[0]), _wd.file("script.py"))
        shutil.copy(
            os.path.join(_root_dir, "experiments/config/config.py"),
            _wd.file("config.py"),
        )

    else:
        out("Could not save calling script.")

    #### Logging ####
    logging.config.fileConfig(
        os.path.join(file_dir, "config/logging.conf"),
        defaults={"logfilepath": _results_dir},
    )
    logger = logging.getLogger()
    np.set_printoptions(linewidth=np.inf)

    # Log program info
    logger.debug(f"Call: {sys.argv}")
    logger.debug(f"Root: {_results_dir}")
    logger.debug(f"Time: {_time}")
    logger.debug(f"Seed: {args.seed}")
    logger.info(f"{Color.WHITE}Args: {args}{Color.END}")

    main(args, config, logger)