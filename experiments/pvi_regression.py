from __future__ import annotations
from copy import copy

import os
import shutil
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gi.client import GI_Client

from slugify import slugify
from varz import Vars, namespace
from wbml import experiment, out, plot

from utils.colors import Color
from dgp import DGP, generate_data, split_data_clients
from priors import build_prior
from utils.gif import make_gif
from utils.metrics import rmse
from utils.optimization import collect_frozen_vp, construct_optimizer, collect_vp, estimate_local_vfe
from utils.log import eval_logging, plot_client_vp, plot_all_inducing_pts
import seaborn as sns


def main(args, config, logger):
    # Lab variable initialization.
    B.default_dtype = torch.float32
    B.epsilon = 0.0
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    torch.set_printoptions(precision=10, sci_mode=False)

    # Setup regression dataset.
    N = args.N  # num/fraction training points
    key, x, y, x_tr, y_tr, x_te, y_te, scale = generate_data(key, args.data, N, xmin=-4.0, xmax=4.0)

    # Normalize data.
    if args.data == DGP.ober_regression:
        y_scale = B.std(y)
        # x_scale = B.std(x_tr)
        # x_te /= x_scale
        # x_tr /= x_scale
        y_tr /= y_scale
        y_te /= y_scale
        # x /= x_scale
        y /= y_scale

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=config.batch_size, shuffle=True, num_workers=0)
    logger.info(f"Y_scale: {scale}")
    N = x_tr.shape[0]

    # Build prior.
    M = args.M  # number of inducing points
    dims = config.dims
    ps = build_prior(*dims, prior=args.prior, bias=config.bias)

    # Likelihood variance is fixed in PVI.
    ll_var = 3 / scale if config.fix_ll else config.ll_var
    likelihood = gi.likelihoods.NormalLikelihood(ll_var)
    logger.info(f"Likelihood variance: {likelihood.var}")
    logger.info(f"LR: {config.lr_global}")

    # Define model and clients.
    model = gi.GIBNN_Regression(nn.functional.relu, config.bias, config.kl, likelihood)
    clients: dict[str, GI_Client] = {}

    # Split dataset
    logger.info(f"{Color.WHITE}Client splits: {config.client_splits}{Color.END}")
    if config.deterministic and config.num_clients == 1:
        splits = [(x_tr, y_tr)]
    else:
        key, splits = split_data_clients(key, x_tr, y_tr, config.client_splits)

    # Build clients.
    for client_i, (client_x_tr, client_y_tr) in enumerate(splits):
        clients[f"client{client_i}"] = GI_Client(
            key, f"client{client_i}", client_x_tr, client_y_tr, M, *dims, random_z=args.random_z, nz_inits=config.nz_inits, linspace_yz=config.linspace_yz
        )
        key = clients[f"client{client_i}"].key

    # Plot initial inducing points.
    if args.data == DGP.ober_regression:
        plot_all_inducing_pts(clients, config.plot_dir)

    # Optimizer parameters.
    S = args.training_samples  # number of training inference samples
    log_step = config.log_step

    # Construct server.
    server = config.server_type(clients, model, args.global_iters)
    server.train_loader = train_loader
    server.test_loader = test_loader

    # Perform PVI. 
    max_global_iters = server.max_iters
    for iter in range(max_global_iters):
        server.curr_iter = iter

        # Construct frozen zs, ts of all clients. Automatically links back the previously updated clients' t & z.
        frozen_ts, frozen_zs = collect_vp(clients)

        # Log performance of global server model.
        with torch.no_grad():
            # Resample <S> inference weights
            key, _ = model.sample_posterior(key, ps, frozen_ts, frozen_zs, S=args.inference_samples, cavity_client=None)

            server.evaluate_performance()

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

        # Get next client(s).
        curr_clients = next(server)

        # Run client-local optimization.
        for idx, curr_client in enumerate(curr_clients):

            # Construct optimiser of only client's parameters.
            opt = construct_optimizer(args, config, curr_client, pvi=True)

            # Compute global (frozen) posterior to communicate to clients.
            if iter == 0:
                # In 1st iter, only prior is communicated to clients.
                tmp_ts = {k: {curr_client.name: curr_client.t[k]} for k, _ in frozen_ts.items()}
                tmp_zs = {curr_client.name: curr_client.z}
            else:
                # All detached except current client.
                tmp_ts, tmp_zs = collect_frozen_vp(frozen_ts, frozen_zs, curr_client)

            # Run client-local optimization.
            client_data_size = curr_client.x.shape[0]
            batch_size = min(len(curr_client.x), min(args.batch_size, N))
            max_local_iters = args.local_iters
            logger.info(f"CLIENT - {curr_client.name} - batch size: {batch_size} - training data size: {client_data_size}")
            for client_iter in range(max_local_iters):

                # Construct epoch-th minibatch {x, y} training data.
                inds = (B.range(batch_size) + batch_size * client_iter) % client_data_size
                x_mb = B.take(curr_client.x, inds)
                y_mb = B.take(curr_client.y, inds)

                key, local_vfe, exp_ll, kl, error = estimate_local_vfe(key, model, curr_client, x_mb, y_mb, ps, tmp_ts, tmp_zs, S, N=client_data_size)
                loss = -local_vfe
                loss.backward()
                opt.step()
                curr_client.update_nz()
                opt.zero_grad()

                # Log results.
                if client_iter == 0 or (client_iter + 1) % log_step == 0 or (client_iter + 1) == max_local_iters:
                    logger.info(
                        f"CLIENT - {curr_client.name} - global iter {iter+1:2}/{max_global_iters} - local iter [{client_iter+1:4}/{max_local_iters:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )

                    # Save client metrics.
                    curr_client.update_log({"global_iter": iter, "local_iter": client_iter, "vfe": local_vfe.item(), "ll": exp_ll.item(), "kl": kl.item(), "error": error.item()})

                    # Only plot every <log_step> epoch
                    if args.plot and ((client_iter + 1) % log_step == 0):
                        plot_client_vp(config, curr_client, iter, client_iter)
                else:
                    logger.debug(
                        f"CLIENT - {curr_client.name} - global {iter+1:2}/{max_global_iters} - local [{client_iter+1:4}/{max_local_iters:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )

    # Log global/server model post training
    server.curr_iter += 1
    with torch.no_grad():
        frozen_ts, frozen_zs = collect_vp(clients)
        key, _ = model.sample_posterior(key, ps, frozen_ts, frozen_zs, S=args.inference_samples, cavity_client=None)

        server.evaluate_performance()

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

    # Save model metrics.
    metrics = pd.DataFrame(server.log)
    metrics.to_csv(os.path.join(config.metrics_dir, f"server_log.csv"), index=False)
    for client_name, _c in clients.items():
        # Save client log.
        metrics = pd.DataFrame(_c.log)
        metrics.to_csv(os.path.join(config.metrics_dir, f"{client_name}_log.csv"), index=False)

    if args.plot:
        for c_name in clients.keys():
            make_gif(config.plot_dir, c_name)

    model_eval(args, config, key, x, y, x_tr, y_tr, x_te, y_te, scale, model, ps, clients)

    logger.info(f"Total time: {(datetime.utcnow() - config.start)} (H:MM:SS:ms)")


def model_eval(args, config, key, x, y, x_tr, y_tr, x_te, y_te, scale, model, ps, clients):
    with torch.no_grad():
        ts, zs = collect_vp(clients)
        key, _ = model.sample_posterior(key, ps, ts, zs, S=args.inference_samples, cavity_client=None)
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

        # Plot regression domain (-1.5*x_tr_max, 1.5*x_tr_max).
        if type(config) == OberConfig:
            domain_x_max = 1.5 * B.max(x_tr).item()

            # Run eval on entire domain (linspace)
            num_pts = 100
            x_domain = B.linspace(-domain_x_max, domain_x_max, num_pts)[..., None]
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
            key, eps = B.randn(key, B.default_dtype, int(num_pts), 1)
            x_domain = B.linspace(-domain_x_max, domain_x_max, num_pts)[..., None]
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
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.fill_between(x_domain[:, 0], mean_ys[:, 0] - 2 * std_ys[:, 0], mean_ys[:, 0] + 2 * std_ys[:, 0], alpha=0.5)
            plt.plot(x_domain, mean_ys)
            lineplot = plot.patch(sns.lineplot)
            lineplot(ax=ax, y=mean_ys, x=x_domain, color=gi.utils.plotting.colors[3])
            plt.scatter(x_tr, y_tr, c="r")
            ax.set_axisbelow(True)  # Show grid lines below other elements.
            ax.grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
            plt.savefig(os.path.join(config.plot_dir, f"ober.png"), pad_inches=0.2, bbox_inches="tight")


if __name__ == "__main__":
    import warnings
    from config.protein import GI_ProteinConfig
    from config.ober import GI_OberConfig

    warnings.filterwarnings("ignore")

    # config = GI_ProteinConfig()
    config = GI_OberConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, help="seed", nargs="?", default=config.seed)
    parser.add_argument("--local_iters", "-l", type=int, help="client-local optimization iterations", default=config.local_iters)
    parser.add_argument("--global_iters", "-g", type=int, help="server iters (running over all clients <iters> times)", default=config.global_iters)
    parser.add_argument("--plot", "-p", action="store_true", help="Plot results", default=config.plot)
    parser.add_argument("--no_plot", action="store_true", help="Do not plot results")
    parser.add_argument("--name", "-n", type=str, help="Experiment name", default="")
    parser.add_argument("--M", "-M", type=int, help="number of inducing points", default=config.M)
    parser.add_argument("--N", "-N", type=int, help="number of training points", default=config.N)
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
        "--batch_size",
        "-b",
        type=int,
        help="training batch size",
        default=config.batch_size,
    )
    parser.add_argument("--data", "-d", type=int, help="dgp/dataset type", default=config.dgp)
    parser.add_argument(
        "--load",
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
            os.path.join(_root_dir, f"experiments/config/{config.location}"),
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
    logger.info(f"{Color.WHITE}Config: {config}{Color.END}")
    logger.info(f"{Color.WHITE}Args: {args}{Color.END}")

    main(args, config, logger)
