from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime

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
import torch
import torch.nn as nn
from gi.client import Client, GI_Client, MFVI_Client
from gi.gibnn import GIBNN_Classification
from gi.mfvi import MFVI_Classification
from gi.server import SequentialServer, SynchronousServer
from matplotlib import pyplot as plt
from slugify import slugify
from torch.utils.data import DataLoader, TensorDataset
from wbml import experiment, out

from config.config import Config, set_partition_factors
from data.split_data import generate_clients_data
from dgp import DGP, generate_data, generate_mnist, split_data_clients
from priors import Prior, build_prior
from utils.colors import Color
from utils.optimization import EarlyStopping, collect_frozen_vp, collect_vp, construct_optimizer, dampen_updates, estimate_local_vfe


def main(args, config, logger):
    # Lab variable initialization
    B.default_dtype = torch.float32
    B.epsilon = 0.0
    key = B.create_random_state(B.default_dtype, seed=args.seed)
    torch.set_printoptions(precision=10, sci_mode=False)

    # Load dataset
    if config.dgp == DGP.mnist:
        train_data, test_data = generate_mnist(data_dir=f"{_root_dir}/gi/data")
        x_tr, y_tr, x_te, y_te = (
            train_data["x"],
            train_data["y"],
            test_data["x"],
            test_data["y"],
        )
        y_tr = torch.squeeze(torch.nn.functional.one_hot(y_tr, num_classes=-1))
        y_te = torch.squeeze(torch.nn.functional.one_hot(y_te, num_classes=-1))
    elif config.dgp == DGP.uci_adult or config.dgp == DGP.uci_bank or config.dgp == DGP.uci_credit:
        key, x, y, x_tr, y_tr, x_te, y_te, scale = generate_data(key, config.dgp)

    # Split dataset.
    splits, N_balance, prop_positive, _ = generate_clients_data(x_tr, y_tr, args.num_clients, config.client_size_factor, config.class_balance_factor, args.seed)
    logger.info(f"{Color.YELLOW}Client data partition size:      {[round(x, 2) for x in N_balance]}{Color.END}")
    logger.info(f"{Color.YELLOW}Client data proportion positive: {[round(x, 2) for x in prop_positive]}{Color.END}")

    y_tr = torch.squeeze(torch.nn.functional.one_hot(y_tr.long(), num_classes=2))
    y_te = torch.squeeze(torch.nn.functional.one_hot(y_te.long(), num_classes=2))
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=args.batch, shuffle=True, num_workers=4)
    N = x_tr.shape[0]

    # Define model and clients.
    model = config.model_type(nn.functional.relu, config.bias, config.kl)
    clients: dict[str, Client] = {}

    S = args.training_samples  # number of training inference samples
    log_step = config.log_step

    # Build prior.
    dims = config.dims
    assert dims[0] == x_tr.shape[1]
    ps = build_prior(*dims, prior=args.prior, bias=config.bias)
    logger.info(f"LR: {args.lr}")

    # Build clients.
    for client_i, client_data in enumerate(splits):
        client_x_tr = client_data["x"]
        client_y_tr = torch.squeeze(torch.nn.functional.one_hot(client_data["y"].long(), num_classes=2))

        if config.model_type == gi.GIBNN_Classification:
            clients[f"client{client_i}"] = GI_Client(
                key, f"client{client_i}", client_x_tr, client_y_tr, args.M, *dims, random_z=args.random_z, nz_inits=config.nz_inits, linspace_yz=args.linspace_yz
            )
        elif config.model_type == gi.MFVI_Classification:
            clients[f"client{client_i}"] = MFVI_Client(key, f"client{client_i}", client_x_tr, client_y_tr, *dims, random_mean_init=args.rand_mean, prec_inits=config.nz_inits, S=S)
        key = clients[f"client{client_i}"].key

    # Construct server.
    server = config.server_type(clients, model, args.global_iters)
    server.train_loader = train_loader
    server.test_loader = test_loader

    # Perform PVI.
    max_global_iters = server.max_iters
    for iter in range(max_global_iters):
        server.curr_iter = iter

        # Construct frozen zs, ts by iterating over all the clients. Automatically links back the previously updated clients' t & z.
        frozen_ts, frozen_zs = collect_vp(clients)

        # Log performance of global server model.
        with torch.no_grad():
            # Resample <S> inference weights
            key, _ = model.sample_posterior(key, ps, frozen_ts, zs=frozen_zs, S=args.inference_samples, cavity_client=None)

            server.evaluate_performance()

        # Save model & client metrics.
        pd.DataFrame(server.log).to_csv(os.path.join(config.metrics_dir, f"server_log.csv"), index=False)
        for client_name, _c in clients.items():
            pd.DataFrame(_c.log).to_csv(os.path.join(config.metrics_dir, f"{client_name}_log.csv"), index=False)

        # Get next client(s).
        curr_clients = next(server)

        # Run client-local optimization.
        for idx, curr_client in enumerate(curr_clients):

            # Construct optimiser of only client's parameters.
            opt = construct_optimizer(args, config, curr_client, pvi=True)

            # Communicated posterior communicated to client in 1st iter is the prior
            if iter == 0:
                tmp_ts = {k: {curr_client.name: curr_client.t[k]} for k, _ in frozen_ts.items()}
                tmp_zs = {curr_client.name: curr_client.z} if config.model_type == gi.GIBNN_Classification else {}
            else:
                # Construct the posterior communicated to client.
                tmp_ts, tmp_zs = collect_frozen_vp(frozen_ts, frozen_zs, curr_client)  # All detached except current client.

            # Run client-local optimization
            client_data_size = curr_client.x.shape[0]
            batch_size = min(client_data_size, min(args.batch, N))
            max_local_iters = args.local_iters
            logger.info(f"CLIENT - {curr_client.name} - batch size: {batch_size} - training data size: {client_data_size}")

            # Save scores
            score_name = "local_vfe"
            scores = {score_name: []}
            min_improvement = 0.0
            patience = args.patience
            stop = EarlyStopping(patience=patience, verbose=True, score_name=score_name, delta=min_improvement)
            for client_iter in range(max_local_iters):

                # Construct client_iter-th minibatch {x, y} training data.
                inds = (B.range(batch_size) + batch_size * client_iter) % client_data_size
                x_mb = B.take(curr_client.x, inds)
                y_mb = B.take(curr_client.y, inds)

                # Run client-local optimization.
                key, local_vfe, exp_ll, kl, error = estimate_local_vfe(key, model, curr_client, x_mb, y_mb, ps, tmp_ts, tmp_zs, S, N=client_data_size)
                loss = -local_vfe
                loss.backward()
                opt.step()
                curr_client.update_nz()
                opt.zero_grad()

                if client_iter == 0 or (client_iter + 1) % log_step == 0 or (client_iter + 1) == max_local_iters:
                    logger.info(
                        f"CLIENT - {curr_client.name} - global {iter+1:2}/{max_global_iters} - local [{client_iter+1:4}/{max_local_iters:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )

                    # Save client metrics.
                    scores["local_vfe"].append(local_vfe.item())
                    curr_client.update_log(
                        {
                            "global_iteration": iter,
                            "local_iteration": client_iter,
                            "total_iteration": iter * max_local_iters + client_iter,
                            "vfe": local_vfe.item(),
                            "ll": exp_ll.item(),
                            "kl": kl.item(),
                            "error": error.item(),
                        }
                    )

                    # If score hasn't improved in past (patience*log_step) iterations, stop.
                    if stop(scores):
                        logger.info(
                            f"CLIENT - {curr_client.name} - early stopping at {client_iter+1}: {scores[score_name][-patience:]} not lower than {scores[score_name][-patience - 1]}"
                        )
                        break

                else:
                    logger.debug(
                        f"CLIENT - {curr_client.name} - global {iter+1:2}/{max_global_iters} - local [{client_iter+1:4}/{max_local_iters:4}] - local vfe: {round(local_vfe.item(), 3):13.3f}, ll: {round(exp_ll.item(), 3):13.3f}, kl: {round(kl.item(), 3):8.3f}, error: {round(error.item(), 5):8.5f}"
                    )

            # After finishing client-local optimization, dampen updates.
            if args.damp:
                dampen_updates(curr_client, args.damp, frozen_ts, frozen_zs)

    # Log global/server model post training
    server.curr_iter += 1
    with torch.no_grad():
        frozen_ts, frozen_zs = collect_vp(clients)
        key, _ = model.sample_posterior(key, ps, frozen_ts, zs=frozen_zs, S=args.inference_samples, cavity_client=None)

        server.evaluate_performance()

    # Save the state of optimizable variables
    _global_vs_state_dict = {}
    for _, _c in clients.items():
        _vs_state_dict = dict(zip(_c.vs.names, [_c.vs[_name] for _name in _c.vs.names]))
        _global_vs_state_dict.update(_vs_state_dict)
    torch.save(_global_vs_state_dict, os.path.join(config.results_dir, "model/_vs.pt"))

    # Save model & client metrics.
    server_log = pd.DataFrame(server.log)
    server_log.to_csv(os.path.join(config.metrics_dir, f"server_log.csv"), index=False)
    for client_name, _c in clients.items():
        pd.DataFrame(_c.log).to_csv(os.path.join(config.metrics_dir, f"{client_name}_log.csv"), index=False)

    import seaborn as sns
    from tueplots import figsizes, fontsizes

    with plt.rc_context({**figsizes.neurips2022(ncols=1), **fontsizes.neurips2022()}):
        fig, ax = plt.subplots(1, 1)
        x_metric = "communications"
        y_metric = "test_mll"

        sns.lineplot(data=server_log, x=x_metric, y=y_metric, ax=ax)

        ax.set_ylabel(" ".join(y_metric.split("_")))
        ax.set_xlabel(" ".join(x_metric.split("_")))

        file_name = f"server_{x_metric}_{y_metric}"
        plt.savefig(os.path.join(config.plot_dir, file_name))
        plt.show()

        fig, ax = plt.subplots(1, 1)
        x_metric = "communications"
        y_metric = "test_acc"
        sns.lineplot(data=server_log, x=x_metric, y=y_metric, ax=ax)

        ax.set_ylabel(" ".join(y_metric.split("_")))
        ax.set_xlabel(" ".join(x_metric.split("_")))

        file_name = f"server_{x_metric}_{y_metric}"
        plt.savefig(os.path.join(config.plot_dir, file_name))

    logger.info(f"Total time: {(datetime.utcnow() - config.start)} (H:MM:SS:ms)")


def set_experiment_name(args):

    name = args.server
    name += f"_{args.q}"
    if args.dgp == "A":
        dgp = "adult"
    elif args.dgp == "B":
        dgp = "bank"
    elif args.dgp == "C":
        dgp = "credit"
    name += f"_{dgp}"
    name += f"_{args.num_clients}c_{args.global_iters}g_{args.local_iters}l_{args.prior}"
    name += f"_split{args.split}"
    name += f"_{args.batch}b"
    name += f"_{args.lr}lr_{args.training_samples}S"

    if args.q == "MFVI":
        if args.rand_mean:
            name += "_rand_mean"
    else:
        name += f"_{args.M}M"

    if args.damp:
        name += f"_damp_{args.damp}"

    return name


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "-q", type=str, default="GI", choices=["GI", "MFVI"])
    parser.add_argument("--dgp", "-d", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--server", type=str, default="seq", choices=["SEQ", "SYNC"])
    parser.add_argument("--prior", type=str, help="prior", nargs="?", default="neal", choices=["neal", "std"])
    parser.add_argument("--seed", "-s", type=int, help="seed", nargs="?", default=0)
    parser.add_argument("--split", type=str, help="dataset split", nargs="?", choices=["A", "B"])
    parser.add_argument("--damp", type=float, default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--local_iters", "-l", type=int, help="client-local optimization iterations", default=1000)
    parser.add_argument("--global_iters", "-g", type=int, help="server iters (running over all clients <iters> times)", default=10)
    parser.add_argument("--num_clients", "-c", type=int, help="number clients", default=10)
    parser.add_argument("--M", "-M", type=int, help="number inducing pts", default=10)
    parser.add_argument("--plot", "-p", action="store_true", help="Plot results", default=True)
    parser.add_argument("--no_plot", action="store_true", help="Do not plot results")
    parser.add_argument("--name", type=str, help="Experiment name", default="")
    parser.add_argument("--training_samples", "-S", type=int, help="number of elbo weight samples", default=2)
    parser.add_argument("--inference_samples", "-I", type=int, help="number of inference weight samples", default=50)
    parser.add_argument("--random_z", action="store_true", help="Init GI z randomly", default=False)
    parser.add_argument("--linspace_yz", action="store_true", help="Init GI yz linearly", default=False)
    parser.add_argument("--rand_mean", action="store_true", help="Init MFVI weights N(0,1)", default=True)
    parser.add_argument("--patience", type=float, help="Init MFVI weights N(0,1)", default=20)

    args = parser.parse_args()

    if args.q == "GI":
        model_type = GIBNN_Classification
    elif args.q == "MFVI":
        model_type = MFVI_Classification

    if args.server.lower() == "seq":
        server_type = SequentialServer
    elif args.server.lower() == "sync":
        server_type = SynchronousServer
    else:
        raise ValueError(f"Unknown server type: {args.server_type}")

    if args.dgp == "A":
        dgp = DGP.uci_adult
        dim_in = 108
    elif args.dgp == "B":
        dgp = DGP.uci_bank
        dim_in = 51
    elif args.dgp == "C":
        dgp = DGP.uci_credit
        dim_in = 46

    config = Config()
    dims = [dim_in, 50, 50, 2]
    config.model_type = model_type
    config.server_type = server_type
    config.dims = dims
    config.dgp = dgp
    config.prior = Prior.NealPrior if args.prior == "neal" else Prior.StandardPrior
    config.model_type = model_type
    config.batch_size = args.batch
    config.nz_inits = [1e3 - (dims[i] + 1) for i in range(len(dims) - 1)]
    config.optimizer_params: dict = {"lr": args.lr}
    config.sep_lr = False
    set_partition_factors(args.split, config)

    # Create experiment directories
    config.name = set_experiment_name(args)
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
