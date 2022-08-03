from __future__ import annotations

from copy import copy
import sys
import os

from typing import Optional

import numpy as np

import gi

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import lab as B
import lab.torch
import torch
from varz import Vars, namespace
from experiments.config.config import Config
from gi.client import Client, GI_Client, MFVI_Client

import logging

logger = logging.getLogger()


def construct_optimizer(args, config: Config, curr_client: Client, pvi: bool, vs: Optional[Vars] = None):
    """Constructs optimizer containing current client's parameters

    Args:
        args: Arguments namespace specifying learning rate parameters
        config (Config): Configuration object
        curr_client (Client): Client running optimization
        pvi (bool): PVI = True. Global VI = False.

    Returns:
        (torch.optim): Optimizer
    """
    if config.sep_lr:
        if isinstance(curr_client, GI_Client):
            params = [
                {"params": curr_client.get_params("ts.*_nz"), "lr": config.lr_nz},
                {"params": curr_client.get_params("zs.*_z"), "lr": config.lr_client_z},  # inducing
                {"params": curr_client.get_params("ts.*_yz"), "lr": config.lr_yz},  # pseudo obs
            ]
        else:
            params = [
                {"params": curr_client.get_params("ts.*_nz"), "lr": config.lr_nz},  # weight precisions
                {"params": curr_client.get_params("ts.*_yz"), "lr": config.lr_yz},  # weight means
            ]

        # If running global VI & optimizing ll variance
        if not pvi and not config.fix_ll:
            params.append({"params": vs.get_params("output_var"), "lr": config.lr_output_var})
    else:
        params = curr_client.get_params()

    opt = getattr(torch.optim, config.optimizer)(params, **config.optimizer_params)

    return opt


def rebuild(vs, likelihood):
    """
    For positive (constrained) variables in vs,
        we need to re-initialize the values of the objects
        to the latest vars in vs for gradient purposes

    :param likelihood: update the output variance
    :param clients: update the pseudo precision
    """

    _idx = vs.name_to_index["output_var"]
    likelihood.scale = vs.transforms[_idx](vs.get_vars()[_idx])
    return likelihood


def collect_vp(clients: dict[str, Client], client_names=None):
    """Collects the variational parameters of all clients in detached (frozen) form

    Args:
        clients (dict[str, Client]): dictionary of clients
        client_names (Optional[Client], optional): Set of clients to include.

    Returns:
        (dict, dict): A tuple of dictionaries of frozen variational parameters.
    """
    tmp_ts: dict[str, dict[str, gi.NormalPseudoObservation]] = {}
    tmp_zs: dict[str, B.Numeric] = {}

    # Construct from scratch to avoid linked copies.
    for client_name, client in clients.items():
        if client_names is None or client_name in client_names:
            if type(client) == GI_Client:
                tmp_zs[client_name] = client.z.detach().clone()

            for layer_name, client_layer_t in client.t.items():
                if layer_name not in tmp_ts:
                    tmp_ts[layer_name] = {}
                tmp_ts[layer_name][client_name] = copy(client_layer_t)

    return tmp_ts, tmp_zs


def collect_frozen_vp(frozen_ts, frozen_zs, curr_client: Client):
    """Collects the variational parameters of all clients in detached (frozen) form, except for the provided current client."""
    tmp_zs = {}
    tmp_ts = {layer_name: {curr_client.name: curr_client_layer_t} for layer_name, curr_client_layer_t in curr_client.t.items()}

    # Copy frozen zs except for cur_client
    if isinstance(curr_client, GI_Client):
        tmp_zs = {curr_client.name: curr_client.z}
        for client_name, client_z in frozen_zs.items():
            if client_name != curr_client.name:
                tmp_zs[client_name] = client_z.detach().clone()

    # Copy frozen zs
    for layer_name, layer_t in frozen_ts.items():
        if layer_name not in tmp_ts:
            tmp_ts[layer_name] = {}

        for client_name, client_layer_t in layer_t.items():
            if client_name != curr_client.name:
                tmp_ts[layer_name][client_name] = copy(client_layer_t)

    return tmp_ts, tmp_zs


def estimate_local_vfe(
    key: B.RandomState,
    model: gi.BaseBNN,
    client: gi.client.Client,
    x,
    y,
    ps: dict[str, gi.NaturalNormal],
    ts: dict[str, dict[str, gi.NormalPseudoObservation]],
    zs: dict[str, B.Numeric],
    S: B.Int,
    N: B.Int,
):
    # Sample from posterior.
    key, _ = model.sample_posterior(key=key, ps=ps, ts=ts, zs=zs, S=S, cavity_client=client.name)

    out = model.propagate(x)  # out : [S x N x Dout]

    # Compute KL divergence.
    kl = model.get_total_kl()

    # Compute the expected log-likelihood.
    exp_ll = model.compute_ell(out, y)  # [S]
    error = model.compute_error(out, y)

    # Mini-batching estimator of ELBO; (N / batch_size)
    # elbo = ((N / len(x)) * exp_ll) - kl / len(x)

    # ELBO per data point
    elbo = exp_ll - kl / N

    # Takes mean wrt q (inference samples)
    return key, elbo.mean(), exp_ll.mean(), kl.mean(), error


def dampen_updates(curr_client: Client, damping_factor: float, frozen_ts, frozen_zs):
    """Dampen the updates of the current client.

    Args:
        curr_client (Client): The current client.
        damping_factor (float): The damping factor.
    """
    rho = damping_factor
    logger.info(f"Damping updates of {curr_client.name} with factor {rho}")
    if type(curr_client) == GI_Client:

        # Handle z dampening
        # delta_z = (curr_client.z - frozen_zs[curr_client.name]).detach().clone()
        # new_z = (frozen_zs[curr_client.name] + rho * delta_z).detach().clone()
        # curr_client.vs.set_latent_vector(B.flatten(new_z), f"zs.{curr_client.name}_z", differentiable=True)

        for layer_name, frozen_t in frozen_ts.items():
            # Compute delta of curr_client's parameters.
            delta_yz = (curr_client.t[layer_name].yz - frozen_t[curr_client.name].yz).detach().clone()
            delta_nz = (curr_client.t[layer_name].nz - frozen_t[curr_client.name].nz).detach().clone()

            # Set curr_client's parameters to old curr_client parameters + delta*rho.
            new_yz = (frozen_t[curr_client.name].yz + rho * delta_yz).detach().clone()
            new_nz = (frozen_t[curr_client.name].nz + rho * delta_nz).detach().clone()
            new_nz = B.log(new_nz)  # latent vector is stored

            curr_client.vs.set_latent_vector(B.flatten(new_yz), f"ts.{curr_client.name}_{layer_name}_yz", differentiable=True)
            curr_client.vs.set_latent_vector(B.flatten(new_nz), f"ts.{curr_client.name}_{layer_name}_nz", differentiable=True)

    elif type(curr_client) == MFVI_Client:
        for layer_name, frozen_t in frozen_ts.items():
            # Compute delta of curr_client's parameters.
            delta_yz = (curr_client.t[layer_name].lam - frozen_t[curr_client.name].lam).detach().clone()
            delta_nz = (curr_client.t[layer_name].prec.diag - frozen_t[curr_client.name].prec.diag).detach().clone()

            # Set curr_client's parameters to old curr_client parameters + delta*rho.
            new_yz = (frozen_t[curr_client.name].lam + rho * delta_yz).detach().clone()
            new_nz = (frozen_t[curr_client.name].prec.diag + rho * delta_nz).detach().clone()
            new_nz = B.log(new_nz)

            curr_client.vs.set_latent_vector(B.flatten(new_yz), f"ts.{curr_client.name}_{layer_name}_yz", differentiable=True)
            curr_client.vs.set_latent_vector(B.flatten(new_nz), f"ts.{curr_client.name}_{layer_name}_nz", differentiable=True)

    # Turn on gradients again and update nz values
    curr_client.vs.requires_grad(True, *curr_client.vs.names)
    curr_client.update_nz()


def get_vs_state(vs):
    """returns dict<key=var_name, value=var_value>"""
    return dict(zip(vs.names, [vs[_name] for _name in vs.names]))


def load_vs(fpath):
    """Load saved vs state dict into new Vars container object"""
    assert os.path.exists(fpath)
    _vs_state_dict = torch.load(fpath)

    vs: Vars = Vars(B.default_dtype)
    for idx, name in enumerate(_vs_state_dict.keys()):
        if name.__contains__("output_var") or name.__contains__("nz"):
            vs.positive(_vs_state_dict[name], name=name)
        else:
            vs.unbounded(_vs_state_dict[name], name=name)

    return vs


class EarlyStopping:
    """
    Returns False if the score doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=5, score_name="elbo", verbose=False, delta=0, stash_model=False):
        """
        :param patience: How long to wait after last time score improved.
        :param verbose: Whether to print message informing of early stopping.
        :param delta: Minimum change to qualify as an improvement.
        :param stash_model: Whether to update the best model with the score.
        """
        self.patience = patience
        self.score_name = score_name
        self.verbose = verbose
        self.delta = delta
        self.stash_model = stash_model
        self.best_model = None
        self.best_score = None

    def __call__(self, scores=None, model=None):
        """
        :param scores: A dict of scores.
        :param model: Current model producing latest score.
        :return: Whether to stop early.
        """
        if scores is None:
            self.best_score = None
            if self.stash_model:
                self.best_model = model

            return

        else:
            vals = scores[self.score_name]

            # Check whether best score has been beaten.
            new_val = vals[-1]
            if self.best_score is None or new_val > self.best_score:
                self.best_score = new_val
                if self.stash_model and model is not None:
                    self.best_model = model

            # Check whether to stop.
            if len(vals) > self.patience:
                prev_vals = np.array(vals[-self.patience :])

                # Last reference value;
                ref_val = np.array(vals[-self.patience - 1]) + self.delta

                if np.all(prev_vals < ref_val):
                    return True
                else:
                    return False
            else:
                return False
