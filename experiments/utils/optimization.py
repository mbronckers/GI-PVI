from __future__ import annotations

from copy import copy
import sys
import os

from typing import Optional

import gi

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import lab as B
import lab.torch
import torch
from varz import Vars, namespace
from experiments.config.config import Config
from gi.client import Client


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
    if args.sep_lr:
        lr = args.lr
        params = [
            {"params": curr_client.get_params("ts.*_nz"), "lr": config.lr_nz},
            {"params": curr_client.get_params("zs.*_z"), "lr": config.lr_client_z},  # inducing
            {"params": curr_client.get_params("ts.*_yz"), "lr": config.lr_yz},  # pseudo obs
        ]

        # If running global VI & optimizing ll variance
        if not pvi and not config.fix_ll:
            params.append({"params": vs.get_params("output_var"), "lr": config.lr_output_var})
    else:
        params = curr_client.get_params()

    opt = getattr(torch.optim, config.optimizer)(params, **config.optimizer_params)
    # opt = torch.optim.Adam(params, **config.optimizer_params)

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
    likelihood.var = vs.transforms[_idx](vs.get_vars()[_idx])
    return likelihood


def cavity_distributions(client: Client, ts: dict[str, dict[str, gi.NormalPseudoObservation]], zs: dict[str, B.Numeric], iter: B.Int):
    """Compute cavity distribution: all except <client>'s approximate likelihoods (and inducing locations)
    Args:
        client (Client): current client being optimized
        ts (dict[str, dict[str, gi.NormalPseudoObservation]]): all clients' approximate likelihoods
        zs (dict[str, B.Numeric]): all clients' inducing locations
        iter (B.Int): server iteration number

    Returns:
        (dict, dict): zs_cav, ts_cav
    """

    if iter == 0 or (len(zs.keys()) == 1):  #
        # If first iteration or only one client, the cavity distributions are equal to the prior
        return None, None

    # Create inducing point collections
    zs_cav: dict[str, B.Numeric] = {}
    for client_name, _client_z in zs.items():
        if client_name != client.name:
            zs_cav[client_name] = _client_z.detach().clone()

    # Create cavity distributions. Construct from scratch to avoid linked copies.
    ts_cav: dict[str, dict[str, gi.NormalPseudoObservation]] = {}
    for layer_name, _t in ts.items():
        ts_cav[layer_name] = {}
        for client_name, client_t in _t.items():
            if client_name != client.name:
                ts_cav[layer_name][client_name] = copy(client_t)

    return zs_cav, ts_cav


def collect_vp(clients: dict[str, Client]):
    """Collects the variational parameters of all clients in detached (frozen) form

    Args:
        clients (dict[str, Client]): dictionary of clients
        curr_client (Optional[Client], optional): A client whose gradient remains connected to the dictionaries returned. Defaults to None.

    Returns:
        (dict, dict): A tuple of dictionaries of frozen variational parameters.
    """
    tmp_ts: dict[str, dict[str, gi.NormalPseudoObservation]] = {}
    tmp_zs: dict[str, B.Numeric] = {}

    for client_name, client in clients.items():
        tmp_zs[client_name] = client.z.detach().clone()

        for layer_name, client_layer_t in client.t.items():
            if layer_name not in tmp_ts:
                tmp_ts[layer_name] = {}
            tmp_ts[layer_name][client_name] = copy(client_layer_t)
    return tmp_ts, tmp_zs


def collect_frozen_vp(frozen_ts, frozen_zs, curr_client: Client):
    """Collects the variational parameters of all clients in detached (frozen) form, except for the provided current client."""

    tmp_ts: dict[str, dict[str, gi.NormalPseudoObservation]] = {}
    tmp_zs: dict[str, B.Numeric] = {}

    # Copy frozen zs except for cur_client
    tmp_zs = {curr_client.name: curr_client.z}
    for client_name, client_z in frozen_zs.items():
        if client_name != curr_client.name:
            tmp_zs[client_name] = client_z.detach().clone()

    # Copy frozen ts except for cur_client
    for layer_name, layer_t in frozen_ts.items():
        if layer_name not in tmp_ts:
            tmp_ts[layer_name] = {}

        for client_name, client_layer_t in layer_t.items():
            if client_name == curr_client.name:
                tmp_ts[layer_name][curr_client.name] = curr_client.t[layer_name]  # client-layer-t is detached, need to take curr_client.t
            else:
                tmp_ts[layer_name][client_name] = copy(client_layer_t)

    return tmp_ts, tmp_zs


def estimate_local_vfe(
    key: B.RandomState,
    model: gi.GIBNN,
    client: gi.client.Client,
    x,
    y,
    ps: dict[str, gi.NaturalNormal],
    ts: dict[str, dict[str, gi.NormalPseudoObservation]],
    zs: dict[str, B.Numeric],
    S: B.Int,
    N: B.Int,
):
    # Compute cavity distributions
    key, _cache = model.sample_posterior(key, ps, ts, zs, S=S, cavity_client=client.name)
    out = model.propagate(x)  # out : [S x N x Dout]

    # Compute KL divergence.
    kl = 0.0
    for layer_name, layer_cache in _cache.items():  # stored KL in cache already
        kl += layer_cache["kl"]

    # Compute the expected log-likelihood.
    exp_ll = model.compute_ell(out, y)
    error = model.compute_error(out, y)

    # Mini-batching estimator of ELBO; (N / batch_size)
    # elbo = ((N / len(x)) * exp_ll) - kl / len(x)

    # ELBO per data point
    elbo = exp_ll - kl / N

    # Takes mean wrt q (inference samples)
    return key, elbo.mean(), exp_ll.mean(), kl.mean(), error


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
