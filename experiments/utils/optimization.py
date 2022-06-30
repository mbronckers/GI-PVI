import sys
import os

from typing import Optional

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import lab as B
import lab.torch
import torch
from varz import Vars, namespace
from config.config import Config
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


@namespace("zs")
def add_zs(vs, zs):
    """Add client inducing points to optimizable params in vs"""
    for client_name, client_z in zs.items():
        vs.unbounded(client_z, name=f"{client_name}_z")


@namespace("ts")
def add_ts(vs, ts):
    """Add client likelihood factors to optimizable params in vs"""
    for layer_name, client_dict in ts.items():
        for client_name, t in client_dict.items():
            vs.unbounded(t.yz, name=f"{client_name}_{layer_name}_yz")
            vs.positive(t.nz, name=f"{client_name}_{layer_name}_nz")


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
