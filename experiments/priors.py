from __future__ import annotations
import enum

import os
import sys
from datetime import datetime
from typing import Callable, Union

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import gi
import lab as B
import lab.torch
import logging
import logging.config

logger = logging.getLogger()


class Prior(enum.IntEnum):
    StandardPrior = 0
    NealPrior = 1


def parse_prior_arg(arg: str):
    if arg.lower().__contains__("standard") or arg.lower().__contains__("normal"):
        return Prior.StandardPrior
    elif arg.lower().__contains__("neal"):
        return Prior.NealPrior
    else:
        logger.warning("Prior type not recognized, defaulting to NealPrior.")
        return Prior.NealPrior


def build_prior(*dims: B.Int, prior: Union[Prior, str], bias: bool):
    """
    :param dims: BNN dimensionality [Din x *D_latents x Dout]
    """

    if type(prior) == str:
        prior = parse_prior_arg(prior)

    ps = {}

    for i in range(len(dims) - 1):
        dim_in = dims[i] + 1 if bias else dims[i]
        mean = B.zeros(B.default_dtype, dims[i + 1], dim_in, 1)  # [Dout x Din+bias x 1]

        # if prior == Prior.StandardPrior:
        #     var = B.eye(B.default_dtype, dim_in)
        # elif prior == Prior.NealPrior:
        #     var = (1 / dim_in) * B.eye(B.default_dtype, dim_in)
        # elif prior == Prior.HePrior:
        #     var = (2 / dim_in) * B.eye(B.default_dtype, dim_in)

        # # [Dout x Din+bias x Din+bias], i.e. [batch x Din x Din]
        # var = B.tile(var, dims[i + 1], 1, 1)
        # ps[f"layer{i}"] = gi.NaturalNormal.from_normal(gi.Normal(mean, var))

        ### PRECISION
        if prior == Prior.StandardPrior:
            prec = B.eye(B.default_dtype, dim_in)
        elif prior == Prior.NealPrior:
            prec = dim_in * B.eye(B.default_dtype, dim_in)

        # [Dout x Din+bias x Din+bias], i.e. [batch x Din x Din]
        prec = B.tile(prec, dims[i + 1], 1, 1)
        ps[f"layer{i}"] = gi.NaturalNormal(mean, prec)

    return ps
