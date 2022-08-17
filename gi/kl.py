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


class KL(enum.IntEnum):
    Analytical = 0
    MC = 1

    def __repr__(self) -> str:
        if self.value == 0:
            return "Analytic"
        else:
            return "MC"

    def __str__(self) -> str:
        if self.value == 0:
            return "Analytic"
        else:
            return "MC"


def parse_kl_arg(arg: str):
    if arg.lower().__contains__("analytic") or arg.lower().__contains__("exact"):
        return KL.Analytical
    elif arg.lower().__contains__("MC") or arg.lower().__contains__("approx"):
        return KL.MC
    else:
        logger.warning("KL type not recognized, defaulting to Analytical.")
        return KL.Analytical


def compute_kl(kl: KL, q, p, w):
    """Compute KL divergence between prior and posterior

    Args:
        q (_type_): Posterior distribution
        p (_type_): Prior distribution
        w (_type_): Drawn weight samples

    Raises:
        ValueError: Unspecified KL type
    """

    if kl == KL.Analytical:
        kl_qp = q.kl(p)
    elif kl == KL.MC:
        logq = q.logpdf(w)
        logp = p.logpdf(w)
        kl_qp = logq - logp  # MC estimator
    else:
        raise ValueError("Unknown KL estimator type")

    return kl_qp
