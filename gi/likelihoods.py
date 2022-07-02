import os
import sys

import torch

file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

import lab as B
import gi
from matrix import Diagonal


class NormalLikelihood:
    def __init__(self, var):
        self.var = var

    def __call__(self, x):
        return torch.distributions.normal.Normal(loc=x, scale=self.var)

    def __repr__(self) -> str:
        return f"var: {self.var}"
