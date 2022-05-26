import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

import lab as B
import gi
from matrix import Diagonal

class NormalLikelihood:
    def __init__(self, var):
        self.var = var

    def __call__(self, x):
        var = B.ones(x) * self.var
        var = Diagonal(var)
        return gi.distributions.Normal(x, var)