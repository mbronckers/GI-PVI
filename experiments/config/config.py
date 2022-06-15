import sys
import os
from dataclasses import dataclass

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

from priors import Prior
from dgp import DGP

@dataclass
class Config:
    name: str = ""
    seed: int = 0
    plot: bool = True
    
    epochs: int = 1000
    
    N: int = 40        # Number of training data pts
    M: int = 100         # Number of inducing points
    S: int = 10         # Number of training weight samples
    I: int = 100        # Number of inference samples

    nz_init: float = 1e-3
    lr: float = 0.15
    ll_var: float = 1e-3
    
    batch_size: int = 100

    prior: Prior = Prior.NealPrior
    dgp: DGP = DGP.ober_regression

    random_z: bool = False
    bias: bool = True

    dims = [1, 50, 50, 1]

    load: str = None

################################################################

# The default config settings follow Ober et al.'s toy regression experiment details