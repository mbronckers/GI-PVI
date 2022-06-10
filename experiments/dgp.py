import lab as B
import lab.torch
import numpy as np
import torch 

from enum import IntEnum
import logging

logger = logging.getLogger()

class DGP(IntEnum):
    ober_regression = 1 
    sinusoid = 2

def generate_data(key, dgp, size, xmin=-4., xmax=4):
    if dgp == DGP.ober_regression:
        return dgp1(key, size, xmin, xmax)
    elif dgp == DGP.sinusoid:
        return dgp2(key, size, xmin, xmax)
    else:
        logger.warning(f"DGP type not recognized, defaulting to DGP 1")
        return dgp1(key, size, xmin, xmax)

def dgp1(key, size, xmin=-4., xmax=4.):
    """ Toy (test) regression dataset from paper """
    x = B.zeros(B.default_dtype, size, 1)
    
    key, x = B.rand(key, B.default_dtype, int(size), 1)

    x = x * (xmax-xmin) + xmin
    
    key, eps = B.randn(key, B.default_dtype, int(size), 1)
    y = x ** 3. + 3*eps
    
    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y = y/scale
    
    return key, x, y

def dgp2(key, size, xmin=-4., xmax=4.):
    
    key, eps1 = B.rand(key, B.default_dtype, int(size), 1)
    key, eps2 = B.rand(key, B.default_dtype, int(size), 1)

    eps1, eps2 = eps1.squeeze(), eps2.squeeze()
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1).squeeze()
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    return key, x[:, None], y[:, None]

def split_data(x, y):
    """ Split toy regression dataset from paper into two domains: ([-4, -2) U (2, 4]) & [-2, 2]"""

    idx_te = torch.logical_and((x >= -2.), x <= 2.)
    idx_tr = torch.logical_or((x < -2.), x > 2.)
    x_te, y_te = x[idx_te][:, None], y[idx_te][:, None]
    x_tr, y_tr = x[idx_tr][:, None], y[idx_tr][:, None]
    
    return x_tr, y_tr, x_te, y_te

