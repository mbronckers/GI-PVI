import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

import gi
import lab as B
import lab.torch
import lab.jax
import torch

def generate_data(key, size):
    """ Toy regression dataset from paper """
    x = B.zeros(B.default_dtype, size)
    
    key, x[:int(size / 2), :] = B.rand(key, B.default_dtype, int(size / 2))
    x[:int(size / 2), :] * 2. - 4.
    key, x[int(size / 2):, :] = B.rand(key, B.default_dtype, int(size / 2))
    x[:int(size / 2), :] * 2. + 2.
    
    key, eps = B.randn(key, B.default_dtype, size)
    y = x ** 3. + 3 * eps

    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y /= scale
    
    return key, x, y

def generate_test_data(key, size):
    """ Toy (test) regression dataset from paper """
    x = B.zeros(B.default_dtype, size)
    
    key, x = B.rand(key, B.default_dtype, int(size))
    x = x * 2. + 2.    
    key, eps = B.randn(key, B.default_dtype, size)
    y = x ** 3. + 3 * eps

    # Rescale the outputs to have unit variance
    scale = B.std(y)
    y /= scale
    
    return key, x, y

def generate_data2(key, size, xmin, xmax):
    
    key, eps1 = B.rand(key, size)
    key, eps2 = B.rand(key, size)
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1)
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    return key, x, y

def build_prior(*dims: B.Int):
    """
    :param dims: BNN dimensionality [Din x *D_latents x Dout]
    """
    ps = {}
    for i in range(len(dims) - 1):
        mean = B.zeros(B.default_dtype, dims[i + 1], dims[i]) # Dout x Din
        var = B.eye(B.default_dtype, dims[i + 1], dims[i]) # Dout (batch) x Din x Din 
        ps[f"layer{i}"] = gi.NaturalNormal.from_normal(gi.Normal(mean, var))
        
    return ps

def build_zs(key: B.RandomState, M: B.Int, x, y):
    """
    Build M inducing points from data (x, y)

    :param zs: inducing inputs
    :param yz: pseudo (inducing) outputs for final layer
    """
    if M <= len(x):
        key, idx = B.randperm(key, B.default_dtype, M)
        z = x[idx]
        yz = y[idx]
    else:
        z = x
        yz = y
        key, z_ = B.randn(key, B.default_dtype, M - len(x), *x.shape[1:])
        key, yz_ = B.randn(key, B.default_dtype, M - len(x), *y.shape[1:])
        z = B.concat(z, z_)
        yz = B.concat(yz, yz_)
        
    return key, z, y
    

def build_ts(M, yz, *dims: B.Int, nz_init=1e-3):
    ts = {}
    for i in range(len(dims) - 1):
        if i == len(dims) - 1: # final layer
            _nz = B.ones(M) * nz_init
            t = gi.NormalPseudoObservation(yz, _nz)
        else:
            _nz = B.ones(M) * nz_init
            key, _yz = B.randn(key, M, dims[i + 1]) # draw yz ~ N(0, 1)
            t = gi.NormalPseudoObservation(_yz, _nz)
            
        ts[f"layer{i}"] = t
        
    return ts


if __name__ == "__main__":
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=0)
    
    key, x_tr, y_tr = generate_data(key, size=100)
    key, x_te, y_te = generate_test_data(key, size=50)
    
    # Define model (i.e. define prior).
    ps = build_prior(1, 50, 50, 1)
    zs = build_zs(key, M=10)
    ts = build_ts(x_tr, y_tr, 1, 50, 50, 1)
    