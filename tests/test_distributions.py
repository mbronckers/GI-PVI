import os
import sys

# Insert package manually
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))

import gi
import numpy as np
import lab as B
import lab.torch
import torch
import pytest
import logging

logger = logging.getLogger()

def approx(x, y, **kwargs):
    x = B.to_numpy(B.dense(x))
    y = B.to_numpy(B.dense(y))
    np.testing.assert_allclose(x, y, **kwargs)
    
def _test_kl_normal(n1: gi.distributions.Normal, n2: gi.distributions.Normal):
    gi_kl = n1.kl(n2)
    
    # Define torch distributions and check our kl = there kl.
    n1_torch = torch.distributions.MultivariateNormal(
        loc=B.cast(torch.float64, B.squeeze(n1.mean, -1)), 
        covariance_matrix=B.cast(torch.float64, B.dense(n1.var))
    )
    n2_torch = torch.distributions.MultivariateNormal(
        loc=B.cast(torch.float64, B.squeeze(n2.mean, -1)), 
        covariance_matrix=B.cast(torch.float64, B.dense(n2.var))
    )
    torch_kl = torch.distributions.kl_divergence(n1_torch, n2_torch)
    
    approx(gi_kl, torch_kl, rtol=1e-6)
    
    return True
    
def _test_kl_naturalnormal(n1: gi.distributions.NaturalNormal, n2: gi.distributions.NaturalNormal):
    gi_kl = n1.kl(n2)
    
    # Define torch distributions and check our kl = there kl.
    n1_torch = torch.distributions.MultivariateNormal(
        loc=B.cast(torch.float64, B.dense(n1.mean[..., 0])), 
        covariance_matrix=B.cast(torch.float64, B.dense(n1.var))
    )
    n2_torch = torch.distributions.MultivariateNormal(
        loc=B.cast(torch.float64, B.dense(n2.mean[..., 0])), 
        covariance_matrix=B.cast(torch.float64, B.dense(n2.var))
    )
    torch_kl = torch.distributions.kl_divergence(n1_torch, n2_torch)
    
    approx(gi_kl, torch_kl)

    return True

def test_kl():
    key = B.create_random_state(B.default_dtype, seed=0)
    
    n = 100
    key, m1 = B.randn(key, B.default_dtype, n)
    key, m2 = B.randn(key, B.default_dtype, n)
    key, L1 = B.randn(key, B.default_dtype, n, n)
    key, L2 = B.randn(key, B.default_dtype, n, n)
    C1, C2 = B.mm(L1, B.transpose(L1)), B.mm(L2, B.transpose(L2))
    
    n1 = gi.distributions.Normal(m1, C1)
    n2 = gi.distributions.Normal(m2, C2)
    assert _test_kl_normal(n1, n2) == True
    
    # B.squeeze doesn't accept axes... Use this instead.
    lam1 = B.mm(B.pd_inv(C1), m1)[..., 0]
    lam2 = B.mm(B.pd_inv(C2), m2)[..., 0]
    n1 = gi.distributions.NaturalNormal(lam1, B.pd_inv(C1))
    n2 = gi.distributions.NaturalNormal(lam2, B.pd_inv(C2))
    assert _test_kl_naturalnormal(n1, n2) == True
    
    # Testing with batch dimension.
    b = 10
    key, m1 = B.randn(key, B.default_dtype, b, n, 1)
    key, m2 = B.randn(key, B.default_dtype, b, n, 1)
    key, L1 = B.randn(key, B.default_dtype, b, n, n)
    key, L2 = B.randn(key, B.default_dtype, b, n, n)
    C1, C2 = B.mm(L1, B.transpose(L1)), B.mm(L2, B.transpose(L2))
    
    n1 = gi.distributions.Normal(m1, C1)
    n2 = gi.distributions.Normal(m2, C2)
    assert _test_kl_normal(n1, n2) == True
    
    # B.squeeze doesn't accept axes... Use this instead.
    lam1 = B.mm(B.pd_inv(C1), m1)
    lam2 = B.mm(B.pd_inv(C2), m2)
    n1 = gi.distributions.NaturalNormal(lam1, B.pd_inv(C1))
    n2 = gi.distributions.NaturalNormal(lam2, B.pd_inv(C2))
    assert _test_kl_naturalnormal(n1, n2) == True
    
def test_distribution_conversion():

    key = B.create_random_state(B.default_dtype, seed=0)
    
    n = 100
    key, m1 = B.randn(key, B.default_dtype, n)
    key, m2 = B.randn(key, B.default_dtype, n)
    key, L1 = B.randn(key, B.default_dtype, n, n)
    key, L2 = B.randn(key, B.default_dtype, n, n)
    C1, C2 = B.mm(L1, B.transpose(L1)), B.mm(L2, B.transpose(L2))
    
    n1 = gi.distributions.Normal(m1, C1)
    n2 = gi.distributions.Normal(m2, C2)
    
    lam1 = B.mm(B.pd_inv(C1), m1)[..., 0]
    lam2 = B.mm(B.pd_inv(C2), m2)[..., 0]
    nn1 = gi.distributions.NaturalNormal(lam1, B.pd_inv(C1))
    nn2 = gi.distributions.NaturalNormal(lam2, B.pd_inv(C2))

    _n1 = gi.distributions.Normal.from_naturalnormal(nn1)
    _n2 = gi.distributions.Normal.from_naturalnormal(nn2)

    approx(B.squeeze(_n1.mean), B.squeeze(n1.mean), rtol=1e-6)
    approx(B.squeeze(_n2.mean), B.squeeze(n2.mean), rtol=1e-6)
    approx(_n1.var, n1.var, rtol=1e-6)
    approx(_n2.var, n2.var, rtol=1e-6)

    _nn1 = gi.distributions.NaturalNormal.from_normal(n1)
    _nn2 = gi.distributions.NaturalNormal.from_normal(n2)

    approx(B.squeeze(_nn1.lam), B.squeeze(nn1.lam), rtol=1e-6)
    approx(B.squeeze(_nn2.lam), B.squeeze(nn2.lam), rtol=1e-6)
    approx(_nn1.prec, nn1.prec, rtol=1e-6)
    approx(_nn2.prec, nn2.prec, rtol=1e-6)


if __name__ == "__main__":
    print("Starting tests...")

    # Set default type
    B.default_dtype = torch.float64

    # Run tests
    test_kl()
    test_distribution_conversion()

    print("Completed tests.")