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

def _test_likelihood_normal(likelihood: gi.likelihoods.NormalLikelihood, y, y_pred):
    from scipy.stats import multivariate_normal    
    
    exp_ll = likelihood(y).logpdf(y_pred).mean(0)

    _cov = np.diag((B.ones(y_pred)*likelihood.var).cpu().detach())
    scipy_ll = multivariate_normal.logpdf(x=y.cpu().detach(), mean=y.cpu().detach(), cov=_cov)

    np.testing.assert_allclose(exp_ll, scipy_ll)


def test_ll():
    key = B.create_random_state(B.default_dtype, seed=0)
    
    n = 100

    key, var = B.randn(key, B.default_dtype, 1)
    key, y = B.randn(key, B.default_dtype, n)
    key, y_pred = B.randn(key, B.default_dtype, n)
    l1 = gi.likelihoods.NormalLikelihood(var)
    assert _test_likelihood_normal(l1, y, y_pred) == True

    
if __name__ == "__main__":
    logger.info("Starting tests...")

    # Set default type
    B.default_dtype = torch.float64

    # Run tests
    test_ll()

    logger.info("Completed tests.")