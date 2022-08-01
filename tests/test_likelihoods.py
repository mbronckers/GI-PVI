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
from scipy.stats import multivariate_normal as mvn

logger = logging.getLogger()


def approx(x, y, **kwargs):
    x = B.to_numpy(B.dense(x))
    y = B.to_numpy(B.dense(y))
    np.testing.assert_allclose(x, y, **kwargs)


def _test_logpdf_normal(likelihood: gi.likelihoods.NormalLikelihood, mean, y):

    normal = likelihood(mean)
    normal_sp = mvn(mean[:, 0], likelihood.scale)

    np.testing.assert_allclose(normal.mean, normal_sp.mean[:, None])

    nlpdf = normal.logpdf(y)
    splpdf = B.zeros(y.shape[0])
    for i, pred in enumerate(y):
        _tmp = mvn(mean[i], cov=0.1)
        splpdf[i] = _tmp.logpdf(pred)

    # splpdf = normal_sp.logpdf(y)
    np.testing.assert_allclose(nlpdf, splpdf)

    return True


# @pytest.fixture()
# def normal1():
#     mean = B.randn(3, 1)
#     chol = B.randn(3, 3)
#     var = chol @ chol.T
#     return gi.distributions.Normal(mean, var)

# def test_logpdf_normal(normal1):
#     """ Running this requires removing [..., None] from iqf_diag in distributions.py """
#     normal1_sp = mvn(normal1.mean[:, 0], B.dense(normal1.var))
#     x = B.randn(3, 10) # Din x N
#     approx(normal1.logpdf(x), normal1_sp.logpdf(x.T), rtol=1e-6)

#     # normal1_sp.logpdf(x.T) => x.T is 10 x 3, var = 3x3, mean = 3

#     # Test the the output of `logpdf` is flattened appropriately.
#     assert B.shape(normal1.logpdf(B.ones(3, 1))) == ()
#     assert B.shape(normal1.logpdf(B.ones(3, 2))) == (2,)


def test_ll():
    key = B.create_random_state(B.default_dtype, seed=0)

    n = 10
    # key, var = B.randn(key, B.default_dtype, 1)
    var = 0.1
    key, y = B.randn(key, B.default_dtype, n)
    key, y_pred = B.randn(key, B.default_dtype, n)

    l1 = gi.likelihoods.NormalLikelihood(var)

    assert _test_logpdf_normal(l1, mean=y_pred[:, None], y=y[:, None]) == True


if __name__ == "__main__":
    logger.info("Starting tests...")

    # Set default type
    B.default_dtype = torch.float64

    # Run tests
    test_ll()

    logger.info("Completed tests.")
