from itertools import accumulate
import lab as B
import lab.torch
import numpy as np
import torch

from enum import IntEnum
import logging

# import matplotlib.pyplot as plt
# from wbml import plot

logger = logging.getLogger()


class DGP(IntEnum):
    ober_regression = 1
    sinusoid = 2


def generate_data(key, dgp, size, xmin=-4.0, xmax=4):
    if dgp == DGP.ober_regression:
        """Build train data with test data in between the train space
        Equal number of training points as test points"""
        key, xl, yl = dgp1(key, int(size / 2), xmin, xmin + ((xmax - xmin) / 4))
        key, xr, yr = dgp1(key, int(size / 2), xmax - ((xmax - xmin) / 4), xmax)
        key, x_te, y_te = dgp1(key, size * 2, xmin + ((xmax - xmin) / 4), xmax - ((xmax - xmin) / 4))

        x_all = B.concat(xl, x_te, xr, axis=0)
        y_all = B.concat(yl, y_te, yr, axis=0)
        scale = B.std(y_all)

        x_tr = B.concat(xl, xr, axis=0)
        y_tr = B.concat(yl, yr, axis=0)

        y_all = y_all / scale
        y_tr = y_tr / scale
        y_te = y_te / scale

        # scatter = plot.patch(plt.scatter)
        # plt.scatter(x_all, y_all)
        # plt.scatter(x_tr, y_tr)
        # plt.scatter(x_te, y_te)
        # plt.show()
        return key, x_all, y_all, x_tr, y_tr, x_te, y_te, scale

    elif dgp == DGP.sinusoid:
        return dgp2(key, size, xmin, xmax)
    else:
        logger.warning(f"DGP type not recognized, defaulting to DGP 1")
        return dgp1(key, size, xmin, xmax)


def dgp1(key, size, xmin=-4.0, xmax=4.0):
    """Toy (test) regression dataset from paper"""
    x = B.zeros(B.default_dtype, size, 1)

    key, x = B.rand(key, B.default_dtype, int(size), 1)

    x = x * (xmax - xmin) + xmin

    key, eps = B.randn(key, B.default_dtype, int(size), 1)
    y = x**3.0 + 3 * eps

    return key, x, y


def dgp2(key, size, xmin=-4.0, xmax=4.0):

    key, eps1 = B.rand(key, B.default_dtype, int(size), 1)
    key, eps2 = B.rand(key, B.default_dtype, int(size), 1)

    eps1, eps2 = eps1.squeeze(), eps2.squeeze()
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1).squeeze()
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    # scale = B.std(y)
    # y = y / scale

    return key, x[:, None], y[:, None]


def split_data(x, y, lb_mid=-2.0, ub_mid=2.0):
    """Split toy regression dataset from paper into two domains: ([-4, -2) U (2, 4]) & [-2, 2]"""

    idx_te = torch.logical_and((x >= lb_mid), x <= ub_mid)
    idx_tr = torch.logical_or((x < lb_mid), x > ub_mid)
    x_te, y_te = x[idx_te][:, None], y[idx_te][:, None]
    x_tr, y_tr = x[idx_tr][:, None], y[idx_tr][:, None]

    return x_tr, y_tr, x_te, y_te


def split_data_clients(x, y, splits):
    """Split data based on list of splits provided"""
    # Cannot verify that dataset is Sized
    if len(x) != len(y) or not (sum(splits) == len(x) or sum(splits) == 1):
        raise ValueError("Mismatch: len(x) != len(y) or sum of input lengths does not equal the length of the input dataset!")

    # If fractions provided, multiply to get lengths/counts
    if sum(splits) == 1.0:
        splits = [int(len(x) * split) for split in splits]

    indices = B.to_numpy(B.randperm(sum(splits)))

    return [(x[indices[offset - length : offset]], y[indices[offset - length : offset]]) for offset, length in zip(accumulate(splits), splits)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=0)

    N = 1000
    key, x, y, x_tr, y_tr, x_te, y_te, scale = generate_data(key, 1, N, xmin=-4.0, xmax=4.0)

    print(f"Scale: {scale}")
    