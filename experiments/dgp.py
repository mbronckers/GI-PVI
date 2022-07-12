from itertools import accumulate
import lab as B
import lab.torch
import numpy as np
import torch

from enum import IntEnum
import logging
import os
# import matplotlib.pyplot as plt
# from wbml import plot

logger = logging.getLogger()


class DGP(IntEnum):
    ober_regression = 1
    sinusoid = 2
    mnist = 3
    cifar = 4
    uci_protein = 5


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

        return key, x_all, y_all, x_tr, y_tr, x_te, y_te, scale

    elif dgp == DGP.sinusoid:
        # return dgp2(key, size, xmin, xmax)
        logger.warning(f"DGP2 is not fixed yet. Defaulting to DGP 1...")
    
    elif dgp == DGP.uci_protein:
        file_dir = os.path.dirname(__file__)
        dir_path = f"{file_dir}/data/uci"
        X, y = uci_protein(dir_path)
        scale = B.std(y)
        X = torch.from_numpy(X).clone()
        y = torch.from_numpy(y).clone()

        key, splits = split_data_clients(key, X, y, [0.8, 0.2])
        x_tr, y_tr = splits[0]
        x_te, y_te = splits[1]
        
        return key, X, y, x_tr, y_tr, x_te, y_te, scale

    elif dgp == DGP.mnist:
        train_data, test_data = generate_mnist(data_dir="data")
        return key, train_data["x"], train_data["y"], test_data["x"], test_data["y"]
    
    elif dgp == DGP.cifar:
        logger.warning(f"CIFAR10 is not supported yet. Defaulting to DGP 1...")
    
    else:
        logger.warning(f"DGP type not recognized. Defaulting to DGP 1...")

    return dgp1(key, size, xmin, xmax)


def dgp1(key, size, xmin=-4.0, xmax=4.0):
    """Toy (test) regression dataset from paper"""
    x = B.zeros(B.default_dtype, size, 1)

    key, x = B.rand(key, B.default_dtype, int(size), 1)

    x = x * (xmax - xmin) + xmin

    key, eps = B.randn(key, B.default_dtype, int(size), 1)
    y = x**3.0 + 3 * eps

    return key, x, y

def uci_protein(dir_path):
    from data.preprocess_data import download_datasets, process_dataset, datasets, protein_config
    download_datasets(root_dir=dir_path, datasets={'protein': datasets['protein']})
    process_dataset(os.path.join(dir_path, "protein"), protein_config)
    data_dir = lambda x: os.path.join(dir_path, "protein", x)

    X, y = np.load(data_dir("x.npy")), np.load(data_dir("y.npy"))

    return X, y

def dgp2(key, size, xmin=-4.0, xmax=4.0):

    key, eps1 = B.rand(key, B.default_dtype, int(size), 1)
    key, eps2 = B.rand(key, B.default_dtype, int(size), 1)

    eps1, eps2 = eps1.squeeze(), eps2.squeeze()
    x = B.expand_dims(eps2 * (xmax - xmin) + xmin, axis=1).squeeze()
    y = x + 0.3 * B.sin(2 * B.pi * (x + eps2)) + 0.3 * B.sin(4 * B.pi * (x + eps2)) + eps1 * 0.02

    # scale = B.std(y)
    # y = y / scale

    return key, x[:, None], y[:, None]


def generate_cifar(augment: bool):
    from torchvision import transforms, datasets
    import torch as t

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    augment = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

    if augment:
        transform_train = transforms.Compose([augment, transform])
    else:
        transform_train = transform

    train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform_train)
    num_classes = max(train_dataset.targets) + 1
    test_dataset = datasets.CIFAR10("data", train=False, transform=transform)


def generate_mnist(data_dir):
    from torchvision import transforms, datasets
    import torch as t

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    train_data = {
        "x": ((train_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": train_set.targets[..., None],
    }

    test_data = {
        "x": ((test_set.data - 0) / 255).reshape(-1, 28 * 28),
        "y": test_set.targets[..., None],
    }

    return train_data, test_data


def split_data(x, y, lb_mid=-2.0, ub_mid=2.0):
    """Split toy regression dataset from paper into two domains: ([-4, -2) U (2, 4]) & [-2, 2]"""

    idx_te = torch.logical_and((x >= lb_mid), x <= ub_mid)
    idx_tr = torch.logical_or((x < lb_mid), x > ub_mid)
    x_te, y_te = x[idx_te][:, None], y[idx_te][:, None]
    x_tr, y_tr = x[idx_tr][:, None], y[idx_tr][:, None]

    return x_tr, y_tr, x_te, y_te


def split_data_clients(key, x, y, splits):
    """Split data based on list of splits provided"""
    # Cannot verify that dataset is Sized
    if len(x) != len(y) or not (sum(splits) == len(x) or sum(splits) == 1):
        raise ValueError("Mismatch: len(x) != len(y) or sum of input lengths does not equal the length of the input dataset!")

    # If fractions provided, multiply to get lengths/counts
    if sum(splits) == 1.0:
        splits = [int(len(x) * split) for split in splits]

    key, indices = B.randperm(key, B.default_dtype, sum(splits))

    indices = B.to_numpy(indices)

    return key, [(x[indices[offset - length : offset]], y[indices[offset - length : offset]]) for offset, length in zip(accumulate(splits), splits)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Lab variable initialization
    B.default_dtype = torch.float64
    key = B.create_random_state(B.default_dtype, seed=0)

    N = 40
    key, x, y, x_tr, y_tr, x_te, y_te, scale = generate_data(key, 1, N, xmin=-4.0, xmax=4.0)

    print(f"Scale: {scale}")
