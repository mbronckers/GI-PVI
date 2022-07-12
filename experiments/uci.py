from data.preprocess_data import download_datasets, process_dataset, datasets, protein_config
from dataclasses import asdict
import sys
import os
from dgp import split_data_clients

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import lab as B
import lab.torch
import torch as t
from torch.utils.data import DataLoader
import numpy as np
import math


dir_path = f"{file_dir}/data/uci"
download_datasets(root_dir=dir_path, datasets={'protein': datasets['protein']})
process_dataset(os.path.join(dir_path, "protein"), protein_config)

data_dir = lambda x: os.path.join(dir_path, "protein", x)

X, y = np.load(data_dir("x.npy")), np.load(data_dir("y.npy"))

key = B.create_random_state(B.default_dtype, seed=0)
splits = split_data_clients(key, X, y, [0.8, 0.2])
# N = len(dataset)
# train_test_split = 0.8
# fraction = math.ceil(train_test_split*N)
# train, test = t.utils.data.random_split(dataset, [fraction, N-fraction])

# train_loader, test_loader = DataLoader(train), DataLoader(test)