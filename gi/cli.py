import argparse

import torch as t
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import RegressionDataset, REGRESSION_CONFIG
from utils import pytorch_setup
from utils.logging_setup import *

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Sets the log level to DEBUG')
    parser.add_argument('--seed', type=int, help='seed', nargs='?', default=0)
    parser.add_argument('--task', choices=['reg', 'class'], help='Task')
    args = parser.parse_args()

    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    if args.seed != 0: t.manual_seed(args.seed)

    logger.info('GI-PVI started...')
    
    # Data loading
    if args.task == 'reg':
        params = REGRESSION_CONFIG
    else:
        params = REGRESSION_CONFIG

    data_params = {
            'train': {
                'size': 40,
                'l_lim': 0.0,
                'u_lim': 0.5},
            'test': {
                'size': 40,
                'l_lim': -0.2,
                'u_lim': 1.4
                }
            }

    # Generate data
    num_training_pts = params.batch_size
    dtype=t.float16
    regression_type = 1
    train_dataset = RegressionDataset(**data_params['train'], type=regression_type)
    test_dataset = RegressionDataset(**data_params['test'], type=regression_type)

    train_loader = DataLoader(train_dataset, **(params.dataloader))
    test_loader = DataLoader(test_dataset, **(params.dataloader))


    logger.info('GI-PVI finished...')


if __name__ == '__main__':
    main()
