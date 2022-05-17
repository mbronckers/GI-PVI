import argparse

import torch as t

from data import RegressionDataset
from utils import pytorch_setup
from utils.logging_setup import *

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Sets the log level to DEBUG')
    parser.add_argument('--seed', type=int, help='seed', nargs='?', default=0)
    
    args = parser.parse_args()

    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    if args.seed != 0: t.manual_seed(args.seed)

    logger.info('GI-PVI started...')

    # Data loading








    logger.info('GI-PVI finished...')





if __name__ == '__main__':
    main()
