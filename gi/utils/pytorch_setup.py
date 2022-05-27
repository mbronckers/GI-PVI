import logging
import lab as B
import lab.torch
import torch

logger = logging.getLogger(__name__)

# Determine if a CUDA enabled GPU is present; use it if it is
if torch.cuda.is_available():
    logger.info('CUDA available - Lab/PyTorch will use GPU')
    dev = 'cuda:0'
    B.set_global_device(dev)
else:
    logger.info('CUDA unavailable - Lab/PyTorch will use CPU')
    dev = 'cpu'

# DEVICE = torch.device(dev)

# Uncomment the below to find where gradient backpropagation issues are occurring
# torch.autograd.set_detect_anomaly(True)
