import logging
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def train_with_tqdm(net: nn.Module, train_data: DataLoader, epochs: int, eval_data: DataLoader = None):
    """Wrapper for training models that will print the progress with tqdm nicely.

    :param net: network to be trained
    :type net: nn.Module
    :param train_data: training data
    :type train_data: DataLoader
    :param epochs: number of epochs to train for
    :type epochs: int
    :param eval_data: optional evaluation data, defaults to None
    :type eval_data: DataLoader, optional
    """
    with tqdm(range(epochs), unit="batch") as t_epoch:
        # Initialise previous loss
        prev_loss = np.inf

        for e_num, epoch in enumerate(t_epoch):
            # Update the tqdm toolbar
            t_epoch.set_description(f"Epoch {epoch}")

            # Run a training step
            loss = net.train_step(train_data)

            # If you want to check the parameter values, switch log level to debug
            logger.debug(net.optimizer.param_groups)

            # if eval_data is not None and not epoch % 20:
            if eval_data is not None:
                net.evaluate(eval_data)

                # Update tqdm progress bar
                t_epoch.set_postfix_str(f"Loss: {loss:.5f}, {net.eval_metric}: {net.eval_score:.5f}")
            else:
                t_epoch.set_postfix_str(f"Loss: {loss:.5f}, {net.eval_metric}: {net.eval_score:.5f}")

            if not hasattr(net, "best_loss") or net.best_loss is None:
                net.best_loss = loss

            # Save the latest model
            if loss <= net.best_loss or not epoch % 100:
                net.best_loss = loss
                torch.save(net.model.state_dict(), net.save_model_path)

                # Early stopping, if conditions are met
                # NOTE: this is intentionally inside the model saving loop; only early stop
                # if a model has just been saved.
                if net.early_stopping and np.abs(loss.item() - prev_loss) < net.early_stopping_thresh:
                    logger.warn(
                        f"Early stopping at epoch {e_num+1} as the absolute loss difference between the previous run and the current was less than {net.early_stopping_thresh}"
                    )
                    break

    # Save the final model
    torch.save(net.model.state_dict(), net.save_model_path)

    # Save the loss history and evaluation metric
    np.save(net.save_loss_path, np.array(net.loss_hist))
    np.save(net.save_eval_metric_path, np.array(net.eval_metric_hist))
