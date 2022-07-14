from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import gi
from config.config import Config
from utils.colors import Color
from gi.utils.plotting import line_plot, plot_confidence, plot_predictions, scatter_plot
from wbml import experiment, out, plot

logger = logging.getLogger()


def plot_client_vp(config, curr_client, iter, epoch):
    _client_plot_dir = os.path.join(config.training_plot_dir, curr_client.name)
    Path(_client_plot_dir).mkdir(parents=True, exist_ok=True)
    fig, _ax = plt.subplots(1, 1, figsize=(10, 10))
    scatterplot = plot.patch(sns.scatterplot)
    _ax.set_title(f"Variational parameters - iter {iter}, epoch {epoch}")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    scatterplot(curr_client.x, curr_client.y, label=f"Training data - {curr_client.name}", ax=_ax)
    scatterplot(curr_client.z, curr_client.get_final_yz(), label=f"Initial inducing points - {curr_client.name}", ax=_ax)

    plot.tweak(_ax)
    plt.savefig(os.path.join(_client_plot_dir, f"{iter}_{epoch}.png"), pad_inches=0.2, bbox_inches="tight")


def plot_all_inducing_pts(clients, _plot_dir):

    fig, _ax = plt.subplots(1, 1, figsize=(10, 10))
    scatterplot = plot.patch(sns.scatterplot)
    _ax.set_title(f"Inducing points and training data - all clients")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    for i, (name, c) in enumerate(clients.items()):
        scatterplot(c.x, c.y, label=f"Training data - {name}", ax=_ax)
        scatterplot(c.z, c.get_final_yz(), label=f"Initial inducing points - {name}", ax=_ax)

    plot.tweak(_ax)
    plt.savefig(os.path.join(_plot_dir, "init_zs.png"), pad_inches=0.2, bbox_inches="tight")


def eval_logging(
    x,
    y,
    x_tr,
    y_tr,
    y_pred,
    error,
    pred_var,
    data_name,
    _results_dir,
    _fname,
    _plot_dir: str = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    save_metrics: bool = False,
    plot_samples: bool = True,
):
    """Logs the model inference results and saves plots

    Args:
        x (_type_): eval input locations
        y (_type_): eval labels
        x_tr (_type_): training data
        y_tr (_type_): training labels
        y_pred (_type_): model predictions (S x Dout)
        error (_type_): (y - y_pred)
        pred_var (_type_): y_pred.var
        data_name (str): type of (x,y) dataset, e.g. "test", "train", "eval", "all"
        _results_dir (_type_): results directory to save plots
        _fname (_type_): plot file name
        _plot (bool): save plot figure
    """
    _S = y_pred.shape[0]  # number of inference samples

    # Log test error and variance
    logger.info(f"{Color.WHITE}{data_name} error (RMSE): {round(error.item(), 3):3}, var: {round(y_pred.var().item(), 3):3}{Color.END}")

    if y_pred.device != y.device:
        y_pred = y_pred.to(y.device)

    if x.shape[-1] == 1:
        _results_eval = pd.DataFrame(
            {
                "x_eval": x.squeeze().detach().cpu(),
                "y_eval": y.squeeze().detach().cpu(),
                "pred_errors": (y - y_pred.mean(0)).squeeze().detach().cpu(),
                "pred_var": pred_var.squeeze().detach().cpu(),
                "y_pred_mean": y_pred.mean(0).squeeze().detach().cpu(),
            }
        )
    else:
        _results_eval = pd.DataFrame(
            {
                "y_eval": y.squeeze().detach().cpu(),
                "pred_errors": (y - y_pred.mean(0)).squeeze().detach().cpu(),
                "pred_var": pred_var.squeeze().detach().cpu(),
                "y_pred_mean": y_pred.mean(0).squeeze().detach().cpu(),
            }
        )
    for num_sample in range(_S):
        _results_eval[f"preds_{num_sample}"] = y_pred[num_sample].squeeze().detach().cpu()

    # Save model predictions
    if save_metrics:
        _results_eval.to_csv(os.path.join(_results_dir, f"model/{_fname}.csv"), index=False)

    # Plot model predictions:
    if x.shape[-1] == 1:
        # Plot eval data, training data, and model predictions (in that order)
        _ax = scatter_plot(
            None,
            x,
            y,
            x_tr,
            y_tr,
            f"{data_name.capitalize()} data",
            "Training data",
            "x",
            "y",
            f"Model predictions on {data_name.lower()} data ({_S} samples)",
            ylim=ylim,
            xlim=xlim,
        )
        scatterplot = plot.patch(sns.scatterplot)
        scatterplot(ax=_ax, y=y_pred.mean(0), x=x, label="Model predictions (μ)", color=gi.utils.plotting.colors[3])

        # Plot confidence bounds (1 and 2 std deviations)
        _preds_idx = [f"preds_{i}" for i in range(_S)]
        # [num quartiles x num preds]
        quartiles = np.quantile(_results_eval[_preds_idx], np.array((0.02275, 0.15865, 0.84135, 0.97725)), axis=1)
        _ax = plot_confidence(_ax, x.squeeze().detach().cpu(), quartiles, all=True)

        # _ax.legend(loc='upper right', prop={'size': 12})

        plot.tweak(_ax)
        plt.savefig(os.path.join(_plot_dir, f"{_fname}.png"), pad_inches=0.2, bbox_inches="tight")

        # Plot all sampled functions
        if plot_samples:
            ax = plot_predictions(None, x, y_pred, "Model predictions", "x", "y", f"Model predictions on {data_name.lower()} data ({_S} samples)", ylim=ylim, xlim=xlim)

            _sampled_funcs_dir = os.path.join(_plot_dir, "sampled_funcs")
            Path(_sampled_funcs_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(_sampled_funcs_dir, f"{_fname}_samples.png"), pad_inches=0.2, bbox_inches="tight")
