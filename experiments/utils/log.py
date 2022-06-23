from __future__ import annotations

import logging
import os
import shutil
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import gi
from config.config import Color, Config
from gi.utils.plotting import (line_plot, plot_confidence, plot_predictions,
                               scatter_plot)
from wbml import experiment, out, plot

logger = logging.getLogger()

def eval_logging(x, y, x_tr, y_tr, y_pred, error, pred_var, data_name, _results_dir, _fname, _plot_dir: str = None):
    """ Logs the model inference results and saves plots

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
    _S = y_pred.shape[0] # number of inference samples

    # Log test error and variance
    logger.info(f"{Color.WHITE} {data_name} error (RMSE): {round(error.item(), 3):3}, var: {round(y_pred.var().item(), 3):3}{Color.END}")

    # Save model predictions
    _results_eval = pd.DataFrame({
        'x_eval': x.squeeze().detach().cpu(),
        'y_eval': y.squeeze().detach().cpu(),
        'pred_errors': (y - y_pred.mean(0)).squeeze().detach().cpu(),
        'pred_var': pred_var.squeeze().detach().cpu(),
        'y_pred_mean': y_pred.mean(0).squeeze().detach().cpu()
    })
    
    for num_sample in range(_S): 
        _results_eval[f'preds_{num_sample}'] = y_pred[num_sample].squeeze().detach().cpu()
    
    _results_eval.to_csv(os.path.join(_results_dir, f"model/{_fname}.csv"), index=False)

    # Plot model predictions
    if _plot_dir:
    
        # Plot eval data, training data, and model predictions (in that order)
        _ax = scatter_plot(None, x, y, x_tr, y_tr, f"{data_name.capitalize()} data", "Training data", "x", "y", f"Model predictions on {data_name.lower()} data ({_S} samples)")
        scatterplot = plot.patch(sns.scatterplot)
        scatterplot(ax=_ax, y=y_pred.mean(0), x=x, label="Model predictions", color=gi.utils.plotting.colors[3])

        # Plot confidence bounds
        _preds_idx = [f'preds_{i}' for i in range(_S)]
        quartiles = np.quantile(_results_eval[_preds_idx], np.array((0.05,0.25,0.75,0.95)), axis=1) # [num quartiles x num preds]
        _ax = plot_confidence(_ax, x.squeeze().detach().cpu(), quartiles, all=True)
        # _ax.legend(loc='upper right', prop={'size': 12})
        plot.tweak(_ax)
        plt.savefig(os.path.join(_plot_dir, f'{_fname}.png'), pad_inches=0.2, bbox_inches='tight')
        
        # Plot all sampled functions
        ax = plot_predictions(None, x, y_pred, "Model predictions", "x", "y", f"Model predictions on {data_name.lower()} data ({_S} samples)")
        
        _sampled_funcs_dir = os.path.join(_plot_dir, "sampled_funcs")
        Path(_sampled_funcs_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(_sampled_funcs_dir, f'{_fname}_samples.png'), pad_inches=0.2, bbox_inches='tight')
        