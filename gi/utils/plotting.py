import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

from torch import Tensor

from wbml import plot

colors = sns.color_palette("bright")
sns.set_palette(colors)
matplotlib.rcParams['figure.dpi'] = 600

def scatter_plot(ax, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor, 
        desc1: str, desc2: str,  xlabel: Optional[str]=None, ylabel: Optional[str]=None, title: Optional[str]=None):

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

    scatterplot = plot.patch(sns.scatterplot)
    scatterplot(y=y1, x=x1, label=desc1, ax=ax)
    scatterplot(y=y2, x=x2, label=desc2, ax=ax)

    # ax.set(ylim=(0.60, 1.01), xlim=(-0.005, 0.20))
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)
    ax.legend(loc='upper right', prop={'size': 12})
    
    plot.tweak(ax)

    return ax

def plot_predictions(ax, x: Tensor, y: Tensor,
        desc: str, xlabel: Optional[str]=None,
        ylabel: Optional[str]=None, title: Optional[str]=None):

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

    scatterplot = plot.patch(sns.scatterplot)
    if len(y.shape) == 3:
        for i in range(y.shape[0]):
            scatterplot(y=y[i], x=x, ax=ax)
    else:
        scatterplot(y=y, x=x, ax=ax)

    # ax.set(ylim=(0.60, 1.01), xlim=(-0.005, 0.20))
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)

    plot.tweak(ax)

    return ax

def plot_confidence(ax, x, quartiles, all: bool = False):
    assert len(quartiles) == 4 # [num quartiles x num preds]
    if x.is_cuda: x = x.detach().cpu()
    
    x_sorted, q0, q1, q2, q3 = zip(*sorted(zip(x, quartiles[0, :], quartiles[1, :], quartiles[2, :], quartiles[3, :])))
    
    if all:
        ax.fill_between(x_sorted, q0, q3, color=colors[7], alpha=0.20, label="5-95th percentile")
    ax.fill_between(x_sorted, q1, q2, color=colors[1], alpha=0.20, label="25-75th percentile")

    return ax

def line_plot(x, y, desc, xlabel=None, ylabel=None, title=None):

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    lw = 2 # linewidth=lw

    def get_values(x):
        return np.array(x.squeeze().detach().cpu())

    sns.lineplot(y=y, x=x, label=f"{desc}", ax=ax, 
                color=colors[0])

    ax.legend()
    # ax.set_xlabel(f'{}')
    # ax.set(ylim=(0.60, 1.01), xlim=(-0.005, 0.20))
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)
    ax.legend(loc='upper right', prop={'size': 12})

    plot.tweak(ax)

    return ax