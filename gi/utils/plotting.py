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
matplotlib.rcParams['figure.dpi'] = 600

def scatter_plot(x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor, 
        desc1: str, desc2: str,  xlabel: Optional[str]=None, ylabel: Optional[str]=None, title: Optional[str]=None):

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    def get_values(x):
        return np.array(x.squeeze().detach().cpu())

    sns.scatterplot(y=get_values(y1), x=get_values(x1), label=f"{desc1}", ax=ax, 
                color=colors[0])

    sns.scatterplot(y=get_values(y2), x=get_values(x2), label=f"{desc2}", ax=ax, 
                color=colors[1])

    ax.legend()
    # ax.set(ylim=(0.60, 1.01), xlim=(-0.005, 0.20))
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)
    ax.legend(loc='upper right', prop={'size': 12})
    
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