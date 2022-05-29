import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

logger = logging.getLogger(__name__)

def plot(fdir, fname, x1, y1, x2, y2, 
        desc1, desc2, xlabel=None, ylabel=None, title=None):
    
    assert os.path.exists(fdir)
    logger.info(f"Saving plots in {fdir}")

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    lw = 2 # linewidth=lw

    def get_values(x):
        return np.array(x.squeeze().detach().cpu())

    sns.scatterplot(y=get_values(y1), x=get_values(x1), label=f"{desc1}", ax=ax, 
                color=colors[0])

    sns.scatterplot(y=get_values(y2), x=get_values(x2), label=f"{desc2}", ax=ax, 
                color=colors[1])

    ax.legend()
    # ax.set_xlabel(f'{}')
    # ax.set(ylim=(0.60, 1.01), xlim=(-0.005, 0.20))
    if xlabel != None: ax.set_xlabel(xlabel)
    if ylabel != None: ax.set_ylabel(ylabel)
    if title != None: ax.set_title(title)
    ax.legend(loc='upper right', prop={'size': 12})

    plt.savefig(os.path.join(fdir, f'{fname}.png'), pad_inches=0.2, bbox_inches='tight')