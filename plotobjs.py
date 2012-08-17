import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import moss


def regplot(x, y, ax=None, xlabel=None, ylabel=None, corr_func=stats.pearsonr):
    """Plot a regression scatter with correlation value.
    
    Parameters
    ----------
    x : sequence
        independent variables
    y : sequence
        dependent variables
    ax : axis object, optional
        plot in given axis; if None creates a new figure
    xlabel, ylabel : string, optional
        label names
    corr_func : callable, optional
        correlation function; expected to return (r, p) double

    Returns
    -------
    ax : matplotlib axis
        axis object, either one passed in or created within function
    
    """
    a, b = np.polyfit(x, y, 1)
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(x, y, "o")
    xlim = ax.get_xlim()
    ax.plot(xlim, np.polyval([a, b], xlim))
    r, p = stats.pearsonr(x, y)
    ax.set_title("r = %.3f; p = %.3g%s" %  (r, p, moss.sig_stars(p)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
