import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import moss

from .rcmod import get_color_list


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
    ax.set_title("r = %.3f; p = %.3g%s" % (r, p, moss.sig_stars(p)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def boxplot(vals, color=None, ax=None, **kwargs):
    """Wrapper for matplotlib boxplot that allows better color control.

    Parameters
    ----------
    vals : sequence of data containers
        data for plot
    color : matplotlib color
        box color
    ax : matplotlib axis, optional
        will plot in axis, or create new figure axis
    kwargs : additional keyword arguments to boxplot

    Returns
    -------
    ax : matplotlib axis
        axis where boxplot is plotted

    """
    if ax is None:
        ax = plt.subplot(111)
    if color is None:
        color = get_color_list()[0]

    boxes = ax.boxplot(vals, patch_artist=True, **kwargs)

    gray = "#555555"
    for i, box in enumerate(boxes["boxes"]):
        box.set_color(color)
        box.set_alpha(.7)
        box.set_linewidth(1.5)
        box.set_edgecolor(gray)
    for i, whisk in enumerate(boxes["whiskers"]):
        whisk.set_color(gray)
        whisk.set_linewidth(2)
        whisk.set_alpha(.7)
        whisk.set_linestyle("-")
    for i, cap in enumerate(boxes["caps"]):
        cap.set_color(gray)
        cap.set_linewidth(1.5)
        cap.set_alpha(.7)
    for i, med in enumerate(boxes["medians"]):
        med.set_color(gray)
        med.set_linewidth(1.5)
    for i, fly in enumerate(boxes["fliers"]):
        fly.set_color(gray)
        fly.set_marker("d")
        fly.set_alpha(.6)

    return ax
