import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import moss

from .utils import ci_to_errsize
from .rcmod import get_color_list


def tsplot(x, data, color=None, err_style=["ci_band"], ci=(16, 84),
           central_func=np.mean, n_boot=10000, smooth=False,
           ax=None, **kwargs):
    """Plot timeseries from a set of observations.

    Parameters
    ----------
    x : n_tp array
        x values
    data : n_obs x n_tp array
        array of timeseries data where first axis is e.g. subjects
    color : matplotlib color
        color of plot trace and error
    err_style : list of strings
        names of ways to plot uncertainty across observations from
        set of {ci_band, ci_bars, boot_traces, obs_traces, obs_points}
    ci : two-tuple
        low, high values for confidence interval
    central_func : callable
        function to determine central trace and to pass to bootstrap
        must take an ``axis`` argument
    n_boot : int
        number of bootstrap iterations
    smooth : boolean
        whether to perform a smooth bootstrap (resample from KDE)
    ax : matplotlib axis
        axis to plot onto, or None for new figure
    kwargs : further keyword arguments for main call to plot()

    Returns
    -------
    ax : matplotlib axis
        axis with plot data

    """
    if ax is None:
        ax = plt.subplot(111)

    boot_data = moss.bootstrap(data, n_boot=n_boot, smooth=smooth,
                               axis=0, func=central_func)
    ci = moss.percentiles(boot_data, ci, axis=0)
    central_data = central_func(data, axis=0)

    line, = ax.plot(x, central_data)
    default_color = line.get_color()
    color = default_color if color is None else color
    line.remove()

    for style in err_style:
        try:
            plot_func = globals()["_plot_%s" % style]
        except KeyError:
            raise ValueError("%s is not a valid err_style" % style)
        plot_func(ax, x, data, boot_data, central_data, ci, color)
    ax.plot(x, central_data, color=color, **kwargs)


def _plot_ci_band(ax, x, data, boot_data,
                  central_data, ci, color):
    """Plot translucent error bands around the central tendancy."""
    low, high = ci
    ax.fill_between(x, low, high, color=color, alpha=0.2)


def _plot_ci_bars(ax, x, data, boot_data,
                  central_data, ci, color):
    """Plot error bars at each data point."""
    err = ci_to_errsize(ci, central_data)
    ax.errorbar(x, central_data, yerr=err, color=color)


def _plot_boot_traces(ax, x, data, boot_data,
                      central_data, ci, color):
    """Plot 250 traces from bootstrap."""
    ax.plot(x, boot_data[:250].T, color=color, alpha=0.25, linewidth=0.25)


def _plot_obs_traces(ax, x, data, boot_data,
                     central_data, ci, color):
    """Plot a trace for each observation in the original data."""
    ax.plot(x, data.T, color=color, alpha=0.2)


def _plot_obs_points(ax, x, data, boot_data,
                     central_data, ci, color):
    """Plot each original data point discretely."""
    ax.plot(x, data.T, "o", color=color, alpha=0.5, markersize=3)


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
    if ax is None:
        ax = plt.subplot(111)
    a, b = np.polyfit(x, y, 1)
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


def kdeplot(a, npts=1000, hist=False, nbins=20, ax=None, **kwargs):
    """Calculate and plot kernel density estimate.

    Parameters
    ----------
    a : ndarray
        input data
    npts : int, optional
        number of x points
    hist : bool, optional
        if True plots (normed) histogram of data
    hist_bins : int, optional
        number of bins if plotting histogram also
    ax : matplotlib axis, optional
        axis to plot on, otherwise creates new one
    kwargs : other keyword arguments for plot()

    Returns
    -------
    ax : matplotlib axis
        axis with plot

    """
    if ax is None:
        ax = plt.subplot(111)
    kde = stats.gaussian_kde(a)
    min = a.min()
    max = a.max()
    range = max - min
    low = min - range * .1
    high = max + range * .1
    x = np.linspace(low, high, npts)
    y = kde(x)
    ax.hist(a, nbins, normed=True, alpha=.5)
    ax.plot(x, y, **kwargs)
    return ax
