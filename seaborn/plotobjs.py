"""High level plotting functions using matplotlib."""

# Except in strange circumstances, all functions in this module
# should take an ``ax`` keyword argument defaulting to None
# (which creates a new subplot) and an open-ended **kwargs to
# pass to the underlying matplotlib function being called.
# They should also return the ``ax`` object.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import moss

from seaborn.utils import ci_to_errsize


def tsplot(x, data, err_style=["ci_band"], ci=(16, 84),
           central_func=np.mean, n_boot=10000, smooth=False,
           ax=None, **kwargs):
    """Plot timeseries from a set of observations.

    Parameters
    ----------
    x : n_tp array
        x values
    data : n_obs x n_tp array
        array of timeseries data where first axis is e.g. subjects
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
    ax = kwargs.pop("ax", plt.subplot(111))

    # Bootstrap the data for confidence intervals
    boot_data = moss.bootstrap(data, n_boot=n_boot, smooth=smooth,
                               axis=0, func=central_func)
    ci = moss.percentiles(boot_data, ci, axis=0)
    central_data = central_func(data, axis=0)

    # Plot the timeseries line to get its color
    line, = ax.plot(x, central_data, **kwargs)
    color = line.get_color()
    line.remove()

    # Use subroutines to plot the uncertainty
    for style in err_style:
        try:
            plot_func = globals()["_plot_%s" % style]
        except KeyError:
            raise ValueError("%s is not a valid err_style" % style)
        plot_func(ax, x, data, boot_data, central_data, ci, color)

    # Replot the central trace so it is prominent
    ax.plot(x, central_data, color=color, **kwargs)

    return ax

# Subroutines for tsplot errorbar plotting
# ----------------------------------------


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


def regplot(x, y, xlabel="", ylabel="", markerstyle="o",
            ax=None, corr_func=stats.pearsonr, **kwargs):
    """Plot a regression scatter with correlation value.

    Parameters
    ----------
    x : sequence
        independent variables
    y : sequence
        dependent variables
    xlabel, ylabel : string, optional
        label names
    markerstyle : string, optional
        markerstyle for scatterplot
    corr_func : callable, optional
        correlation function; expected to return (r, p) double
    ax : axis object, optional
        plot in given axis; if None creates a new figure
    kwargs : further keyword arguments for regression line plot

    Returns
    -------
    ax : matplotlib axis
        axis object, either one passed in or created within function

    """
    if ax is None:
        ax = plt.subplot(111)
    a, b = np.polyfit(x, y, 1)
    ax.plot(x, y, markerstyle)
    xlim = ax.get_xlim()
    ax.plot(xlim, np.polyval([a, b], xlim), **kwargs)
    r, p = corr_func(x, y)
    ax.set_title("r = %.3f; p = %.3g%s" % (r, p, moss.sig_stars(p)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def boxplot(vals, join_rm=False, names=None, color=None, **kwargs):
    """Wrapper for matplotlib boxplot that allows better color control.

    Parameters
    ----------
    vals : sequence of data containers
        data for plot
    join_rm : boolean, optional
        if True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot
    names : list of strings, optional
        names to plot on x axis, otherwise plots numbers
    color : matplotlib color, optional
        box color
    ax : matplotlib axis, optional
        will plot in axis, or create new figure axis
    kwargs : additional keyword arguments to boxplot

    Returns
    -------
    ax : matplotlib axis
        axis where boxplot is plotted

    """
    ax = kwargs.pop("ax", plt.subplot(111))
    if color is None:
        pos = kwargs.pop("positions", [1])[0]
        line, = ax.plot(pos, np.mean(vals[0]), **kwargs)
        color = line.get_color()
        line.remove()

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

    if join_rm:
        ax.plot(range(1, len(np.transpose(vals)) + 1), np.transpose(vals),
                color=color, alpha=2./3)

    if names is not None:
        if len(vals) != len(names):
            raise ValueError("Length of names list must match nuber of bins")
        ax.set_xticklabels(names)

    return ax


def kdeplot(a, npts=1000, hist=False, nbins=20, rug=False,
            shade=False, color=None, ax=None, **kwargs):
    """Calculate and plot kernel density estimate.

    Parameters
    ----------
    a : ndarray
        input data
    npts : int, optional
        number of x points
    hist : bool, optional
        if True plots (normed) histogram of data
    nbins : int, optional
        number of bins if plotting histogram also
    rug : boolean, optional
        if True, plots rug (vertical lines at data points)
    shade : bool, optional
        if true shade under KDE curve
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
    a = np.asarray(a)
    kde = stats.gaussian_kde(a.astype(float).ravel())
    x = _kde_support(a, kde, npts)
    y = kde(x)
    color = _get_cycle_state(x, y, ax) if color is None else color
    if hist:
        ax.hist(a, nbins, normed=True, color=color, alpha=.4)
    ax.plot(x, y, color=color, **kwargs)
    if rug:
        rug_height = y.max() * .05
        rugplot(a, height=rug_height, ax=ax,
                color=color, alpha=.7, linewidth=2)
        ymin, ymax = ax.get_ylim()
    if shade:
        ax.fill_between(x, 0, y, color=color, alpha=0.25)
    return ax


def rugplot(a, height=None, ax=None, **kwargs):
    """Plot datapoints in an array as sticks on an axis."""
    if ax is None:
        ax = plt.subplot(111)
    ymin, ymax = ax.get_ylim()
    if height is None:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        height = yrange * .05
    ax.plot([a, a], [ymin, ymin + height], **kwargs)
    return ax


def violin(vals, inner="box", position=None, widths=.3, join_rm=False,
           names=None, color=None, ax=None):
    """Create a violin plot (a combination of boxplot and KDE plot.

    Parameters
    ----------
    vals : array or sequence of arrays
        data to plot
    inner : box | sticks
        plot quartiles or individual sample values inside violin
    positions : number or sequence of numbers
        position of first violin or positions of each violin
    widths : float
        width of each violin at maximum density
    join_rm : boolean, optional
        if True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot
    names : list of strings, optional
        names to plot on x axis, otherwise plots numbers
    color : matplotlib color
        color for violin fill
    ax : matplotlib axis
        axis to plot on, or None to generate new axis

    Returns
    -------
    ax: : matplotlib axis
        axis with violin plot

    """
    if ax is None:
        ax = plt.subplot(111)

    if hasattr(vals, 'shape'):
        if len(vals.shape) == 1:
            if hasattr(vals[0], 'shape'):
                vals = list(vals)
            else:
                vals = [vals]
        elif len(vals.shape) == 2:
            nr, nc = vals.shape
            if nr == 1:
                vals = [vals]
            elif nc == 1:
                vals = [vals.ravel()]
            else:
                vals = [vals[:, i] for i in xrange(nc)]
        else:
            raise ValueError("Input x can have no more than 2 dimensions")
    if not hasattr(vals[0], '__len__'):
        vals = [vals]

    vals = [np.asarray(a, float) for a in vals]

    gray = "#555555"
    if color is None:
        color = _get_cycle_state(vals[0].mean(), vals[0].mean(), ax)
    if position is None:
        position = np.arange(1, len(vals) + 1)
    elif not hasattr(position, "__iter__"):
        position = np.arange(position, len(vals) + position)
    for i, a in enumerate(vals):
        x = position[i]
        kde = stats.gaussian_kde(a)
        y = _kde_support(a, kde, 1000)
        dens = kde(y)
        scl = 1 / (dens.max() / (widths / 2))
        dens *= scl

        ax.fill_betweenx(y, x - dens, x + dens, alpha=.7, color=color)
        if inner == "box":
            for quant in moss.percentiles(a, [25, 75]):
                q_x = kde(quant) * scl
                q_x = [x - q_x, x + q_x]
                ax.plot(q_x, [quant, quant], gray,
                        linestyle=":", linewidth=1.5)
            med = np.median(a)
            m_x = kde(med) * scl
            m_x = [x - m_x, x + m_x]
            ax.plot(m_x, [med, med], gray,
                    linestyle="--", linewidth=1.2)
        elif inner == "stick":
            x_vals = kde(a) * scl
            x_vals = [x - x_vals, x + x_vals]
            ax.plot(x_vals, [a, a], gray, linewidth=.7, alpha=.7)
        for side in [-1, 1]:
            ax.plot((side * dens) + x, y, gray, linewidth=1)

    if join_rm:
        ax.plot(range(1, len(np.transpose(vals)) + 1), np.transpose(vals),
                color=color, alpha=2./3)

    ax.set_xticks(position)
    if names is not None:
        if len(vals) != len(names):
            raise ValueError("Length of names list must match nuber of bins")
        ax.set_xticklabels(names)
    ax.set_xlim(position[0] - .5, position[-1] + .5)


def _kde_support(a, kde, npts):
    """Establish support for a kernel density estimate."""
    min = a.min()
    max = a.max()
    range = max - min
    x = np.linspace(min - range, max + range, npts * 2)
    y = kde(x)
    mask = y > y.max() * 1e-4
    return x[mask]


def _get_cycle_state(x, y, ax):
    """Hack to get matplotlib cycle state."""
    line, = ax.plot(x, y)
    color = line.get_color()
    line.remove()
    return color
