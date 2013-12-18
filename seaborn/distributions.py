"""Plottng functions for visualizing distributions."""
from __future__ import division
import colorsys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import moss

from seaborn.utils import (color_palette, husl_palette,
                           desaturate, _kde_support)


def boxplot(vals, groupby=None, names=None, join_rm=False, color=None,
            alpha=None, fliersize=3, linewidth=1.5, widths=.8, ax=None,
            **kwargs):
    """Wrapper for matplotlib boxplot that allows better color control.

    Parameters
    ----------
    vals : sequence of data containers
        data for plot
    groupby : grouping object
        if `vals` is a Series, this is used to group
    names : list of strings, optional
        names to plot on x axis, otherwise plots numbers
    join_rm : boolean, optional
        if True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot
    color : mpl color, sequence of colors, or seaborn palette name
        inner box color
    alpha : float
        transparancy of the inner box color
    fliersize : float, optional
        markersize for the fliers
    linewidth : float, optional
        width for the box outlines and whiskers
    ax : matplotlib axis, optional
        will plot in axis, or create new figure axis
    kwargs : additional keyword arguments to boxplot

    Returns
    -------
    ax : matplotlib axis
        axis where boxplot is plotted

    """
    if ax is None:
        ax = plt.gca()

    if isinstance(vals, pd.DataFrame):
        if names is None:
            names = vals.columns
        if vals.columns.name is not None:
            xlabel = vals.columns.name
        else:
            xlabel = None
        vals = vals.values
        ylabel = None

    elif isinstance(vals, pd.Series) and groupby is not None:
        if names is None:
            names = np.sort(pd.unique(groupby))
        if hasattr(groupby, "name"):
            xlabel = groupby.name
        ylabel = vals.name
        grouped_vals = pd.groupby(vals, groupby).values
        vals = grouped_vals.values
    else:
        xlabel = None
        ylabel = None

    boxes = ax.boxplot(vals, patch_artist=True, widths=widths, **kwargs)
    vals = np.atleast_2d(vals).T

    if color is None:
        colors = husl_palette(len(vals), l=.7)
    else:
        if hasattr(color, "__iter__") and not isinstance(color, tuple):
            colors = color
        else:
            try:
                color = mpl.colors.colorConverter.to_rgb(color)
                colors = [color for _ in vals]
            except ValueError:
                colors = color_palette(color, len(vals))

    colors = [mpl.colors.colorConverter.to_rgb(c) for c in colors]
    colors = [desaturate(c, .7) for c in colors]

    light_vals = [colorsys.rgb_to_hls(*c)[1] for c in colors]
    l = min(light_vals) * .6
    gray = (l, l, l)

    for i, box in enumerate(boxes["boxes"]):
        box.set_color(colors[i])
        if alpha is not None:
            box.set_alpha(alpha)
        box.set_edgecolor(gray)
        box.set_linewidth(linewidth)
    for i, whisk in enumerate(boxes["whiskers"]):
        whisk.set_color(gray)
        whisk.set_linewidth(linewidth)
        whisk.set_linestyle("-")
    for i, cap in enumerate(boxes["caps"]):
        cap.set_color(gray)
        cap.set_linewidth(linewidth)
    for i, med in enumerate(boxes["medians"]):
        med.set_color(gray)
        med.set_linewidth(linewidth)
    for i, fly in enumerate(boxes["fliers"]):
        fly.set_color(gray)
        fly.set_marker("d")
        fly.set_markeredgecolor(gray)
        fly.set_markersize(fliersize)

    if join_rm:
        ax.plot(range(1, len(vals.T) + 1), vals.T,
                color=gray, alpha=2. / 3)

    if names is not None:
        ax.set_xticklabels(names)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.xaxis.grid(False)
    return ax


def distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, xlabel=None, ax=None):
    """Flexibly plot a distribution of observations.

    Parameter
    a : (squeezable to) 1d array
        Observed data.
    bins : argument for matplotlib hist(), or None
        Specification of hist bins, or None to use Freedman-Diaconis rule.
    hist : bool, default True
        Whether to plot a (normed) histogram.
    kde : bool, default True
        Whether to plot a gaussian kernel density estimate.
    rug : bool, default False
        Whether to draw a rugplot on the support axis.
    fit : random variable object
        An object with `fit` method, returning a tuple that can be passed to a
        `pdf` method a positional arguments following an grid of values to
        evaluate the pdf on.
    {hist, kde, rug, fit}_kws : dictionaries
        Keyword arguments for underlying plotting functions.
    color : matplotlib color, optional
        Color to plot everything but the fitted curve in.
    vertical : bool, default False
        If True, oberved values are on y-axis.
    xlabel : string, False, or None
        Name for the x axis label. if None, will try to get it from a.name
        if False, do not set the x label.
    ax : matplotlib axis, optional
        if provided, plot on this axis

    Returns
    -------
    ax : matplotlib axis

    """
    if ax is None:
        ax = plt.gca()

    # Intelligently label the axis
    label_x = bool(xlabel)
    if xlabel is None and hasattr(a, "name"):
        xlabel = a.name
        if xlabel is not None:
            label_x = True

    # Make a a 1-d array
    a = np.asarray(a).squeeze()

    # Handle dictionary defaults
    if hist_kws is None:
        hist_kws = dict()
    if kde_kws is None:
        kde_kws = dict()
    if rug_kws is None:
        rug_kws = dict()
    if fit_kws is None:
        fit_kws = dict()

    # Get the color from the current color cycle
    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()

    if hist:
        if bins is None:
            # From http://stats.stackexchange.com/questions/798/
            h = 2 * moss.iqr(a) * len(a) ** -(1 / 3)
            bins = (a.max() - a.min()) / h
        hist_alpha = hist_kws.pop("alpha", 0.4)
        orientation = "horizontal" if vertical else "vertical"
        hist_color = hist_kws.pop("color", color)
        ax.hist(a, bins, normed=True, color=hist_color, alpha=hist_alpha,
                orientation=orientation, **hist_kws)

    if kde:
        kde_color = kde_kws.pop("color", color)
        kdeplot(a, vertical=vertical, color=kde_color, ax=ax, **kde_kws)

    if rug:
        rug_color = rug_kws.pop("color", color)
        axis = "y" if vertical else "x"
        rugplot(a, axis=axis, color=rug_color, ax=ax, **rug_kws)

    if fit is not None:
        fit_color = fit_kws.pop("color", "#282828")
        npts = fit_kws.pop("npts", 1000)
        support_thresh = fit_kws.pop("support_thresh", 1e-4)
        params = fit.fit(a)
        pdf = lambda x: fit.pdf(x, *params)
        x = _kde_support(a, pdf, npts, support_thresh)
        y = pdf(x)
        if vertical:
            x, y = y, x
        ax.plot(x, y, color=fit_color, **fit_kws)

    if label_x:
        ax.set_xlabel(xlabel)

    return ax


def kdeplot(a, shade=False, npts=1000, support_thresh=1e-4,
            support_min=-np.inf, support_max=np.inf, bw=None,
            vertical=False, ax=None, **kwargs):
    """Calculate and plot a one-dimentional kernel density estimate.

    Parameters
    ----------
    a : ndarray
        Input data.
    shade : bool, optional
        If true, shade in the area under the KDE curve.
    npts : int, optional
        Number of points in the evaluation grid.
    support_thresh : float, optional
        Draw density for values up to support_thresh * max(density).
    support_{min, max}: floats, optional
        If provided, do not draw above or below these values
        (does not affect the actual estimation)
    bw : {'scott' | 'silverman' | scalar | callable}
        name of method to determine kernel size, scalar factor, or callable
        to determine size given a kde instance
    vertical : bool
        If True, density is on x-axis.
    ax : matplotlib axis, optional
        Axis to plot on, otherwise uses current axis.
    kwargs : other keyword arguments for plot()

    Returns
    -------
    ax : matplotlib axis
        Axis with plot.

    """
    if ax is None:
        ax = plt.gca()

    # Check if a label was specified in the call
    label = kwargs.pop("label", None)

    # Otherwise check if the data object has a name
    if label is None and hasattr(a, "name"):
        label = a.name

    # Decide if we're going to add a legend
    legend = not label is None
    label = "_nolegend_" if label is None else label

    # Compute the KDE
    a = np.asarray(a)
    kde = stats.gaussian_kde(a.astype(float).ravel(), bw_method=bw)
    x = _kde_support(a, kde, npts, support_thresh)
    x = x[x >= support_min]
    x = x[x <= support_max]
    y = kde(x)
    if vertical:
        y, x = x, y

    # Find a color for the plot in a way that uses the active color cycle
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    if shade:
        ax.fill_between(x, 1e-12, y, color=color, alpha=0.25)

    # Draw the legend here
    if legend:
        ax.legend(loc="best")

    return ax


def rugplot(a, height=None, axis="x", ax=None, **kwargs):
    """Plot datapoints in an array as sticks on an axis."""
    if ax is None:
        ax = plt.gca()
    other_axis = dict(x="y", y="x")[axis]
    min, max = getattr(ax, "get_%slim" % other_axis)()
    if height is None:
        range = max - min
        height = range * .05
    if axis == "x":
        ax.plot([a, a], [min, min + height], **kwargs)
    else:
        ax.plot([min, min + height], [a, a], **kwargs)
    return ax


def violin(*args, **kwargs):
    """Deprecated old name for violinplot. Please update your code."""
    import warnings
    warnings.warn("violin() is deprecated; please use violinplot()",
                  UserWarning)
    return violinplot(*args, **kwargs)


def violinplot(vals, groupby=None, inner="box", color=None, positions=None,
               names=None, bw=None, widths=.8, alpha=None, join_rm=False,
               kde_thresh=1e-2, inner_kws=None, ax=None,  **kwargs):
    """Create a violin plot (a combination of boxplot and KDE plot).

    Parameters
    ----------
    vals : array or sequence of arrays
        data to plot
    groupby : grouping object
        if `vals` is a Series, this is used to group
    inner : box | sticks | points
        plot quartiles or individual sample values inside violin
    color : mpl color, sequence of colors, or seaborn palette name
        inner violin colors
    positions : number or sequence of numbers
        position of first violin or positions of each violin
    names : list of strings, optional
        names to plot on x axis, otherwise plots numbers
    bw : {'scott' | 'silverman' | scalar | callable}
        name of method to determine kernel size, scalar factor, or callable
        to determine size given a kde instance
    widths : float
        width of each violin at maximum density
    alpha : float, optional
        transparancy of violin fill
    join_rm : boolean, optional
        if True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot
    kde_thresh : float, optional
        proportion of maximum at which to threshold the KDE curve
    inner_kws : dict, optional
        keyword arugments for inner plot
    ax : matplotlib axis, optional
        axis to plot on, otherwise grab current axis
    kwargs : additional parameters to fill_betweenx

    Returns
    -------
    ax : matplotlib axis
        axis with violin plot

    """
    if ax is None:
        ax = plt.gca()

    if isinstance(vals, pd.DataFrame):
        if names is None:
            names = vals.columns
        if vals.columns.name is not None:
            xlabel = vals.columns.name
        else:
            xlabel = None
        ylabel = None
        vals = vals.values

    elif isinstance(vals, pd.Series) and groupby is not None:
        if hasattr(groupby, "name"):
            xlabel = groupby.name
        if names is None:
            names = np.sort(pd.unique(groupby))
        ylabel = vals.name
        grouped_vals = pd.groupby(vals, groupby).values
        vals = grouped_vals.values
    else:
        xlabel = None
        ylabel = None

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

    if color is None:
        colors = husl_palette(len(vals), l=.7)
    else:
        if hasattr(color, "__iter__") and not isinstance(color, tuple):
            colors = color
        else:
            try:
                color = mpl.colors.colorConverter.to_rgb(color)
                colors = [color for _ in vals]
            except ValueError:
                colors = color_palette(color, len(vals))

    colors = [mpl.colors.colorConverter.to_rgb(c) for c in colors]
    colors = [desaturate(c, .7) for c in colors]

    light_vals = [colorsys.rgb_to_hls(*c)[1] for c in colors]
    l = min(light_vals) * .6
    gray = (l, l, l)

    if inner_kws is None:
        inner_kws = {}

    if positions is None:
        positions = np.arange(1, len(vals) + 1)
    elif not hasattr(positions, "__iter__"):
        positions = np.arange(positions, len(vals) + positions)

    in_alpha = inner_kws.pop("alpha", .6 if inner == "points" else 1)
    in_alpha *= 1 if alpha is None else alpha
    in_color = inner_kws.pop("color", gray)
    in_marker = inner_kws.pop("marker", ".")
    in_lw = inner_kws.pop("lw", 1.5 if inner == "box" else .8)

    # Set the default linewidth if not provided in kwargs
    try:
        lw = kwargs[({"lw", "linewidth"} & set(kwargs)).pop()]
    except KeyError:
        lw = 1.5

    for i, a in enumerate(vals):
        x = positions[i]
        kde = stats.gaussian_kde(a, bw_method=bw)
        y = _kde_support(a, kde, 1000, kde_thresh)
        dens = kde(y)
        scl = 1 / (dens.max() / (widths / 2))
        dens *= scl

        ax.fill_betweenx(y, x - dens, x + dens, alpha=alpha, color=colors[i])
        if inner == "box":
            for quant in moss.percentiles(a, [25, 75]):
                q_x = kde(quant) * scl
                q_x = [x - q_x, x + q_x]
                ax.plot(q_x, [quant, quant], color=in_color,
                        linestyle=":", linewidth=in_lw, **inner_kws)
            med = np.median(a)
            m_x = kde(med) * scl
            m_x = [x - m_x, x + m_x]
            ax.plot(m_x, [med, med], color=in_color,
                    linestyle="--", linewidth=in_lw, **inner_kws)
        elif inner == "stick":
            x_vals = kde(a) * scl
            x_vals = [x - x_vals, x + x_vals]
            ax.plot(x_vals, [a, a], color=in_color,
                    linewidth=in_lw, alpha=in_alpha, **inner_kws)
        elif inner == "points":
            x_vals = [x for _ in a]
            ax.plot(x_vals, a, in_marker, color=in_color,
                    alpha=in_alpha, mew=0, **inner_kws)
        for side in [-1, 1]:
            ax.plot((side * dens) + x, y, c=gray, lw=lw)

    if join_rm:
        ax.plot(range(1, len(vals) + 1), vals,
                color=in_color, alpha=2. / 3)

    ax.set_xticks(positions)
    if names is not None:
        if len(vals) != len(names):
            raise ValueError("Length of names list must match nuber of bins")
        ax.set_xticklabels(names)
    ax.set_xlim(positions[0] - .5, positions[-1] + .5)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.xaxis.grid(False)
    return ax
