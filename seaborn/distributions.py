"""Plottng functions for visualizing distributions."""
from __future__ import division
import colorsys
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from six.moves import range
import warnings
import moss

from seaborn.utils import (color_palette, husl_palette, blend_palette,
                           desaturate, _kde_support)


def _box_reshape(vals, groupby, names, order):
    """Reshape the box/violinplot input options and find plot labels."""

    # Set up default label outputs
    xlabel, ylabel = None, None

    # If order is provided, make sure it was used correctly
    if order is not None:
        # Assure that order is the same length as names, if provided
        if names is not None:
            if len(order) != len(names):
                raise ValueError("`order` must have same length as `names`")
        # Assure that order is only used with the right inputs
        is_pd = isinstance(vals, pd.Series) or isinstance(vals, pd.DataFrame)
        if not is_pd:
            raise ValueError("`vals` must be a Pandas object to use `order`.")

    # Handle case where data is a wide DataFrame
    if isinstance(vals, pd.DataFrame):
        if order is not None:
            vals = vals[order]
        if names is None:
            names = vals.columns.tolist()
        if vals.columns.name is not None:
            xlabel = vals.columns.name
        vals = vals.values.T

    # Handle case where data is a long Series and there is a grouping object
    elif isinstance(vals, pd.Series) and groupby is not None:
        groups = pd.groupby(vals, groupby).groups
        order = sorted(groups) if order is None else order
        if hasattr(groupby, "name"):
            if groupby.name is not None:
                xlabel = groupby.name
        if vals.name is not None:
            ylabel = vals.name
        vals = [vals.reindex(groups[name]) for name in order]
        if names is None:
            names = order

    else:

        # Handle case where the input data is an array or there was no groupby
        if hasattr(vals, 'shape'):
            if len(vals.shape) == 1:
                if np.isscalar(vals[0]):
                    vals = [vals]
                else:
                    vals = list(vals)
            elif len(vals.shape) == 2:
                nr, nc = vals.shape
                if nr == 1:
                    vals = [vals]
                elif nc == 1:
                    vals = [vals.ravel()]
                else:
                    vals = [vals[:, i] for i in range(nc)]
            else:
                error = "Input `vals` can have no more than 2 dimensions"
                raise ValueError(error)

        # This should catch things like flat lists
        elif np.isscalar(vals[0]):
            vals = [vals]

        # By default, just use the plot positions as names
        if names is None:
            names = list(range(1, len(vals) + 1))
        elif hasattr(names, "name"):
            if names.name is not None:
                xlabel = names.name

    # Now convert vals to a common representation
    # The plotting functions will work with a list of arrays
    # The list allows each array to possibly be of a different length
    vals = [np.asarray(a, np.float) for a in vals]

    return vals, xlabel, ylabel, names


def _box_colors(vals, color):
    """Find colors to use for boxplots or violinplots."""
    if color is None:
        colors = husl_palette(len(vals), l=.7)
    else:
        try:
            color = mpl.colors.colorConverter.to_rgb(color)
            colors = [color for _ in vals]
        except ValueError:
                colors = color_palette(color, len(vals))

    # Desaturate a bit because these are patches
    colors = [mpl.colors.colorConverter.to_rgb(c) for c in colors]
    colors = [desaturate(c, .7) for c in colors]

    # Determine the gray color for the lines
    light_vals = [colorsys.rgb_to_hls(*c)[1] for c in colors]
    l = min(light_vals) * .6
    gray = (l, l, l)

    return colors, gray


def boxplot(vals, groupby=None, names=None, join_rm=False, order=None,
            color=None, alpha=None, fliersize=3, linewidth=1.5, widths=.8,
            ax=None, **kwargs):
    """Wrapper for matplotlib boxplot with better aesthetics and functionality.

    Parameters
    ----------
    vals : DataFrame, Series, 2D array, list of vectors, or vector.
        Data for plot. DataFrames and 2D arrays are assuemd to be "wide" with
        each column mapping to a box. Lists of data are assumed to have one
        element per box.  Can also provide one long Series in conjunction with
        a grouping element as the `groupy` parameter to reshape the data into
        several boxes. Otherwise 1D data will produce a single box.
    groupby : grouping object
        If `vals` is a Series, this is used to group into boxes by calling
        pd.groupby(vals, groupby).
    names : list of strings, optional
        Names to plot on x axis; otherwise plots numbers. This will override
        names inferred from Pandas inputs.
    order : list of strings, optional
        If vals is a Pandas object with name information, you can control the
        order of the boxes by providing the box names in your preferred order.
    join_rm : boolean, optional
        If True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot.
    color : mpl color, sequence of colors, or seaborn palette name
        Inner box color.
    alpha : float
        Transparancy of the inner box color.
    fliersize : float, optional
        Markersize for the fliers.
    linewidth : float, optional
        Width for the box outlines and whiskers.
    ax : matplotlib axis, optional
        Existing axis to plot into, otherwise grab current axis.
    kwargs : additional keyword arguments to boxplot

    Returns
    -------
    ax : matplotlib axis
        Axis where boxplot is plotted.

    """
    if ax is None:
        ax = plt.gca()

    # Reshape and find labels for the plot
    vals, xlabel, ylabel, names = _box_reshape(vals, groupby, names, order)

    # Draw the boxplot using matplotlib
    boxes = ax.boxplot(vals, patch_artist=True, widths=widths, **kwargs)

    # Find plot colors
    colors, gray = _box_colors(vals, color)

    # Set the new aesthetics
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

    # Is this a vertical plot?
    vertical = kwargs.get("vert", True)

    # Draw the joined repeated measures
    if join_rm:
        x, y = np.arange(1, len(np.transpose(vals)) + 1), np.transpose(vals)
        if not vertical:
            x, y = y, x
        ax.plot(x, y, color=gray, alpha=2. / 3)

    # Label the axes and ticks
    if vertical:
        ax.set_xticklabels(names)
    else:
        ax.set_yticklabels(names)
        xlabel, ylabel = ylabel, xlabel
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Turn off the grid parallel to the boxes
    if vertical:
        ax.xaxis.grid(False)
    else:
        ax.yaxis.grid(False)

    return ax


def violin(*args, **kwargs):
    """Deprecated old name for violinplot. Please update your code."""
    warnings.warn("violin() is deprecated; please use violinplot()",
                  UserWarning)
    return violinplot(*args, **kwargs)


def violinplot(vals, groupby=None, inner="box", color=None, positions=None,
               names=None, order=None, kernel="gau", bw="scott", widths=.8,
               alpha=None, join_rm=False, gridsize=100, cut=3, inner_kws=None,
               ax=None, **kwargs):

    """Create a violin plot (a combination of boxplot and kernel density plot).

    Parameters
    ----------
    vals : DataFrame, Series, 2D array, or list of vectors.
        Data for plot. DataFrames and 2D arrays are assuemd to be "wide" with
        each column mapping to a box. Lists of data are assumed to have one
        element per box.  Can also provide one long Series in conjunction with
        a grouping element as the `groupy` parameter to reshape the data into
        several violins. Otherwise 1D data will produce a single violins.
    groupby : grouping object
        If `vals` is a Series, this is used to group into boxes by calling
        pd.groupby(vals, groupby).
    inner : box | sticks | points
        Plot quartiles or individual sample values inside violin.
    color : mpl color, sequence of colors, or seaborn palette name
        Inner violin colors
    positions : number or sequence of numbers
        Position of first violin or positions of each violin.
    names : list of strings, optional
        Names to plot on x axis; otherwise plots numbers. This will override
        names inferred from Pandas inputs.
    order : list of strings, optional
        If vals is a Pandas object with name information, you can control the
        order of the plot by providing the violin names in your preferred
        order.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }
        Code for shape of kernel to fit with.
    bw : {'scott' | 'silverman' | scalar}
        Name of reference method to determine kernel size, or size as a
        scalar.
    widths : float
        Width of each violin at maximum density.
    alpha : float, optional
        Transparancy of violin fill.
    join_rm : boolean, optional
        If True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot.
    gridsize : int
        Number of discrete gridpoints to evaluate the density on.
    cut : scalar
        Draw the estimate to cut * bw from the extreme data points.
    inner_kws : dict, optional
        Keyword arugments for inner plot.
    ax : matplotlib axis, optional
        Axis to plot on, otherwise grab current axis.
    kwargs : additional parameters to fill_betweenx

    Returns
    -------
    ax : matplotlib axis
        Axis with violin plot.

    """
    if ax is None:
        ax = plt.gca()

    # Reshape and find labels for the plot
    vals, xlabel, ylabel, names = _box_reshape(vals, groupby, names, order)

    # Sort out the plot colors
    colors, gray = _box_colors(vals, color)

    # Initialize the kwarg dict for the inner plot
    if inner_kws is None:
        inner_kws = {}
    in_alpha = inner_kws.pop("alpha", .6 if inner == "points" else 1)
    in_alpha *= 1 if alpha is None else alpha
    in_color = inner_kws.pop("color", gray)
    in_marker = inner_kws.pop("marker", ".")
    in_lw = inner_kws.pop("lw", 1.5 if inner == "box" else .8)

    # Find where the violins are going
    if positions is None:
        positions = np.arange(1, len(vals) + 1)
    elif not hasattr(positions, "__iter__"):
        positions = np.arange(positions, len(vals) + positions)

    # Set the default linewidth if not provided in kwargs
    try:
        lw = kwargs[({"lw", "linewidth"} & set(kwargs)).pop()]
    except KeyError:
        lw = 1.5

    # Iterate over the variables
    for i, a in enumerate(vals):

        # Fit the KDE
        x = positions[i]
        kde = sm.nonparametric.KDEUnivariate(a)
        fft = kernel == "gau"
        kde.fit(bw=bw, kernel=kernel, gridsize=gridsize, cut=cut, fft=fft)
        y, dens = kde.support, kde.density
        scl = 1 / (dens.max() / (widths / 2))
        dens *= scl

        # Draw the violin
        ax.fill_betweenx(y, x - dens, x + dens, alpha=alpha, color=colors[i])
        if inner == "box":
            for quant in moss.percentiles(a, [25, 75]):
                q_x = kde.evaluate(quant) * scl
                q_x = [x - q_x, x + q_x]
                ax.plot(q_x, [quant, quant], color=in_color,
                        linestyle=":", linewidth=in_lw, **inner_kws)
            med = np.median(a)
            m_x = kde.evaluate(med) * scl
            m_x = [x - m_x, x + m_x]
            ax.plot(m_x, [med, med], color=in_color,
                    linestyle="--", linewidth=in_lw, **inner_kws)
        elif inner == "stick":
            x_vals = kde.evaluate(a) * scl
            x_vals = [x - x_vals, x + x_vals]
            ax.plot(x_vals, [a, a], color=in_color,
                    linewidth=in_lw, alpha=in_alpha, **inner_kws)
        elif inner == "points":
            x_vals = [x for _ in a]
            ax.plot(x_vals, a, in_marker, color=in_color,
                    alpha=in_alpha, mew=0, **inner_kws)
        for side in [-1, 1]:
            ax.plot((side * dens) + x, y, c=gray, lw=lw)

    # Draw the repeated measure bridges
    if join_rm:
        ax.plot(range(1, len(vals) + 1), vals,
                color=in_color, alpha=2. / 3)

    # Add in semantic labels
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


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    h = 2 * moss.iqr(a) / (len(a) ** (1 / 3))
    return np.ceil((a.max() - a.min()) / h)


def distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, axlabel=None, ax=None):
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
    axlabel : string, False, or None
        Name for the support axis label. If None, will try to get it
        from a.namel if False, do not set a label.
    ax : matplotlib axis, optional
        if provided, plot on this axis

    Returns
    -------
    ax : matplotlib axis

    """
    if ax is None:
        ax = plt.gca()

    # Intelligently label the support axis
    label_ax = bool(axlabel)
    if axlabel is None and hasattr(a, "name"):
        axlabel = a.name
        if axlabel is not None:
            label_ax = True

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
            bins = _freedman_diaconis_bins(a)
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
        gridsize = fit_kws.pop("gridsize", 500)
        cut = fit_kws.pop("cut", 3)
        clip = fit_kws.pop("clip", (-np.inf, np.inf))
        bw = sm.nonparametric.bandwidths.bw_scott(a)
        x = _kde_support(a, bw, gridsize, cut, clip)
        params = fit.fit(a)
        pdf = lambda x: fit.pdf(x, *params)
        y = pdf(x)
        if vertical:
            x, y = y, x
        ax.plot(x, y, color=fit_color, **fit_kws)

    if label_ax:
        if vertical:
            ax.set_ylabel(axlabel)
        else:
            ax.set_xlabel(axlabel)

    return ax


def _univariate_kdeplot(data, shade, vertical, kernel, bw, gridsize, cut,
                        clip, legend, ax, **kwargs):
    """Plot a univariate kernel density estimate on one of the axes."""

    # Sort out the clipping
    if clip is None:
        clip = (-np.inf, np.inf)

    # Calculate the KDE
    try:
        # Prefer using statsmodels for kernel flexibility
        x, y = _statsmodels_univariate_kde(data, kernel, bw,
                                           gridsize, cut, clip)
    except ImportError:
        # Fall back to scipy if missing statsmodels
        if kernel != "gau":
            kernel = "gau"
            msg = "Kernel other than `gau` requires statsmodels."
            warnings.warn(msg, UserWarning)
        x, y = _scipy_univariate_kde(data, bw, gridsize, cut, clip)

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)

    # Flip the data if the plot should be on the y axis
    if vertical:
        x, y = y, x

    # Check if a label was specified in the call
    label = kwargs.pop("label", None)

    # Otherwise check if the data object has a name
    if label is None and hasattr(data, "name"):
        label = data.name

    # Decide if we're going to add a legend
    legend = not label is None and legend
    label = "_nolegend_" if label is None else label

    # Use the active color cycle to find the plot color
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    alpha = kwargs.pop("alpha", 0.25)
    if shade:
        ax.fill_between(x, 1e-12, y, color=color, alpha=alpha)

    # Draw the legend here
    if legend:
        ax.legend(loc="best")

    return ax


def _statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using statsmodels."""
    from statsmodels import nonparametric
    fft = kernel == "gau"
    kde = nonparametric.kde.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    grid, y = kde.support, kde.density
    return grid, y


def _scipy_univariate_kde(data, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using scipy."""
    kde = stats.gaussian_kde(data, bw_method=bw)
    if isinstance(bw, str):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)()
    grid = _kde_support(data, bw, gridsize, cut, clip)
    y = kde(grid)
    return grid, y


def _bivariate_kdeplot(x, y, filled, kernel, bw, gridsize, cut, clip, axlabel,
                       ax, **kwargs):
    """Plot a joint KDE estimate as a bivariate contour plot."""

    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    # Calculate the KDE
    try:
        xx, yy, z = _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
    except ImportError:
        xx, yy, z = _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)
    cmap = kwargs.pop("cmap", "BuGn" if filled else "BuGn_d")
    if isinstance(cmap, str):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
    contour_func = ax.contourf if filled else ax.contour
    contour_func(xx, yy, z, n_levels, cmap=cmap, **kwargs)

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    return ax


def _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels."""
    from statsmodels import nonparametric
    if isinstance(bw, str):
        bw_func = getattr(nonparametric.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]
    kde = nonparametric.kernel_density.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T)
    if isinstance(bw, str):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)()
    x_support = _kde_support(data[:, 0], bw, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def kdeplot(data, data2=None, shade=False, vertical=False, kernel="gau",
            bw="scott", gridsize=100, cut=3, clip=None, legend=True, ax=None,
            **kwargs):
    """Fit and plot a univariate or bivarate kernel density estimate.

    Parameters
    ----------
    data : 1d or 2d array-like
        Input data. If two-dimensional, assumed to be shaped (n_unit x n_var),
        and a bivariate contour plot will be drawn.
    data2: 1d array-like
        Second input data. If provided `data` must be one-dimensional, and
        a bivariate plot is produced.
    shade : bool, optional
        If true, shade in the area under the KDE curve (or draw with filled
        contours when data is bivariate).
    vertical : bool
        If True, density is on x-axis.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with. Bivariate KDE can only use
        gaussian kernel.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor,
        or scalar for each dimension of the bivariate plot.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    legend : bool, optoinal
        If True, add a legend or label the axes when possible.
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

    bivariate = False
    if isinstance(data, np.ndarray) and np.ndim(data) > 1:
        bivariate = True
        x, y = data.astype(np.float64).T
    elif isinstance(data, pd.DataFrame) and np.ndim(data) > 1:
        bivariate = True
        x = data.iloc[:, 0].values.astype(np.float64)
        y = data.iloc[:, 1].values.astype(np.float64)
    elif data2 is not None:
        bivariate = True
        x = data
        y = data2

    if bivariate:
        ax = _bivariate_kdeplot(x, y, shade, kernel, bw, gridsize,
                                cut, clip, legend, ax, **kwargs)
    else:
        ax = _univariate_kdeplot(data, shade, vertical, kernel, bw,
                                 gridsize, cut, clip, legend, ax, **kwargs)

    return ax


def rugplot(a, height=None, axis="x", ax=None, **kwargs):
    """Plot datapoints in an array as sticks on an axis.

    Parameters
    ----------
    a : vector
        1D array of datapoints.
    height : scalar, optional
        Height of ticks, if None draw at 5% of axis range.
    axis : {'x' | 'y'}, optional
        Axis to draw rugplot on.
    ax : matplotlib axis
        Axis to draw plot into; otherwise grabs current axis.
    kwargs : other keyword arguments for plt.plot()

    Returns
    -------
    ax : matplotlib axis
        Axis with rugplot.

    """
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
