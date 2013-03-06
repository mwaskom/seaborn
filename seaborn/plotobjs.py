"""High level plotting functions using matplotlib."""

# Except in strange circumstances, all functions in this module
# should take an ``ax`` keyword argument defaulting to None
# (which creates a new subplot) and an open-ended **kwargs to
# pass to the underlying matplotlib function being called.
# They should also return the ``ax`` object.

import numpy as np
from scipy import stats, interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import moss

from seaborn.utils import color_palette, ci_to_errsize


def tsplot(x, data, err_style=["ci_band"], ci=68, interpolate=True,
           estimator=np.mean, n_boot=10000, smooth=False,
           err_palette=None, ax=None, **kwargs):
    """Plot timeseries from a set of observations.

    Parameters
    ----------
    x : n_tp array
        x values
    data : n_obs x n_tp array
        array of timeseries data where first axis is e.g. subjects
    err_style : list of strings
        names of ways to plot uncertainty across observations from set of
       {ci_band, ci_bars, boot_traces, book_kde, obs_traces, obs_points}
    ci : int or list of ints
        confidence interaval size(s). if a list, it will stack the error
        plots for each confidence interval
    estimator : callable
        function to determine centralt tendency and to pass to bootstrap
        must take an ``axis`` argument
    n_boot : int
        number of bootstrap iterations
    smooth : boolean
        whether to perform a smooth bootstrap (resample from KDE)
    ax : axis object, optional
        plot in given axis; if None creates a new figure
    kwargs : further keyword arguments for main call to plot()

    Returns
    -------
    ax : matplotlib axis
        axis with plot data

    """
    if ax is None:
        ax = plt.subplot(111)

    # Bootstrap the data for confidence intervals
    boot_data = moss.bootstrap(data, n_boot=n_boot, smooth=smooth,
                               axis=0, func=estimator)
    ci_list = hasattr(ci, "__iter__")
    if not ci_list:
        ci = [ci]
    ci_vals = [(50 - w / 2, 50 + w / 2) for w in ci]
    cis = [moss.percentiles(boot_data, ci, axis=0) for ci in ci_vals]
    central_data = estimator(data, axis=0)

    # Plot the timeseries line to get its color
    line, = ax.plot(x, central_data, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)

    # Use subroutines to plot the uncertainty
    for style in err_style:

        # Grab the function from the global environment
        try:
            plot_func = globals()["_plot_%s" % style]
        except KeyError:
            raise ValueError("%s is not a valid err_style" % style)

        # Possibly set up to plot each observation in a different color
        if err_palette is not None and "obs" in style:
            orig_color = color
            color = color_palette(err_palette, len(data), desat=.99)

        plot_kwargs = dict(ax=ax, x=x, data=data,
                           boot_data=boot_data,
                           central_data=central_data,
                           color=color)

        for ci_i in cis:
            plot_kwargs["ci"] = ci_i
            plot_func(**plot_kwargs)

        if err_palette is not None and "obs" in style:
            color = orig_color
    # Replot the central trace so it is prominent
    marker = kwargs.pop("marker", "" if interpolate else "o")
    linestyle = kwargs.pop("linestyle", "-" if interpolate else "")
    ax.plot(x, central_data, color=color,
            marker=marker, linestyle=linestyle, **kwargs)

    return ax

# Subroutines for tsplot errorbar plotting
# ----------------------------------------


def _plot_ci_band(ax, x, ci, color, **kwargs):
    """Plot translucent error bands around the central tendancy."""
    low, high = ci
    ax.fill_between(x, low, high, color=color, alpha=0.2)


def _plot_ci_bars(ax, x, central_data, ci, color, **kwargs):
    """Plot error bars at each data point."""
    err = ci_to_errsize(ci, central_data)
    ax.errorbar(x, central_data, yerr=err, fmt=None, ecolor=color)


def _plot_boot_traces(ax, x, boot_data, color, **kwargs):
    """Plot 250 traces from bootstrap."""
    ax.plot(x, boot_data[:250].T, color=color, alpha=0.25, linewidth=0.25)


def _plot_obs_traces(ax, x, data, ci, color, **kwargs):
    """Plot a trace for each observation in the original data."""
    if isinstance(color, list):
        for i, obs in enumerate(data):
            ax.plot(x, obs, color=color[i], alpha=0.5)
    else:
        ax.plot(x, data.T, color=color, alpha=0.2)


def _plot_obs_points(ax, x, data, color, **kwargs):
    """Plot each original data point discretely."""
    if isinstance(color, list):
        for i, obs in enumerate(data):
            ax.plot(x, obs, "o", color=color[i], alpha=0.8, markersize=4)
    else:
        ax.plot(x, data.T, "o", color=color, alpha=0.5, markersize=4)


def _plot_boot_kde(ax, x, boot_data, color, **kwargs):
    """Plot the kernal density estimate of the bootstrap distribution."""
    kwargs.pop("data")
    _ts_kde(ax, x, boot_data, color, **kwargs)


def _plot_obs_kde(ax, x, data, color, **kwargs):
    """Plot the kernal density estimate over the sample."""
    _ts_kde(ax, x, data, color, **kwargs)


def _ts_kde(ax, x, data, color, **kwargs):
    """Upsample over time and plot a KDE of the bootstrap distribution."""
    kde_data = []
    y_min, y_max = moss.percentiles(data, [1, 99])
    y_vals = np.linspace(y_min, y_max, 100)
    upsampler = interpolate.interp1d(x, data)
    data_upsample = upsampler(np.linspace(x.min(), x.max(), 100))
    for pt_data in data_upsample.T:
        pt_kde = stats.kde.gaussian_kde(pt_data)
        kde_data.append(pt_kde(y_vals))
    kde_data = np.transpose(kde_data)
    rgb = mpl.colors.ColorConverter().to_rgb(color)
    img = np.zeros((kde_data.shape[0], kde_data.shape[1], 4))
    img[:, :, :3] = rgb
    kde_data /= kde_data.max(axis=0)
    kde_data[kde_data > 1] = 1
    img[:, :, 3] = kde_data
    ax.imshow(img, interpolation="spline16", zorder=1,
              extent=(x.min(), x.max(), y_min, y_max),
              aspect="auto", origin="lower")


def lmplot(x, y, data, color=None, row=None, col=None,
           x_mean=False, x_ci=95, fit_line=True, ci=95,
           sharex=True, sharey=True, palette="hls", size=None,
           scatter_kws=None, line_kws=None, palette_kws=None):
    """Plot a linear model from a DataFrame.

    Parameters
    ----------
    x, y : strings
        column names in `data` DataFrame for x and y variables
    data : DataFrame
        source of data for the model
    color : string, optional
        DataFrame column name to group the model by color
    row, col : strings, optional
        DataFrame column names to make separate plot facets
    x_mean, x_ci : bool, int optional
        if True, take the mean over each unique x value and
        plot as a point estimate with the given confidence interval
    fit_line : bool, optional
        if True fit a regression line by color/row/col and plot
    ci: int, optional
        confidence interval for the regression line
    sharex, sharey : bools, optional
        only relevant if faceting; passed to plt.subplots
    palette : seaborn color palette argument
        if using separate plots by color, draw with this color palette
    size : float, optional
        size (plots are square) for each plot facet
    {scatter, line}_kws : dictionary
        keyword arguments to pass to the underlying plot functions
    palette_kws : dictionary
        keyword arguments for seaborn.color_palette

    """
    # TODO
    # - position_{dodge, jitter}
    # - more general x-axis factor summary

    # First sort out the general figure layout
    if size is None:
        size = mpl.rcParams["figure.figsize"][1]

    nrow = 1 if row is None else len(data[row].unique())
    ncol = 1 if col is None else len(data[col].unique())

    f, axes = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey,
                           figsize=(size * ncol, size * nrow))
    axes = np.atleast_2d(axes).reshape(nrow, ncol)

    if nrow == 1:
        row_masks = [np.repeat(True, len(data))]
    else:
        row_vals = np.sort(data[row].unique())
        row_masks = [data[row] == val for val in row_vals]

    if ncol == 1:
        col_masks = [np.repeat(True, len(data))]
    else:
        col_vals = np.sort(data[col].unique())
        col_masks = [data[col] == val for val in col_vals]

    if palette_kws is None:
        palette_kws = {}

    # Sort out the plot colors
    color_factor = color
    if color is None:
        hue_masks = [np.repeat(True, len(data))]
        colors = ["#222222"]
    else:
        hue_vals = np.sort(data[color].unique())
        hue_masks = [data[color] == val for val in hue_vals]
        colors = color_palette(palette, len(hue_masks), **palette_kws)

    # Default keyword arguments for plot components
    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}

    # First walk through the facets and plot the scatters
    for row_i, row_mask in enumerate(row_masks):
        for col_j, col_mask in enumerate(col_masks):
            ax = axes[row_i, col_j]
            ax.set_xlabel(x)
            ax.set_ylabel(y)

            # Title the plot if we are faceting
            title = ""
            if row is not None:
                title += "%s = %s" % (row, row_vals[row_i])
            if row is not None and col is not None:
                title += " | "
            if col is not None:
                title += "%s = %s" % (col, col_vals[col_j])
            ax.set_title(title)

            for hue_k, hue_mask in enumerate(hue_masks):
                color = colors[hue_k]
                data_ijk = data[row_mask & col_mask & hue_mask]

                if x_mean:
                    ms = scatter_kws.pop("ms", 7)
                    mew = scatter_kws.pop("mew", 0)
                    x_vals = data_ijk[x].unique()
                    y_grouped = [np.array(data_ijk[y][data_ijk[x] == v])
                                 for v in x_vals]
                    y_mean = [np.mean(y_i) for y_i in y_grouped]
                    y_boots = [moss.bootstrap(np.array(y_i))
                               for y_i in y_grouped]
                    ci_lims = [50 - x_ci / 2., 50 + x_ci / 2.]
                    y_ci = [moss.percentiles(y_i, ci_lims) for y_i in y_boots]
                    y_error = ci_to_errsize(np.transpose(y_ci), y_mean)

                    ax.plot(x_vals, y_mean, "o", mew=mew, ms=ms,
                            color=color, **scatter_kws)
                    ax.errorbar(x_vals, y_mean, y_error,
                                fmt=None, ecolor=color)
                else:
                    ms = scatter_kws.pop("ms", 4)
                    mew = scatter_kws.pop("mew", 0)
                    ax.plot(data_ijk[x], data_ijk[y], "o",
                            color=color, mew=mew, ms=ms, **scatter_kws)

    for ax_i in np.ravel(axes):
        ax_i.set_xmargin(.15)
        ax_i.autoscale_view()

    # Now walk through again and plot the regression estimate
    # and a confidence interval for the regression line
    if fit_line:
        for row_i, row_mask in enumerate(row_masks):
            for col_j, col_mask in enumerate(col_masks):
                ax = axes[row_i, col_j]
                xlim = ax.get_xlim()
                xx = np.linspace(xlim[0], xlim[1], 100)

                # Inner function to bootstrap the regression
                def _bootstrap_reg(x, y):
                    fit = np.polyfit(x, y, 1)
                    return np.polyval(fit, xx)

                for hue_k, hue_mask in enumerate(hue_masks):
                    color = colors[hue_k]
                    data_ijk = data[row_mask & col_mask & hue_mask]
                    x_vals = np.array(data_ijk[x])
                    y_vals = np.array(data_ijk[y])

                    # Regression line confidence interval
                    if ci is not None:
                        ci_lims = [50 - ci / 2., 50 + ci / 2.]
                        boots = moss.bootstrap(x_vals, y_vals,
                                               func=_bootstrap_reg)
                        ci_band = moss.percentiles(boots, ci_lims, axis=0)
                        ax.fill_between(xx, *ci_band, color=color, alpha=.15)

                    fit = np.polyfit(x_vals, y_vals, 1)
                    reg = np.polyval(fit, xx)
                    if color_factor is None:
                        label = ""
                    else:
                        label = hue_vals[hue_k]
                    ax.plot(xx, reg, color=color,
                            label=str(label), **line_kws)
                    ax.set_xlim(xlim)

    # Plot the legend on the upper left facet and adjust the layout
    if color_factor is not None:
        axes[0, 0].legend(loc="best", title=color_factor)
    plt.tight_layout()


def regplot(x, y, data=None, corr_func=stats.pearsonr, xlabel="", ylabel="",
            ci=95, size=None, annotloc=None, color=None, reg_kws=None,
            scatter_kws=None, dist_kws=None, text_kws=None):
    """Scatterplot with regreesion line, marginals, and correlation value.

    Parameters
    ----------
    x : sequence
        independent variables
    y : sequence
        dependent variables
    data : dataframe, optional
        if dataframe is given, x, and y are interpreted as
        string keys mapping to dataframe column names
    corr_func : callable, optional
        correlation function; expected to take two arrays
        and return a (statistic, pval) tuple
    xlabel, ylabel : string, optional
        label names
    ci : int or None
        confidence interval for the regression line
    size: int
        figure size (will be a square; only need one int)
    annotloc : two or three tuple
        (xpos, ypos [, horizontalalignment])
    color : matplotlib color scheme
        color of everything but the regression line
        overridden by passing `color` to subfunc kwargs
    {reg, scatter, dist, text}_kws: dicts
        further keyword arguments for the constituent plots


    """
    # Interperet inputs
    if data is not None:
        xlabel, ylabel = x, y
        x = np.array(data[x])
        y = np.array(data[y])

    # Set up the figure and axes
    size = 6 if size is None else size
    fig = plt.figure(figsize=(size, size))
    ax_scatter = fig.add_axes([0.05, 0.05, 0.75, 0.75])
    ax_x_marg = fig.add_axes([0.05, 0.82, 0.75, 0.13])
    ax_y_marg = fig.add_axes([0.82, 0.05, 0.13, 0.75])

    # Plot the scatter
    if scatter_kws is None:
        scatter_kws = {}
    if color is not None and "color" not in scatter_kws:
        scatter_kws.update(color=color)
    marker = scatter_kws.pop("markerstyle", "o")
    alpha_maker = stats.norm(0, 100)
    alpha = alpha_maker.pdf(len(x)) / alpha_maker.pdf(0)
    alpha = max(alpha, .1)
    alpha = scatter_kws.pop("alpha", alpha)
    ax_scatter.plot(x, y, marker, alpha=alpha, mew=0, **scatter_kws)
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)

    # Marginal plots using our distplot function
    if dist_kws is None:
        dist_kws = {}
    if color is not None and "color" not in dist_kws:
        dist_kws.update(color=color)
    if "legend" not in dist_kws:
        dist_kws["legend"] = False
    distplot(x, ax=ax_x_marg, **dist_kws)
    distplot(y, ax=ax_y_marg, vertical=True, **dist_kws)
    for ax in [ax_x_marg, ax_y_marg]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax_x_marg.set_xlim(ax_scatter.get_xlim())
    ax_x_marg.set_yticks([])
    ax_y_marg.set_ylim(ax_scatter.get_ylim())
    ax_y_marg.set_xticks([])

    # Regression line plot
    xlim = ax_scatter.get_xlim()
    a, b = np.polyfit(x, y, 1)
    if reg_kws is None:
        reg_kws = {}
    reg_color = reg_kws.pop("color", "#222222")
    ax_scatter.plot(xlim, np.polyval([a, b], xlim),
                    color=reg_color, **reg_kws)

    # Bootstrapped regression standard error
    if ci is not None:
        xx = np.linspace(xlim[0], xlim[1], 100)

        def _bootstrap_reg(x, y):
            fit = np.polyfit(x, y, 1)
            return np.polyval(fit, xx)

        boots = moss.bootstrap(x, y, func=_bootstrap_reg)
        ci_lims = [50 - ci / 2., 50 + ci / 2.]
        ci_band = moss.percentiles(boots, ci_lims, axis=0)
        ax_scatter.fill_between(xx, *ci_band, color=reg_color, alpha=.15)
        ax_scatter.set_xlim(xlim)

    # Calcluate a correlation statistic and p value
    r, p = corr_func(x, y)
    msg = "%s: %.3f (p=%.3g%s)" % (corr_func.__name__, r, p, moss.sig_stars(p))
    if annotloc is None:
        xmin, xmax = xlim
        x_range = xmax - xmin
        if r < 0:
            xloc, align = xmax - x_range * .02, "right"
        else:
            xloc, align = xmin + x_range * .02, "left"
        ymin, ymax = ax_scatter.get_ylim()
        y_range = ymax - ymin
        yloc = ymax - y_range * .02
    else:
        if len(annotloc) == 3:
            xloc, yloc, align = annotloc
        else:
            xloc, yloc = annotloc
            align = "left"
    if text_kws is None:
        text_kws = {}
    ax_scatter.text(xloc, yloc, msg, ha=align, va="top", **text_kws)


def boxplot(vals, join_rm=False, names=None, color=None, ax=None,
            **kwargs):
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
    if ax is None:
        ax = plt.subplot(111)
    if color is None:
        pos = kwargs.get("positions", [1])[0]
        line, = ax.plot(pos, np.mean(vals[0]), **kwargs)
        color = line.get_color()
        line.remove()
        kwargs.pop("color", None)

    widths = kwargs.pop("widths", .5)
    boxes = ax.boxplot(vals, patch_artist=True, widths=widths, **kwargs)

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
        ax.plot(range(1, len(vals) + 1), vals,
                color=color, alpha=2. / 3)

    if names is not None:
        if len(vals) != len(names):
            raise ValueError("Length of names list must match nuber of bins")
        ax.set_xticklabels(names)

    return ax


def distplot(a, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, legend=True, ax=None):
    """Flexibly plot a distribution of observations.

    Parameters
    ----------
    a : (squeezable to) 1d array
        observed data
    hist : bool, default True
        whether to plot a (normed) histogram
    kde : bool, defualt True
        whether to plot a gaussian kernel density estimate
    rug : bool, default False
        whether to draw a rugplot on the support axis
    fit : random variable object
        object with `fit` method returning a tuple that can be
        passed to a `pdf` method a positional arguments following
        an array of values to evaluate the pdf at
    {hist, kde, rug, fit}_kws : dictionaries
        keyword arguments for underlying plotting functions
    color : matplotlib color, optional
        color to plot everything but the fitted curve in
    vertical : bool, default False
        if True, oberved values are on y-axis
    legend : bool, default True
        if True, add a legend to the plot
    ax : matplotlib axis, optional
        if provided, plot on this axis

    Returns
    -------
    ax : matplotlib axis

    """
    if ax is None:
        ax = plt.subplot(111)
    a = np.asarray(a).squeeze()

    if hist_kws is None:
        hist_kws = dict()
    if kde_kws is None:
        kde_kws = dict()
    if rug_kws is None:
        rug_kws = dict()
    if fit_kws is None:
        fit_kws = dict()

    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()

    if hist:
        nbins = hist_kws.pop("nbins", 20)
        hist_alpha = hist_kws.pop("alpha", 0.4)
        orientation = "horizontal" if vertical else "vertical"
        hist_color = hist_kws.pop("color", color)
        ax.hist(a, nbins, normed=True, color=hist_color, alpha=hist_alpha,
                orientation=orientation, **hist_kws)

    if kde:
        kde_color = kde_kws.pop("color", color)
        kde_kws["label"] = "kde"
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
        fit_kws["label"] = fit.name + " fit"
        ax.plot(x, y, color=fit_color, **fit_kws)

    if legend:
        ax.legend(loc="best")

    return ax


def kdeplot(a, npts=1000, shade=False, support_thresh=1e-4,
            support_min=-np.inf, support_max=np.inf,
            vertical=False, ax=None, **kwargs):
    """Calculate and plot kernel density estimate.

    Parameters
    ----------
    a : ndarray
        input data
    npts : int, optional
        number of x points
    shade : bool, optional
        whether to shade under kde curve
    support_thresh : float, default 1e-4
        draw density for values up to support_thresh * max(density)
    support_{min, max}: float, default to (-) inf
        if given, do not draw above or below these values
        (does not affect the actual estimation)
    vertical : bool, defualt False
        if True, density is on x-axis
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
    x = _kde_support(a, kde, npts, support_thresh)
    x = x[x >= support_min]
    x = x[x <= support_max]
    y = kde(x)
    if vertical:
        y, x = x, y

    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)

    ax.plot(x, y, color=color, **kwargs)
    if shade:
        ax.fill_between(x, 0, y, color=color, alpha=0.25)
    return ax


def rugplot(a, height=None, axis="x", ax=None, **kwargs):
    """Plot datapoints in an array as sticks on an axis."""
    if ax is None:
        ax = plt.subplot(111)
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


def violin(vals, inner="box", position=None, widths=.5, join_rm=False,
           names=None, ax=None, **kwargs):
    """Create a violin plot (a combination of boxplot and KDE plot.

    Parameters
    ----------
    vals : array or sequence of arrays
        data to plot
    inner : box | sticks | points
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
    ax : matplotlib axis, optional
        axis to plot on, otherwise creates new one

    Returns
    -------
    ax : matplotlib axis
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

    line, = ax.plot(vals[0].mean(), vals[0].mean(), **kwargs)
    color = line.get_color()
    line.remove()

    gray = "#555555"

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
        elif inner == "points":
            x_vals = [x for i in a]
            ax.plot(x_vals, a, "o", color=gray, alpha=.3)
        for side in [-1, 1]:
            ax.plot((side * dens) + x, y, gray, linewidth=1)

    if join_rm:
        ax.plot(range(1, len(vals) + 1), vals,
                color=color, alpha=2. / 3)

    ax.set_xticks(position)
    if names is not None:
        if len(vals) != len(names):
            raise ValueError("Length of names list must match nuber of bins")
        ax.set_xticklabels(names)
    ax.set_xlim(position[0] - .5, position[-1] + .5)

    return ax


def corrplot(data, names=None, sig_stars=True, sig_tail="both", sig_corr=True,
             cmap="Spectral_r", cmap_range=None, cbar=True, ax=None, **kwargs):
    """Plot a correlation matrix with colormap and r values.

    Parameters
    ----------
    data : nvars x nobs array
        data array where rows are variables and columns are observations
    names : sequence of strings
        names to associate with variables; should be short
    sig_stars : bool
        if True, get significance with permutation test and denote with stars
    sig_tail : both | upper | lower
        direction for significance test
    sig_corr : bool
        if True, use FWE-corrected significance
    cmap : colormap
        colormap name as string or colormap object
    cmap_range : None, "full", (low, high)
        either truncate colormap at (-max(abs(r)), max(abs(r))), use the
        full range (-1, 1), or specify (min, max) values for the colormap
    cbar : boolean
        if true, plots the colorbar legend
    kwargs : other keyword arguments
        passed to ax.matshow()

    Returns
    -------
    ax : matplotlib axis
        axis object with plot

    """
    corrmat = np.corrcoef(data)

    if sig_stars:
        p_mat = moss.randomize_corrmat(data, sig_tail, sig_corr)
    else:
        p_mat = None

    if cmap_range is None:
        triu = np.triu_indices(len(data), 1)
        vmax = min(1, np.max(np.abs(corrmat[triu])) * 1.15)
        vmin = -vmax
        cmap_range = vmin, vmax
    elif cmap_range == "full":
        cmap_range = (-1, 1)

    ax = symmatplot(corrmat, p_mat, names, cmap, cmap_range,
                    cbar, ax, **kwargs)

    return ax


def symmatplot(mat, p_mat=None, names=None, cmap="Spectral_r", cmap_range=None,
               cbar=True, ax=None, **kwargs):
    """Plot a symettric matrix with colormap and statistic values."""
    if ax is None:
        ax = plt.subplot(111)

    nvars = len(mat)
    plotmat = mat.copy()
    plotmat[np.triu_indices(nvars)] = np.nan

    if cmap_range is None:
        vmax = np.nanmax(plotmat) * 1.15
        vmin = np.nanmin(plotmat) * 1.15
    elif len(cmap_range) == 2:
        vmin, vmax = cmap_range
    else:
        raise ValueError("cmap_range argument not understood")

    mat_img = ax.matshow(plotmat, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if cbar:
        plt.colorbar(mat_img)

    if p_mat is None:
        p_mat = np.ones((nvars, nvars))

    for i, j in zip(*np.triu_indices(nvars, 1)):
        val = mat[i, j]
        stars = moss.sig_stars(p_mat[i, j])
        ax.text(j, i, "\n%.3g\n%s" % (val, stars),
                fontdict=dict(ha="center", va="center"))

    if names is None:
        names = ["var%d" % i for i in range(nvars)]
    for i, name in enumerate(names):
        ax.text(i, i, name, fontdict=dict(ha="center", va="center",
                                          weight="bold"))

    ticks = np.linspace(.5, nvars - .5, nvars)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.grid(True, linestyle="-")

    return ax


def _kde_support(a, kde, npts, thresh=1e-4):
    """Establish support for a kernel density estimate."""
    min = a.min()
    max = a.max()
    range = max - min
    x = np.linspace(min - range, max + range, npts * 2)
    y = kde(x)
    mask = y > y.max() * thresh
    return x[mask]
