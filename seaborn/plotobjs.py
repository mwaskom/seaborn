"""High level plotting functions using matplotlib."""

# There are two different kind of functions here. Some are very high level and
# expect to have full reign of their figure to draw subplot grids or marginals.
# Others are constrained to a single axis and should take and return an `ax`
# argument. When `ax` is None, these functions should grab the current axis and
# plot into that. These two types of functions may be split into different
# modules to make the division a little bit more obvious.

from __future__ import division
import colorsys
import itertools
import numpy as np
import pandas as pd
from scipy import stats, interpolate
import statsmodels.api as sm
import statsmodels.formula.api as sf
import matplotlib as mpl
import matplotlib.pyplot as plt
import moss

from seaborn.utils import (color_palette, ci_to_errsize,
                           husl_palette, desaturate, _kde_support)


def tsplot(data, time=None, unit=None, condition=None, value=None,
           err_style="ci_band", ci=68, interpolate=True, color=None,
           estimator=np.mean, n_boot=5000, err_palette=None, err_kws=None,
           legend=True, ax=None, **kwargs):
    """Plot one or more timeseries with flexible representation of uncertainty.

    This function can take data specified either as a long-form (tidy)
    DataFrame or as an ndarray with dimensions for sampling unit, time, and
    (optionally) condition. The interpretation of some of the other parameters
    changes depending on the type of object passed as data.

    Parameters
    ----------
    data : DataFrame or ndarray
        Data for the plot. Should either be a "long form" dataframe or an
        array with dimensions (unit, time, condition). In both cases, the
        condition field/dimension is optional. The type of this argument
        determines the interpretation of the next few parameters.
    time : string or series-like
        Either the name of the field corresponding to time in the data
        DataFrame or x values for a plot when data is an array. If a Series,
        the name will be used to label the x axis.
    value : string
        Either the name of the field corresponding to the data values in
        the data DataFrame (i.e. the y coordinate) or a string that forms
        the y axis label when data is an array.
    unit : string
        Field in the data DataFrame identifying the sampling unit (e.g.
        subject, neuron, etc.). The error representation will collapse over
        units at each time/condition observation. This has no role when data
        is an array.
    condition : string or Series-like
        Either the name of the field identifying the condition an observation
        falls under in the data DataFrame, or a sequence of names with a length
        equal to the size of the third dimension of data. There will be a
        separate trace plotted for each condition. If condition is a Series
        with a name attribute, the name will form the title for the plot
        legend (unless legend is set to False).
    err_style : string or list of strings or None
        Names of ways to plot uncertainty across units from set of
       {ci_band, ci_bars, boot_traces, book_kde, unit_traces, unit_points}.
       Can use one or more than one method.
    ci : float or list of floats in [0, 100]
        Confidence interaval size(s). If a list, it will stack the error
        plots for each confidence interval. Only relevant for error styles
        with "ci" in the name.
    interpolate : boolean
        Whether to do a linear interpolation between each timepoint when
        plotting. The value of this parameter also determines the marker
        used for the main plot traces, unless marker is specified as a keyword
        argument.
    color : seaborn palette or matplotlib color name or dictionary
        Palette or color for the main plots and error representation (unless
        plotting by unit, which can be separately controlled with err_palette).
        If a dictionary, should map condition name to color spec.
    estimator : callable
        Function to determine central tendency and to pass to bootstrap
        must take an ``axis`` argument.
    n_boot : int
        Number of bootstrap iterations.
    err_palette: seaborn palette
        Palette name or list of colors used when plotting data for each unit.
    err_kws : dict, optional
        Keyword argument dictionary passed through to matplotlib function
        generating the error plot,
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    kwargs :
        Other keyword arguments are passed to main plot() call

    Returns
    -------
    ax : matplotlib axis
        axis with plot data

    """
    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    if err_kws is None:
        err_kws = {}

    # Handle case where data is an array
    if isinstance(data, pd.DataFrame):

        xlabel = time
        ylabel = value

        # Condition is optional
        if condition is None:
            condition = pd.Series(np.ones(len(data)))
            legend = False
            legend_name = None
            n_cond = 1
        else:
            legend = True and legend
            legend_name = condition
            n_cond = len(data[condition].unique())

    else:
        data = np.asarray(data)

        # Data can be a timecourse from a single unit or
        # several observations in one condition
        if data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]
        elif data.ndim == 2:
            data = data[:, :, np.newaxis]
        n_unit, n_time, n_cond = data.shape

        # Units are experimental observations. Maybe subjects, or neurons
        if unit is None:
            units = np.arange(n_unit)
        unit = "unit"
        units = np.repeat(units, n_time * n_cond)
        ylabel = None

        # Time forms the xaxis of the plot
        if time is None:
            times = np.arange(n_time)
        else:
            times = np.asarray(time)
        xlabel = None
        if hasattr(time, "name"):
            xlabel = time.name
        time = "time"
        times = np.tile(np.repeat(times, n_cond), n_unit)

        # Conditions split the timeseries plots
        if condition is None:
            conds = range(n_cond)
            legend = False
            if isinstance(color, dict):
                raise ValueError("Must have condition names if using color dict.")
        else:
            conds = np.asarray(condition)
            legend = True and legend
            if hasattr(condition, "name"):
                legend_name = condition.name
            else:
                legend_name = None
        condition = "cond"
        conds = np.tile(conds, n_unit * n_time)

        # Value forms the y value in the plot
        if value is None:
            ylabel = None
        else:
            ylabel = value
        value = "value"

        # Convert to long-form DataFrame
        data = pd.DataFrame(dict(value=data.ravel(),
                                 time=times,
                                 unit=units,
                                 cond=conds))

    # Set up the err_style and ci arguments for teh loop below
    if not hasattr(err_style, "__iter__"):
        err_style = [err_style]
    elif err_style is None:
        err_style = []
    if not hasattr(ci, "__iter__"):
        ci = [ci]

    # Set up the color palette
    if color is None:
        colors = color_palette()
    elif isinstance(color, dict):
        colors = [color[c] for c in data[condition].unique()]
    else:
        try:
            colors = color_palette(color, n_cond)
        except ValueError:
            color = mpl.colors.colorConverter.to_rgb(color)
            colors = [color] * n_cond

    # Do a groupby with condition and plot each trace
    for c, (cond, df_c) in enumerate(data.groupby(condition, sort=False)):

        df_c = df_c.pivot(unit, time, value)
        x = df_c.columns.values.astype(np.float)

        # Bootstrap the data for confidence intervals
        boot_data = moss.bootstrap(df_c.values, n_boot=n_boot,
                                   axis=0, func=estimator)
        cis = [moss.ci(boot_data, v, axis=0) for v in ci]
        central_data = estimator(df_c.values, axis=0)

        # Get the color for this condition
        color = colors[c]

        # Use subroutines to plot the uncertainty
        for style in err_style:

            # Allow for null style (only plot central tendency)
            if style is None:
                continue

            # Grab the function from the global environment
            try:
                plot_func = globals()["_plot_%s" % style]
            except KeyError:
                raise ValueError("%s is not a valid err_style" % style)

            # Possibly set up to plot each observation in a different color
            if err_palette is not None and "unit" in style:
                orig_color = color
                color = color_palette(err_palette, len(df_c.values))

            # Pass all parameters to the error plotter as keyword args
            plot_kwargs = dict(ax=ax, x=x, data=df_c.values,
                               boot_data=boot_data,
                               central_data=central_data,
                               color=color, err_kws=err_kws)

            # Plot the error representation, possibly for multiple cis
            for ci_i in cis:
                plot_kwargs["ci"] = ci_i
                plot_func(**plot_kwargs)

            if err_palette is not None and "unit" in style:
                color = orig_color

        # Plot the central trace
        marker = kwargs.pop("marker", "" if interpolate else "o")
        linestyle = kwargs.pop("linestyle", "-" if interpolate else "")
        label = kwargs.pop("label", cond if legend else "_nolegend_")
        ax.plot(x, central_data, color=color, label=label,
                marker=marker, linestyle=linestyle,  **kwargs)

    # Pad the sides of the plot only when not interpolating
    ax.set_xlim(x.min(), x.max())
    x_diff = x[1] - x[0]
    if not interpolate:
        ax.set_xlim(x.min() - x_diff, x.max() + x_diff)

    # Add the plot labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=0, title=legend_name)

    return ax

# Subroutines for tsplot errorbar plotting
# ----------------------------------------


def _plot_ci_band(ax, x, ci, color, err_kws, **kwargs):
    """Plot translucent error bands around the central tendancy."""
    low, high = ci
    if "alpha" not in err_kws:
        err_kws["alpha"] = 0.2
    ax.fill_between(x, low, high, color=color, **err_kws)


def _plot_ci_bars(ax, x, central_data, ci, color, err_kws, **kwargs):
    """Plot error bars at each data point."""
    for x_i, y_i, (low, high) in zip(x, central_data, ci.T):
        ax.plot([x_i, x_i], [low, high], color=color,
                solid_capstyle="round", **err_kws)


def _plot_boot_traces(ax, x, boot_data, color, err_kws, **kwargs):
    """Plot 250 traces from bootstrap."""
    if "alpha" not in err_kws:
        err_kws["alpha"] = 0.25
    if "lw" in err_kws:
        err_kws["linewidth"] = err_kws.pop("lw")
    if "linewidth" not in err_kws:
        err_kws["linewidth"] = 0.25
    ax.plot(x, boot_data.T, color=color, label="_nolegend_", **err_kws)


def _plot_unit_traces(ax, x, data, ci, color, err_kws, **kwargs):
    """Plot a trace for each observation in the original data."""
    if isinstance(color, list):
        if "alpha" not in err_kws:
            err_kws["alpha"] = .5
        for i, obs in enumerate(data):
            ax.plot(x, obs, color=color[i], label="_nolegend_", **err_kws)
    else:
        if "alpha" not in err_kws:
            err_kws["alpha"] = .2
        ax.plot(x, data.T, color=color, label="_nolegend_", **err_kws)


def _plot_unit_points(ax, x, data, color, err_kws, **kwargs):
    """Plot each original data point discretely."""
    if isinstance(color, list):
        for i, obs in enumerate(data):
            ax.plot(x, obs, "o", color=color[i], alpha=0.8, markersize=4,
                    label="_nolegend_", **err_kws)
    else:
        ax.plot(x, data.T, "o", color=color, alpha=0.5, markersize=4,
                label="_nolegend_", **err_kws)


def _plot_boot_kde(ax, x, boot_data, color, **kwargs):
    """Plot the kernal density estimate of the bootstrap distribution."""
    kwargs.pop("data")
    _ts_kde(ax, x, boot_data, color, **kwargs)


def _plot_unit_kde(ax, x, data, color, **kwargs):
    """Plot the kernal density estimate over the sample."""
    _ts_kde(ax, x, data, color, **kwargs)


def _ts_kde(ax, x, data, color, **kwargs):
    """Upsample over time and plot a KDE of the bootstrap distribution."""
    kde_data = []
    y_min, y_max = data.min(), data.max()
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
    ax.imshow(img, interpolation="spline16", zorder=2,
              extent=(x.min(), x.max(), y_min, y_max),
              aspect="auto", origin="lower")


def lmplot(x, y, data, color=None, row=None, col=None, col_wrap=None,
           x_estimator=None, x_ci=95, n_boot=5000, fit_reg=True,
           order=1, ci=95, logistic=False, truncate=False,
           x_partial=None, y_partial=None, x_jitter=None, y_jitter=None,
           sharex=True, sharey=True, palette="husl", size=None,
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
    col_wrap : int, optional
        wrap col variable at this width - cannot be used with row facet
    x_estimator : callable, optional
        Interpret X values as factor labels and use this function
        to plot the point estimate and bootstrapped CI
    x_ci : int optional
        size of confidence interval for x_estimator error bars
    n_boot : int, optional
        number of bootstrap iterations to perform
    fit_reg : bool, optional
        if True fit a regression model by color/row/col and plot
    order : int, optional
        order of the regression polynomial to fit (default = 1)
    ci : int, optional
        confidence interval for the regression line
    logistic : bool, optional
        fit the regression line with logistic regression
    truncate : bool, optional
        if True, only fit line from data min to data max
    {x, y}_partial : string or list of strings, optional
        regress these variables out of the factors before plotting
    {x, y}_jitter : float, optional
        parameters for uniformly distributed random noise added to positions
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
    # - legend when fit_line is False
    # - wrap title when wide

    # First sort out the general figure layout
    if size is None:
        size = mpl.rcParams["figure.figsize"][1]

    if col is None and col_wrap is not None:
        raise ValueError("Need column facet variable for `col_wrap`")
    if row is not None and col_wrap is not None:
        raise ValueError("Cannot facet rows when using `col_wrap`")

    nrow = 1 if row is None else len(data[row].unique())
    ncol = 1 if col is None else len(data[col].unique())

    if col_wrap is not None:
        ncol = col_wrap
        nrow = int(np.ceil(len(data[col].unique()) / col_wrap))

    f, axes = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey,
                           figsize=(size * ncol, size * nrow))
    axes = np.atleast_2d(axes).reshape(nrow, ncol)

    if nrow == 1 or col_wrap is not None:
        row_masks = [np.repeat(True, len(data))]
    else:
        row_vals = np.sort(data[row].unique())
        row_masks = [data[row] == val for val in row_vals]

    if ncol == 1:
        col_masks = [np.repeat(True, len(data))]
    else:
        col_vals = np.sort(data[col].unique())
        col_masks = [data[col] == val for val in col_vals]

    if x_partial is not None:
        if not isinstance(x_partial, list):
            x_partial = [x_partial]
    if y_partial is not None:
        if not isinstance(y_partial, list):
            y_partial = [y_partial]

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
    scatter_ms = scatter_kws.pop("ms", 4)
    scatter_mew = mew = scatter_kws.pop("mew", 0)
    scatter_alpha = mew = scatter_kws.pop("alpha", .77)
    for row_i, row_mask in enumerate(row_masks):
        for col_j, col_mask in enumerate(col_masks):
            if col_wrap is not None:
                f_row = col_j // ncol
                f_col = col_j % ncol
            else:
                f_row, f_col = row_i, col_j
            ax = axes[f_row, f_col]
            if f_row + 1 == nrow:
                ax.set_xlabel(x)
            if f_col == 0:
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

                if x_estimator is not None:
                    ms = scatter_kws.pop("ms", 7)
                    mew = scatter_kws.pop("mew", 0)
                    x_vals = data_ijk[x].unique()
                    y_vals = data_ijk[y]

                    if y_partial is not None:
                        for var in y_partial:
                            conf = data_ijk[var]
                            conf -= conf.mean()
                            y_mean = y_vals.mean()
                            y_vals = moss.vector_reject(y_vals - y_mean, conf)
                            y_vals += y_mean

                    y_grouped = [np.array(y_vals[data_ijk[x] == v])
                                 for v in x_vals]

                    y_est = [x_estimator(y_i) for y_i in y_grouped]
                    y_boots = [moss.bootstrap(np.array(y_i),
                                              func=x_estimator,
                                              n_boot=n_boot)
                               for y_i in y_grouped]
                    ci_lims = [50 - x_ci / 2., 50 + x_ci / 2.]
                    y_ci = [moss.percentiles(y_i, ci_lims) for y_i in y_boots]
                    y_error = ci_to_errsize(np.transpose(y_ci), y_est)

                    ax.plot(x_vals, y_est, "o", mew=mew, ms=ms,
                            color=color, **scatter_kws)
                    ax.errorbar(x_vals, y_est, y_error,
                                fmt=None, ecolor=color)
                else:
                    x_ = data_ijk[x]
                    y_ = data_ijk[y]

                    if x_partial is not None:
                        for var in x_partial:
                            conf = data_ijk[var]
                            conf -= conf.mean()
                            x_mean = x_.mean()
                            x_ = moss.vector_reject(x_ - x_mean, conf)
                            x_ += x_mean
                    if y_partial is not None:
                        for var in y_partial:
                            conf = data_ijk[var]
                            conf -= conf.mean()
                            y_mean = y_.mean()
                            y_ = moss.vector_reject(y_ - y_mean, conf)
                            y_ += y_mean

                    if x_jitter is not None:
                        x_ += np.random.uniform(-x_jitter, x_jitter, x_.shape)
                    if y_jitter is not None:
                        y_ += np.random.uniform(-y_jitter, y_jitter, y_.shape)
                    ax.plot(x_, y_, "o", color=color, alpha=scatter_alpha,
                            mew=scatter_mew, ms=scatter_ms, **scatter_kws)

    for ax_i in np.ravel(axes):
        ax_i.set_xmargin(.05)
        ax_i.autoscale_view()

    # Now walk through again and plot the regression estimate
    # and a confidence interval for the regression line
    if fit_reg:
        for row_i, row_mask in enumerate(row_masks):
            for col_j, col_mask in enumerate(col_masks):
                if col_wrap is not None:
                    f_row = col_j // ncol
                    f_col = col_j % ncol
                else:
                    f_row, f_col = row_i, col_j
                ax = axes[f_row, f_col]
                xlim = ax.get_xlim()

                for hue_k, hue_mask in enumerate(hue_masks):
                    color = colors[hue_k]
                    data_ijk = data[row_mask & col_mask & hue_mask]
                    x_vals = np.array(data_ijk[x])
                    y_vals = np.array(data_ijk[y])
                    if not len(x_vals):
                        continue

                    # Sort out the limit of the fit
                    if truncate:
                        xx = np.linspace(x_vals.min(),
                                         x_vals.max(), 100)
                    else:
                        xx = np.linspace(xlim[0], xlim[1], 100)
                    xx_ = sm.add_constant(xx, prepend=True)

                    # Inner function to bootstrap the regression
                    def _regress(x, y):
                        if logistic:
                            x_ = sm.add_constant(x, prepend=True)
                            fit = sm.GLM(y, x_,
                                         family=sm.families.Binomial()).fit()
                            reg = fit.predict(xx_)
                        else:
                            fit = np.polyfit(x, y, order)
                            reg = np.polyval(fit, xx)
                        return reg

                    # Remove nuisance variables with vector rejection
                    if x_partial is not None:
                        for var in x_partial:
                            conf = data_ijk[var]
                            conf -= conf.mean()
                            x_mean = x_vals.mean()
                            x_vals = moss.vector_reject(x_vals - x_mean, conf)
                            x_vals += x_mean
                    if y_partial is not None:
                        for var in y_partial:
                            conf = data_ijk[var]
                            conf -= conf.mean()
                            y_mean = y_vals.mean()
                            y_vals = moss.vector_reject(y_vals - y_mean, conf)
                            y_vals += y_mean

                    # Regression line confidence interval
                    if ci is not None:
                        ci_lims = [50 - ci / 2., 50 + ci / 2.]
                        boots = moss.bootstrap(x_vals, y_vals,
                                               func=_regress,
                                               n_boot=n_boot)
                        ci_band = moss.percentiles(boots, ci_lims, axis=0)
                        ax.fill_between(xx, *ci_band, color=color, alpha=.15)

                    # Regression line
                    reg = _regress(x_vals, y_vals)
                    if color_factor is None:
                        label = ""
                    else:
                        label = hue_vals[hue_k]
                    ax.plot(xx, reg, color=color,
                            label=str(label), **line_kws)
                    ax.set_xlim(xlim)

    # Plot the legend on the upper left facet and adjust the layout
    if color_factor is not None and color_factor not in [row, col]:
        axes[0, 0].legend(loc="best", title=color_factor)
    plt.tight_layout()


def regplot(x, y, data=None, corr_func=stats.pearsonr, func_name=None,
            xlabel="", ylabel="", ci=95, size=None, annotloc=None, color=None,
            reg_kws=None, scatter_kws=None, dist_kws=None, text_kws=None):
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
        and return a numeric or (statistic, pval) tuple
    func_name : string, optional
        use for fit statistic annotation in lieu of function name
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
        if not any(map(bool, [xlabel, ylabel])):
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
    dist_kws["xlabel"] = False
    distplot(x, ax=ax_x_marg, **dist_kws)
    distplot(y, ax=ax_y_marg, vertical=True, **dist_kws)
    for ax in [ax_x_marg, ax_y_marg]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

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

    # Calcluate a fit statistic and p value
    if func_name is None:
        func_name = corr_func.__name__
    out = corr_func(x, y)
    try:
        s, p = out
        msg = "%s: %.3f (p=%.3g%s)" % (func_name, s, p, moss.sig_stars(p))
    except TypeError:
        s = corr_func(x, y)
        msg = "%s: %.3f" % (func_name, s)

    if annotloc is None:
        xmin, xmax = xlim
        x_range = xmax - xmin
        # Assume the fit statistic is correlation-esque for some
        # intuition on where the fit annotation should go
        if s < 0:
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

    # Set the axes on the marginal plots
    ax_x_marg.set_xlim(ax_scatter.get_xlim())
    ax_x_marg.set_yticks([])
    ax_y_marg.set_ylim(ax_scatter.get_ylim())
    ax_y_marg.set_xticks([])


def coefplot(formula, data, groupby=None, intercept=False, ci=95,
             palette="husl"):
    """Plot the coefficients from a linear model.

    Parameters
    ----------
    formula : string
        patsy formula for ols model
    data : dataframe
        data for the plot; formula terms must appear in columns
    groupby : grouping object, optional
        object to group data with to fit conditional models
    intercept : bool, optional
        if False, strips the intercept term before plotting
    ci : float, optional
        size of confidence intervals
    palette : seaborn color palette, optional
        palette for the horizonal plots

    """
    alpha = 1 - ci / 100
    if groupby is None:
        coefs = sf.ols(formula, data).fit().params
        cis = sf.ols(formula, data).fit().conf_int(alpha)
    else:
        grouped = data.groupby(groupby)
        coefs = grouped.apply(lambda d: sf.ols(formula, d).fit().params).T
        cis = grouped.apply(lambda d: sf.ols(formula, d).fit().conf_int(alpha))

    # Possibly ignore the intercept
    if not intercept:
        coefs = coefs.ix[1:]

    n_terms = len(coefs)

    # Plot seperately depending on groupby
    w, h = mpl.rcParams["figure.figsize"]
    hsize = lambda n: n * (h / 2)
    wsize = lambda n: n * (w / (4 * (n / 5)))
    if groupby is None:
        colors = itertools.cycle(color_palette(palette, n_terms))
        f, ax = plt.subplots(1, 1, figsize=(wsize(n_terms), hsize(1)))
        for i, term in enumerate(coefs.index):
            color = colors.next()
            low, high = cis.ix[term]
            ax.plot([i, i], [low, high], c=color,
                    solid_capstyle="round", lw=2.5)
            ax.plot(i, coefs.ix[term], "o", c=color, ms=8)
        ax.set_xlim(-.5, n_terms - .5)
        ax.axhline(0, ls="--", c="dimgray")
        ax.set_xticks(range(n_terms))
        ax.set_xticklabels(coefs.index)

    else:
        n_groups = len(coefs.columns)
        f, axes = plt.subplots(n_terms, 1, sharex=True,
                               figsize=(wsize(n_groups), hsize(n_terms)))
        if n_terms == 1:
            axes = [axes]
        colors = itertools.cycle(color_palette(palette, n_groups))
        for ax, term in zip(axes, coefs.index):
            for i, group in enumerate(coefs.columns):
                color = colors.next()
                low, high = cis.ix[(group, term)]
                ax.plot([i, i], [low, high], c=color,
                        solid_capstyle="round", lw=2.5)
                ax.plot(i, coefs.loc[term, group], "o", c=color, ms=8)
            ax.set_xlim(-.5, n_groups - .5)
            ax.axhline(0, ls="--", c="dimgray")
            ax.set_title(term)
        ax.set_xlabel(groupby)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(coefs.columns)


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
             color=None, vertical=False, legend=False, xlabel=None, ax=None):
    """Flexibly plot a distribution of observations.

    Parameters
    ----------
    a : (squeezable to) 1d array
        observed data
    bins : argument for matplotlib hist(), or None
        specification of bins or None to use Freedman-Diaconis rule
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
        if True, add a legend to the plot with what the plotted lines are
    xlabel : string, False, or None
        name for the x axis label. if None, will try to get it from a.name
        if False, do not set the x label
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
        fit_kws["label"] = fit.name
        ax.plot(x, y, color=fit_color, **fit_kws)

    if legend:
        ax.legend(loc="best")

    if label_x:
        ax.set_xlabel(xlabel)

    return ax


def kdeplot(a, npts=1000, shade=False, support_thresh=1e-4,
            support_min=-np.inf, support_max=np.inf,
            vertical=False, legend=False, ax=None, **kwargs):
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
    legend : bool, default True
        if True, add a legend to the plot with what the plotted lines are
    ax : matplotlib axis, optional
        axis to plot on, otherwise creates new one
    kwargs : other keyword arguments for plot()

    Returns
    -------
    ax : matplotlib axis
        axis with plot

    """
    if ax is None:
        ax = plt.gca()

    # If the input data has a name attribute then we can use it as a label.
    # However, if the user sets a label in the function call, use that value
    # instead of the object name. If there's no label set but the user wants
    # a legend anyway then use "kde" as a default name.
    if "label" not in kwargs:
        if a.name:
            kwargs["label"] = a.name
        else:
            kwargs["label"] = "kde"
    else:
        if "label" not in kwargs:
            kwargs["label"] = "kde"

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

    # Sometimes this can mess with the limits at the bottom of the plot
    if vertical:
        max = ax.get_xlim()[1]
        ax.set_xlim(0, max)
    else:
        max = ax.get_ylim()[1]
        ax.set_ylim(0, max)

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


def violin(vals, groupby=None, inner="box", color=None, positions=None,
           names=None, widths=.8, alpha=None, join_rm=False, kde_thresh=1e-2,
           inner_kws=None, ax=None, **kwargs):
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
    widths : float
        width of each violin at maximum density
    alpha : float, optional
        transparancy of violin fill
    join_rm : boolean, optional
        if True, positions in the input arrays are treated as repeated
        measures and are joined with a line plot
    names : list of strings, optional
        names to plot on x axis, otherwise plots numbers
    kde_thresh : float, optional
        proportion of maximum at which to threshold the KDE curve
    inner_kws : dict, optional
        keyword arugments for inner plot
    ax : matplotlib axis, optional
        axis to plot on, otherwise creates new one

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

    for i, a in enumerate(vals):
        x = positions[i]
        kde = stats.gaussian_kde(a)
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
            ax.plot((side * dens) + x, y, c=gray, linewidth=1.5)

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


def corrplot(data, names=None, annot=True, sig_stars=True, sig_tail="both",
             sig_corr=True, cmap=None, cmap_range=None, cbar=True,
             diag_names=True, ax=None, **kwargs):
    """Plot a correlation matrix with colormap and r values.

    Parameters
    ----------
    data : Dataframe or nobs x nvars array
        Rectangular nput data with variabes in the columns.
    names : sequence of strings
        Names to associate with variables if `data` is not a DataFrame.
    annot : bool
        Whether to annotate the upper triangle with correlation coefficients.
    sig_stars : bool
        If True, get significance with permutation test and denote with stars.
    sig_tail : both | upper | lower
        Direction for significance test. Also controls the default colorbar.
    sig_corr : bool
        If True, use FWE-corrected p values for the sig stars.
    cmap : colormap
        Colormap name as string or colormap object.
    cmap_range : None, "full", (low, high)
        Either truncate colormap at (-max(abs(r)), max(abs(r))), use the
        full range (-1, 1), or specify (min, max) values for the colormap.
    cbar : bool
        If true, plot the colorbar legend.
    ax : matplotlib axis
        Axis to draw plot in.
    kwargs : other keyword arguments
        Passed to ax.matshow()

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot.

    """
    if not isinstance(data, pd.DataFrame):
        if names is None:
            names = ["var_%d" % i for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=names, dtype=np.float)

    # Calculate the correlation matrix of the dataframe
    corrmat = data.corr()

    # Pandas will drop non-numeric columns; let's keep track of that operation
    names = corrmat.columns
    data = data[names]

    # Get p values with a permutation test
    if annot and sig_stars:
        p_mat = moss.randomize_corrmat(data.values.T, sig_tail, sig_corr)
    else:
        p_mat = None

    # Sort out the color range
    if cmap_range is None:
        triu = np.triu_indices(len(corrmat), 1)
        vmax = min(1, np.max(np.abs(corrmat.values[triu])) * 1.15)
        vmin = -vmax
        if sig_tail == "both":
            cmap_range = vmin, vmax
        elif sig_tail == "upper":
            cmap_range = 0, vmax
        elif sig_tail == "lower":
            cmap_range = vmin, 0
    elif cmap_range == "full":
        cmap_range = (-1, 1)

    # Find a colormapping, somewhat intelligently
    if cmap is None:
        if min(cmap_range) >= 0:
            cmap = "PuRd"
        elif max(cmap_range) <= 0:
            cmap = "GnBu_r"
        else:
            cmap = "coolwarm"
    if cmap == "jet":
        # Paternalism
        raise ValueError("Never use the 'jet' colormap!")



    # Plot using the more general symmatplot function
    ax = symmatplot(corrmat, p_mat, names, cmap, cmap_range,
                    cbar, annot, diag_names, ax, **kwargs)

    return ax


def symmatplot(mat, p_mat=None, names=None, cmap="Greys", cmap_range=None,
               cbar=True, annot=True, diag_names=True, ax=None, **kwargs):
    """Plot a symettric matrix with colormap and statistic values."""
    if ax is None:
        ax = plt.gca()

    nvars = len(mat)
    if isinstance(mat, pd.DataFrame):
        plotmat = mat.values.copy()
        mat = mat.values
    else:
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
        plt.colorbar(mat_img, shrink=.75)

    if p_mat is None:
        p_mat = np.ones((nvars, nvars))

    if annot:
        for i, j in zip(*np.triu_indices(nvars, 1)):
            val = mat[i, j]
            stars = moss.sig_stars(p_mat[i, j])
            ax.text(j, i, "\n%.3g\n%s" % (val, stars),
                    fontdict=dict(ha="center", va="center"))
    else:
        fill = np.ones_like(plotmat)
        fill[np.tril_indices_from(fill, -1)] = np.nan
        ax.matshow(fill, cmap="Greys", vmin=0, vmax=0, zorder=2)

    if names is None:
        names = ["var%d" % i for i in range(nvars)]
    
    if diag_names:
        for i, name in enumerate(names):
            ax.text(i, i, name, fontdict=dict(ha="center", va="center",
                                              weight="bold", rotation=45))
        ax.set_xticklabels(())
        ax.set_yticklabels(())
    else:
        ax.xaxis.set_ticks_position("bottom")
        xnames = names if annot else names[:-1]
        ax.set_xticklabels(xnames, rotation=90)
        ynames = names if annot else names[1:]
        ax.set_yticklabels(ynames)

    
    minor_ticks = np.linspace(-.5, nvars - 1.5, nvars)
    ax.set_xticks(minor_ticks, True)
    ax.set_yticks(minor_ticks, True)
    major_ticks = np.linspace(0, nvars - 1, nvars)
    xticks = major_ticks if annot else major_ticks[:-1]
    ax.set_xticks(xticks)
    yticks = major_ticks if annot else major_ticks[1:]
    ax.set_yticks(yticks)
    ax.grid(False, which="major")
    ax.grid(True, which="minor", linestyle="-")

    return ax


def palplot(pal, size=1):
    """Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot

    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
