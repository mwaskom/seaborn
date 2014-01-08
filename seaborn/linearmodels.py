"""Plotting functions for linear models (broadly construed)."""
from __future__ import division
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance
import statsmodels.api as sm
import statsmodels.formula.api as sf
import matplotlib as mpl
import matplotlib.pyplot as plt
import moss

from six.moves import range

from .utils import color_palette
from .axisgrid import FacetGrid


def lmplot(x, y, data, hue=None, col=None, row=None, palette="husl",
           col_wrap=None, size=5, aspect=1, sharex=True, sharey=True,
           **kwargs):

    # Backwards-compatibility warning layer
    if "color" in kwargs:
        msg = "`color` is deprecated and will be removed; using `hue` instead."
        warnings.warn(msg, UserWarning)
        hue = kwargs.pop("color")

    # Initialize the grid
    facets = FacetGrid(data, row, col, hue, palette=palette,
                       size=size, aspect=aspect, col_wrap=col_wrap)

    # Hack to set the x limits properly, which needs to happen here
    # because the extent of the regression estimate is determined
    # by the limits of the plot
    if sharex:
        for ax in facets.axes.flat:
            scatter = ax.scatter(data[x], np.ones(len(data)) * data[y].mean())
            scatter.remove()

    # Draw the regression plot on each facet
    facets.map(regplot, x, y, **kwargs)
    return facets


def factorplot(x, y=None, data=None, hue=None, row=None, col=None,
               col_wrap=None, estimator=np.mean, ci=95, n_boot=1000,
               kind="auto", dodge=0, join=True, size=5, aspect=1, palette=None,
               legend=True, legend_out=True, dropna=True,
               sharex=True, sharey=True):
    """Plot a dependent variable with uncertainty sorted by discrete factors.

    Parameters
    ----------
    x : string
        Variable name in `data` for splitting the plot on the x axis.
    y : string, optional
        Variable name in `data` for the dependent variable. If omitted, the
        counts within each bin are plotted (without confidence intervals).
    data : DataFrame
        Long-form (tidy) dataframe with variables in columns and observations
        in rows.
    hue : string, optional
        Variable name in `data` for splitting the plot by color. In the case
        of `kind="bar"`, this also influences the placement on the x axis.
    row, col : strings, optional
        Variable name(s) in `data` for splitting the plot into a facet grid
        along row and columns.
    col_wrap : int or None, optional
        Wrap the column variable at this width (incompatible with `row`).
    estimator : vector -> scalar function, optional
        Function to aggregate `y` values at each level of the factors.
    ci : int in {0, 100}, optional
        Size of confidene interval to draw around the aggregated value.
    n_boot : int, optional
        Number of bootstrap resamples used to compute confidence interval.
    kind : {"auto", "point", "bar"}, optional
        Visual representation of the plot. "auto" uses a few heuristics to
        guess whether "bar" or "point" is more appropriate.
    dodge : positive scalar, optional
        Horizontal offset applies to different `hue` levels. Only relevant
        when kind is "point".
    join : boolean, optional
        Whether points from the same level of `hue` should be joined. Only
        relevant when kind is "point".
    size : positive scalar, optional
        Height (in inches) of each facet.
    aspect : positive scalar, optional
        Ratio of facet width to facet height.
    palette : seaborn color palette, optional
        Palette to map `hue` variable with (or `x` variable when `hue` is
        None).
    legend : boolean, optional
        Draw a legend, only if `hue` is used and does not overlap with other
        variables.
    legend_out : boolean, optional
        Draw the legend outside the grid; otherwise it is placed within the
        first facet.
    dropna : boolean, optional
        Remove observations that are NA within any variables used to make
        the plot.
    share{x, y} : booleans, optional
        Lock the limits of the vertical and/or horizontal axes across the
        facets.

    Returns
    -------
    facet : FacetGrid
        FacetGrid with the plot on it.

    """
    if hue is not None and palette is None:
        palette = "husl"
    elif hue is None and palette is not None:
        hue = x

    facet = FacetGrid(data, row, col, hue, col_wrap=col_wrap, size=size,
                      aspect=aspect, legend=legend, legend_out=legend_out,
                      palette=palette, dropna=True,
                      sharex=sharex, sharey=sharey)

    if kind == "auto":
        if y is None:
            kind = "bar"
        elif (data[y] <= 1).all():
            kind = "point"
        elif (data[y].mean() / data[y].std()) < 2.5:
            kind = "bar"
        else:
            kind = "point"

    mask_gen = facet.facet_data()

    x_order = sorted(data[x].unique())
    if dropna:
        x_order = list(filter(pd.notnull, x_order))

    if y is None:
        estimator = len
        ci = None
        y = x
        y_count = True
    else:
        y_count = False

    if hue is None:
        colors = ["#777777" if kind == "bar" else "#333333"]
        n_colors = 1
        width = .8
        pos_adjust = [0]
    else:
        hue_order = sorted(data[hue].unique())
        if dropna:
            hue_order = list(filter(pd.notnull, hue_order))
        if isinstance(palette, dict):
            palette = [palette[h] for h in hue_order]
        n_colors = len(hue_order)
        colors = color_palette(palette, n_colors)
        if hue == row or hue == col or hue == x:
            width = .8
            pos_adjust = [0] * n_colors
        else:
            width = .8 / n_colors
            if kind == "bar":
                pos_adjust = np.linspace(0, .8 - width, n_colors)
                pos_adjust -= pos_adjust.mean()
            elif kind == "point" and dodge:
                pos_adjust = np.linspace(0, dodge, n_colors)
                pos_adjust -= pos_adjust.mean()
            else:
                pos_adjust = [0] * n_colors

    draw_legend = hue is not None and hue not in [x, row, col]

    for (row_i, col_j, hue_k), data_ijk in mask_gen:

        if col_wrap is not None:
            f_row = col_j // facet._ncol
            f_col = col_j % facet._ncol
        else:
            f_row, f_col = row_i, col_j

        ax = facet.axes[f_row, f_col]

        plot_data = data_ijk[[x] if y_count else [x, y]]
        if dropna:
            plot_data.dropna()

        grouped = {g: data[y] for g, data in plot_data.groupby(x)}

        plot_pos = []
        plot_heights = []
        plot_cis = []
        for pos, var in enumerate(x_order):

            if var not in grouped:
                continue

            plot_pos.append(pos + pos_adjust[hue_k])
            plot_heights.append(estimator(grouped[var]))
            if ci is not None:
                boots = moss.bootstrap(grouped[var].values,
                                       func=estimator, n_boot=n_boot)
                plot_cis.append(moss.ci(boots, ci))

        kwargs = {}
        if draw_legend:
            kwargs["label"] = facet.hue_names[hue_k]

        if kind == "bar":

            ax.bar(plot_pos, plot_heights, width, align="center",
                   color=colors[hue_k], **kwargs)
            facet._update_legend_data(ax)
            if ci is not None:
                for pos, ci_ in zip(plot_pos, plot_cis):
                    ax.plot([pos, pos], ci_, linewidth=2.5, color="#222222")

        elif kind == "point":

            hue = colors[hue_k]
            ls = "-" if join else ""
            lw = mpl.rcParams["lines.linewidth"] * 1.5
            ax.plot(plot_pos, plot_heights, color=hue, marker="o", ms=9,
                    ls=ls, lw=lw, **kwargs)
            facet._update_legend_data(ax)
            if ci is not None:
                for pos, ci_ in zip(plot_pos, plot_cis):
                    ax.plot([pos, pos], ci_, linewidth=lw, color=hue)

    n_x = len(x_order)
    facet.set(xticks=range(n_x), xticklabels=x_order, xlim=(-.5, n_x - .5))

    for ax in facet.axes.flat:
        ax.xaxis.grid(False)

    y_label = "count" if y_count else y
    facet.set_axis_labels(x, y_label)
    facet.set_titles()
    if draw_legend:
        facet.set_legend()

    return facet


def _regress_fast(grid, x, y, ci, n_boot):
    """Low-level regression and prediction using linear algebra."""
    X = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    reg_func = lambda _x, _y: np.linalg.pinv(_x).dot(_y)
    y_hat = grid.dot(reg_func(X, y))
    if ci is None:
        return y_hat, None

    beta_boots = moss.bootstrap(X, y, func=reg_func, n_boot=n_boot).T
    y_hat_boots = grid.dot(beta_boots).T
    return y_hat, y_hat_boots


def _regress_poly(grid, x, y, order, ci, n_boot):
    """Regression using numpy polyfit for higher-order trends."""
    reg_func = lambda _x, _y: np.polyval(np.polyfit(_x, _y, order), grid)
    y_hat = reg_func(x, y)
    if ci is None:
        return y_hat, None

    y_hat_boots = moss.bootstrap(x, y, func=reg_func, n_boot=n_boot)
    return y_hat, y_hat_boots


def _regress_statsmodels(grid, x, y, model, ci, n_boot, **kwargs):
    """More general regression function using statsmodels objects."""
    X = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    reg_func = lambda _x, _y: model(_y, _x, **kwargs).fit().predict(grid)
    y_hat = reg_func(X, y)
    if ci is None:
        return y_hat, None

    y_hat_boots = moss.bootstrap(X, y, func=reg_func, n_boot=n_boot)
    return y_hat, y_hat_boots


def _bin_predictor(x, bins):
    """Discretize a continuous predictor by assigning value to closest bin."""
    if np.isscalar(bins):
        bins = np.c_[np.linspace(x.min(), x.max(), bins + 2)[1:-1]]
    else:
        bins = np.c_[np.ravel(bins)]

    dist = distance.cdist(np.c_[x], bins)
    x_binned = bins[np.argmin(dist, axis=1)].ravel()

    return x_binned, bins.ravel()


def _point_est(x, y, estimator, ci, n_boot):
    """Find point estimate and bootstrapped ci for discrete x values."""
    vals = sorted(np.unique(x))
    points, cis = [], []
    for val in vals:

        _y = y[x == val]
        est = estimator(_y)
        points.append(est)

        if ci is None:
            cis.append(None)
        else:
            _ci = moss.ci(moss.bootstrap(_y, func=estimator,
                                         n_boot=n_boot), ci)
            cis.append(_ci)

    return vals, points, cis


def regplot(x, y, data=None, x_estimator=None, x_bins=None, x_ci=95,
            fit_reg=True, ci=95, n_boot=1000,
            order=1, logistic=False, robust=False, partial=None,
            truncate=False, dropna=True, x_jitter=None, y_jitter=None,
            xlabel=None, ylabel=None, label=None,
            color=None, scatter_kws=None, line_kws=None,
            ax=None):
    """Draw a scatter plot between x and y with a regression line.


    Parameters
    ----------
    x : vector or string
        Data or column name in `data` for the predictor variable.
    y : vector or string
        Data or column name in `data` for response predictor variable.
    data : DataFrame, optional
        DataFrame to use if `x` and `y` are column names.
    x_estimator : function that aggregates a vector into one value, optional
        When `x` is a discrete variable, apply this estimator to the data
        at each value and plot the data as a series of point estimates and
        confidence intervals rather than a scatter plot.
    x_bins : int or vector, optional
        When `x` is a continuous variable, use the values in this vector (or
        a vector of evenly spaced values with this length) to discretize the
        data by assigning each point to the closest bin value. This applies
        only to the plot; the regression is fit to the original data. This
        implies that `x_estimator` is numpy.mean if not otherwise provided.
    x_ci: int between 0 and 100, optional
        Confidence interval to compute and draw around the point estimates
        when `x` is treated as a discrete variable.
    fit_reg : boolean, optional
        If False, don't fit a regression; just draw the scatterplot.
    ci : int between 0 and 100 or None, optional
        Confidence interval to compute for regression estimate, which is drawn
        as translucent bands around the regression line.
    n_boot : int, optional
        Number of bootstrap resamples used to compute the confidence intervals.
    order : int, optional
        Order of the polynomial to fit. Use order > 1 to explore higher-order
        trends in the relationship.
    logistic : boolean, optional
        Fit a logistic regression model. This requires `y` to be dichotomous
        with values of either 0 or 1.
    robust : boolean, optional
        Fit a robust linear regression, which may be useful when the data
        appear to have outliers.
    partial : matrix or string(s) , optional
        Matrix with same first dimension as x and y, or column name(s) in
        `data`. These variables are treated as confounding and are removed from
        the data.
    truncate : boolean, optional
        If True, truncate the regression estimate at the minimum and maximum
        values of the `x` variable.
    dropna : boolean, optional
        Remove observations that are NA in at least one of the variables.
    {x, y}_jitter : floats, optional
        Add uniform random noise from within this range (in data coordinates)
        to each datapoint in the x and/or y direction. This can be helpful when
        plotting discrete values.
    {x, y}_label : None, string, or boolean, optional
        If None, try to infer variable names from the data objects and use them
        to annotate the plot. Otherwise, use the names provided here. Set to
        False to avoid altering the axis labels.
    label : string, optional
        Label to use for the regression line, or for the scatterplot if not
        fitting a regression.
    color : matplotlib color, optional
        Color to use for all elements of the plot. Can set the scatter and
        regression colors separately using the `kws` dictionaries. If not
        provided, the current color in the axis cycle is used.
    {scatter, line}_kws : dictionaries, optional
        Additional keyword arguments passed to scatter() and plot() for drawing
        the components of the plot.
    ax : matplotlib axis, optional
        Plot into this axis, otherwise grab the current axis or make a new
        one if not existing.

    Returns
    -------
    ax: matplotlib axis
        Axis with the regression plot.

    See Also
    --------
    TODO

    """
    if ax is None:
        ax = plt.gca()

    # Get the variables
    if data is not None:
        x = data[x]
        y = data[y]

    # Drop NA values
    if dropna:
        not_na = pd.notnull(x) & pd.notnull(y)
        x = x[not_na]
        y = y[not_na]

    # Try to find variable names
    x_name, y_name = None, None
    if hasattr(x, "name"):
        x_name = x.name
    if hasattr(y, "name"):
        y_name = y.name

    # Apply names to the plot
    if xlabel is None and x_name is not None:
        ax.set_xlabel(x_name)
    elif xlabel:
        ax.set_xlabel(x_name)
    if ylabel is None and y_name is not None:
        ax.set_ylabel(y_name)
    elif ylabel:
        ax.set_ylabel(y_name)

    # Coerce the input data to arrays
    x = np.asarray(x, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    if partial is not None:
        # This is a heuristic but unlikely to be wrong
        if data is not None and len(partial) != len(x):
            partial = data[partial]
        partial = np.asarray(partial, dtype=np.float)

    # Set mutable default arguments
    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}

    # Label the proper plot element
    if fit_reg:
        line_kws["label"] = label
    else:
        scatter_kws["label"] = label

    # Grab the current color in the cycle if one isn't provided
    if color is None:
        lines, = plt.plot(x.mean(), y.mean())
        color = lines.get_color()
        lines.remove()

    # Possibly regress confounding variables out of the dependent variable
    if partial is None:
        y_scatter = y.copy()
    else:
        X = np.c_[np.ones_like(x), x, partial]
        y_scatter = y - X[:, 2:].dot(np.linalg.pinv(X).dot(y)[2:])

    # Possibly bin the predictor variable, which implies a point estimate
    if x_bins is not None:
        x_estimator = np.mean if x_estimator is None else x_estimator
        x_discrete, x_bins = _bin_predictor(x, x_bins)

    # Add in some jitter
    if x_jitter is None:
        x_scatter = x
    else:
        x_scatter = x + np.random.uniform(-x_jitter, x_jitter, x.size)
    if y_jitter is not None:
        y_scatter += np.random.uniform(-y_jitter, y_jitter, y.size)

    # Get some defaults
    scatter_color = scatter_kws.pop("c", color)
    scatter_color = scatter_kws.pop("color", scatter_color)
    lw = scatter_kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.75)
    lw = scatter_kws.pop("linewidth", lw)

    # Draw the datapoints either as a scatter or point estimate with CIs
    if x_estimator is None:
        alpha = scatter_kws.pop("alpha", .8)
        ax.scatter(x_scatter, y_scatter,
                   color=scatter_color, alpha=alpha, **scatter_kws)
    else:
        if x_bins is None:
            x_discrete = x
        point_data = _point_est(x_discrete, y_scatter,
                                x_estimator, x_ci, n_boot)
        for x_val, height, ci_bounds in zip(*point_data):
            size = scatter_kws.pop("s", 50)
            size = scatter_kws.pop("size", size)
            ax.scatter(x_val, height, size, color=scatter_color, **scatter_kws)
            ax.plot([x_val, x_val], ci_bounds, color=scatter_color,
                    lw=lw, **scatter_kws)

    # Just bail out here if we don't want a regression
    if not fit_reg:
        return ax

    # Get the plot limits and set up a grid for the regression
    if truncate:
        x_min, x_max = x.min(), x.max()
    else:
        x_min, x_max = ax.get_xlim()
    grid = np.linspace(x_min, x_max, 100)

    # Validate the regression parameters
    if sum((order > 1, logistic, robust, partial is not None)) > 1:
        raise ValueError("`order` > 1 and `logistic` are mutually exclusive")

    # Fit the regression and bootstrap the prediction.
    # This gets delegated to one of several functions depending on the options.
    if order > 1:
        y_hat, y_hat_boots = _regress_poly(grid, x, y, order, ci, n_boot)

    elif logistic:
        binomial = sm.families.Binomial()
        y_hat, y_hat_boots = _regress_statsmodels(grid, x, y, sm.GLM, ci,
                                                  n_boot, family=binomial)
    elif robust:
        y_hat, y_hat_boots = _regress_statsmodels(grid, x, y, sm.RLM,
                                                  ci, n_boot)
    elif partial is not None:
        _x = np.c_[x, partial]
        _grid = np.c_[grid, np.zeros((grid.size, partial.shape[1]))]
        kwargs = dict()
        y_hat, y_hat_boots = _regress_statsmodels(_grid, _x, y, sm.OLS, kwargs,
                                                  ci, n_boot)
    else:
        y_hat, y_hat_boots = _regress_fast(grid, x, y, ci, n_boot)

    # Compute the confidence interval for the regression estimate
    if ci is not None:
        err_bands = moss.ci(y_hat_boots, ci, axis=0)

    # Draw the regression and standard error bands
    lw = line_kws.pop("linewidth", mpl.rcParams["lines.linewidth"] * 1.5)
    lw = line_kws.pop("lw", lw)
    _color = line_kws.pop("c", color)
    _color = line_kws.pop("color", _color)
    ax.plot(grid, y_hat, lw=lw, color=_color, **line_kws)
    if ci is not None:
        ax.fill_between(grid, *err_bands, color=_color, alpha=.15)

    # Reset the x limits in case they got stretched from fill bleedover
    ax.set_xlim(x_min, x_max)

    # Reset the y limits if this is a logistic plot to incude 0 and 1
    if logistic:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, 0), max(ymax, 1))

    return ax


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
            color = next(colors)
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
                color = next(colors)
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


def interactplot(x1, x2, y, data=None, filled=False, cmap="RdBu_r",
                 colorbar=True, levels=30, logistic=False,
                 contour_kws=None, scatter_kws=None, ax=None):
    """Visualize a continuous two-way interaction with a contour plot.

    Parameters
    ----------
    x1, x2, y, strings or array-like
        Either the two independent variables and the dependent variable,
        or keys to extract them from `data`
    data : DataFrame
        Pandas DataFrame with the data in the columns.
    filled : bool
        Whether to plot with filled or unfilled contours
    cmap : matplotlib colormap
        Colormap to represent yhat in the countour plot.
    colorbar : bool
        Whether to draw the colorbar for interpreting the color values.
    levels : int or sequence
        Number or position of contour plot levels.
    logistic : bool
        Fit a logistic regression model instead of linear regression.
    contour_kws : dictionary
        Keyword arguments for contour[f]().
    scatter_kws : dictionary
        Keyword arguments for plot().
    ax : matplotlib axis
        Axis to draw plot in.

    Returns
    -------
    ax : Matplotlib axis
        Axis with the contour plot.

    """
    # Handle the form of the data
    if data is not None:
        x1 = data[x1]
        x2 = data[x2]
        y = data[y]
    if hasattr(x1, "name"):
        xlabel = x1.name
    else:
        xlabel = None
    if hasattr(x2, "name"):
        ylabel = x2.name
    else:
        ylabel = None
    if hasattr(y, "name"):
        clabel = y.name
    else:
        clabel = None
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y = np.asarray(y)

    # Initialize the scatter keyword dictionary
    if scatter_kws is None:
        scatter_kws = {}
    if not ("color" in scatter_kws or "c" in scatter_kws):
        scatter_kws["color"] = "#222222"
    if not "alpha" in scatter_kws:
        scatter_kws["alpha"] = 0.75

    # Intialize the contour keyword dictionary
    if contour_kws is None:
        contour_kws = {}

    # Initialize the axis
    if ax is None:
        ax = plt.gca()

    # Plot once to let matplotlib sort out the axis limits
    ax.plot(x1, x2, "o", **scatter_kws)

    # Find the plot limits
    x1min, x1max = ax.get_xlim()
    x2min, x2max = ax.get_ylim()

    # Make the grid for the contour plot
    x1_points = np.linspace(x1min, x1max, 100)
    x2_points = np.linspace(x2min, x2max, 100)
    xx1, xx2 = np.meshgrid(x1_points, x2_points)

    # Fit the model with an interaction
    X = np.c_[np.ones(x1.size), x1, x2, x1 * x2]
    if logistic:
        lm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    else:
        lm = sm.OLS(y, X).fit()

    # Evaluate the model on the grid
    eval = np.vectorize(lambda x1_, x2_: lm.predict([1, x1_, x2_, x1_ * x2_]))
    yhat = eval(xx1, xx2)

    # Default color limits put the midpoint at mean(y)
    y_bar = y.mean()
    c_min = min(y.min(), yhat.min())
    c_max = max(y.max(), yhat.max())
    delta = max(c_max - y_bar, y_bar - c_min)
    c_min, cmax = y_bar - delta, y_bar + delta
    vmin = contour_kws.pop("vmin", c_min)
    vmax = contour_kws.pop("vmax", c_max)

    # Draw the contour plot
    func_name = "contourf" if filled else "contour"
    contour = getattr(ax, func_name)
    c = contour(xx1, xx2, yhat, levels, cmap=cmap,
                vmin=vmin, vmax=vmax, **contour_kws)

    # Draw the scatter again so it's visible
    ax.plot(x1, x2, "o", **scatter_kws)

    # Draw a colorbar, maybe
    if colorbar:
        bar = plt.colorbar(c)

    # Label the axes
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if clabel is not None and colorbar:
        clabel = "P(%s)" % clabel if logistic else clabel
        bar.set_label(clabel, labelpad=15, rotation=270)

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
            cmap = "OrRd"
        elif max(cmap_range) <= 0:
            cmap = "PuBu_r"
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
            ax.text(j, i, "\n%.2g\n%s" % (val, stars),
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
