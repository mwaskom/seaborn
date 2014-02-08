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

from six import string_types
from six.moves import range

from .utils import color_palette
from .axisgrid import FacetGrid


def lmplot(x, y, data, hue=None, col=None, row=None, palette="husl",
           col_wrap=None, size=5, aspect=1, sharex=True, sharey=True,
           col_order=None, row_order=None, hue_order=None, dropna=True,
           legend=True, legend_out=True, **kwargs):

    # Backwards-compatibility warning layer
    if "color" in kwargs:
        msg = "`color` is deprecated and will be removed; using `hue` instead."
        warnings.warn(msg, UserWarning)
        hue = kwargs.pop("color")

    # Reduce the dataframe to only needed columns
    # Otherwise when dropna is True we could lose data because it is missing
    # in a column that isn't relevant ot this plot
    units = kwargs.get("units", None)
    x_partial = kwargs.get("x_partial", None)
    y_partial = kwargs.get("y_partial", None)
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = [a for a in need_cols if a is not None]
    data = data[cols]

    # Initialize the grid
    facets = FacetGrid(data, row, col, hue, palette=palette,
                       row_order=row_order, col_order=col_order,
                       hue_order=hue_order, dropna=dropna,
                       size=size, aspect=aspect, col_wrap=col_wrap,
                       legend=legend, legend_out=legend_out)

    # Hack to set the x limits properly, which needs to happen here
    # because the extent of the regression estimate is determined
    # by the limits of the plot
    if sharex:
        for ax in facets.axes.flat:
            scatter = ax.scatter(data[x], np.ones(len(data)) * data[y].mean())
            scatter.remove()

    # Draw the regression plot on each facet
    facets.map_dataframe(regplot, x, y, **kwargs)
    return facets


def _factorplot():
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


def factorplot(x, y=None, hue=None, data=None, row=None, col=None,
               col_wrap=None,  estimator=np.mean, ci=95, n_boot=1000,
               units=None, x_order=None, hue_order=None, col_order=None,
               row_order=None, kind="auto", dodge=0, join=True, size=5,
               aspect=1, palette=None, legend=True, legend_out=True,
               dropna=True, sharex=True, sharey=True):

    cols = [a for a in [x, y, hue, col, row, units] if a is not None]
    cols = pd.unique(cols).tolist()
    data = data[cols]

    facet_hue = hue if hue in [row, col] else None
    facet_palette = palette if hue in [row, col] else None

    # Initialize the grid
    facets = FacetGrid(data, row, col, facet_hue, palette=facet_palette,
                       row_order=row_order, col_order=col_order, dropna=dropna,
                       size=size, aspect=aspect, col_wrap=col_wrap, legend=legend,
                       legend_out=legend_out)

    if kind == "auto":
        if y is None:
            kind = "bar"
        elif (data[y] <= 1).all():
            kind = "point"
        elif (data[y].mean() / data[y].std()) < 2.5:
            kind = "bar"
        else:
            kind = "point"

    # Draw the plot on each facet
    kwargs = dict(estimator=estimator, ci=ci, n_boot=n_boot, units=units,
                  x_order=x_order, hue_order=hue_order)

    if hue in [row, col]:
        hue = None
    else:
        kwargs["palette"] = palette

    if kind == "bar":
        facets.map_dataframe(barplot, x, y, hue, **kwargs)
    else:
        kwargs.update(dict(dodge=dodge, join=join))
        facets.map_dataframe(pointplot, x, y, hue, **kwargs)

    if legend and (hue is not None) and (hue not in [x, row, col]):
        facets.set_legend(title=hue)

    return facets


class _DiscretePlotter(object):

    def __init__(self, x, y=None, hue=None, data=None, units=None,
                 x_order=None, hue_order=None, color=None, palette=None,
                 kind="auto", dodge=0, join=True, hline=0,
                 estimator=np.mean, ci=95, n_boot=1000):

        # This implies we have a single bar/point for each level of `x`
        # but that the different levels should be mapped with a palette
        self.x_palette = hue is None and palette is not None

        # Set class attributes based on inputs
        self.estimator = len if y is None else estimator
        self.ci = None if y is None else ci
        self.join = join
        self.n_boot = n_boot
        self.hline = hline

        # Other attributs that are hardcoded for now
        self.bar_widths = .8
        self.err_color = "#444444"

        # Once we've set the above values, if `y` is None we want the actual
        # y values to be the x values so we can count them
        self.y_count = y is None
        if y is None:
            y = x

        # Ascertain which values will be associated with what values
        self.establish_variables(x, y, hue, units, data, x_order, hue_order)

        # Settle whe kind of plot this is going to be
        self.establish_plot_kind(kind)

        # Determine the color palette
        self.establish_palette(color, palette)

        # Figure out where the data should be drawn
        self.establish_positions(dodge)

    def establish_variables(self, x, y, hue, units, data,  x_order, hue_order):
        """Extract variables from data or use directly."""
        self.data = data

        # Determine the value for each variable, which could be passed
        # as an array-like or a key into the `data` DataFrame
        for var in ["x", "y", "hue", "units"]:
            val = locals()[var]
            if isinstance(val, string_types):
                setattr(self, var, data[val])
            else:
                setattr(self, var, val)

        # Determine the order that levels of `x` and `hue` should appear
        x_sorted = np.sort(pd.unique(self.x))
        self.x_order = x_sorted if x_order is None else x_order

        if self.hue is not None:
            hue_sorted = np.sort(pd.unique(self.hue))
            self.hue_order = hue_sorted if hue_order is None else hue_order
        else:
            self.hue_order = [None]

    def establish_palette(self, color, palette):
        """Set a list of colors for each plot element."""
        n_hues = len(self.x_order) if self.x_palette else len(self.hue_order)
        hue_names = self.x_order if self.x_palette else self.hue_order
        if self.hue is None and not self.x_palette:
            if color is None:
                color = color_palette()[0]
            palette = [color for _ in self.x_order]
        elif palette is None:
            palette = color_palette(n_colors=n_hues)
        elif isinstance(palette, dict):
            palette = [palette[k] for k in hue_names]
            palette = color_palette(palette, n_hues)
        else:
            palette = color_palette(palette, n_hues)
        self.palette = palette

        if self.kind == "point":
            self.err_palette = palette
        else:
            # TODO make this smarter
            self.err_palette = [self.err_color] * len(palette)

    def establish_positions(self, dodge):
        """Make list of center values for each x and offset for each hue."""
        self.positions = np.arange(len(self.x_order))

        # If there's no hue variable kind is irrelevant
        if self.hue is None:
            n_hues = 1
            width = .8
            offset = np.zeros(n_hues)
        else:
            n_hues = len(self.hue_order)

            # Bar offset is set by hardcoded bar width
            if self.kind == "bar":
                width = self.bar_widths / n_hues
                offset = np.linspace(0, self.bar_widths - width, n_hues)
                self.bar_widths = width

            # Point offset is set by `dodge` parameter
            elif self.kind == "point":
                offset = np.linspace(0, dodge, n_hues)
            offset -= offset.mean()

        self.offset = offset

    def establish_plot_kind(self, kind):
        """Use specified kind of apply heuristics to decide automatically."""
        if kind == "auto":
            y = self.y

            # Walk through some heuristics to automatically assign a kind
            if self.y_count:
                kind = "bar"
            elif y.max() <= 1:
                kind = "point"
            elif (y.mean() / y.std()) < 2.5:
                kind = "bar"
            else:
                kind = "point"
            self.kind = kind
        elif kind in ["bar", "point"]:
            self.kind = kind
        else:
            raise ValueError("%s is not a valid kind of plot" % kind)

    @property
    def plot_data(self):

        # First iterate through the hues, as plots are drawn for all
        # positions of a given hue at the same time
        for i, hue in enumerate(self.hue_order):

            # Build intermediate lists of the values for each drawing
            pos = []
            height = []
            ci = []
            for j, x in enumerate(self.x_order):

                pos.append(self.positions[j] + self.offset[i])

                # Focus on the data for this specific bar/point
                current_data = (self.x == x) & (self.hue == hue)
                y_data = self.y[current_data]
                if self.units is None:
                    unit_data = None
                else:
                    unit_data = self.units[current_data]

                # This is where the main computation happens
                height.append(self.estimator(y_data))
                if self.ci is not None:
                    boots = moss.bootstrap(y_data, func=self.estimator,
                                           n_boot=self.n_boot,
                                           units=unit_data)
                    ci.append(moss.ci(boots, self.ci))

            yield pos, height, ci

    def plot(self, ax):
        """Plot based on the stored value for kind of plot."""
        plotter = getattr(self, self.kind + "plot")
        plotter(ax)

        # Set the plot attributes (these are shared across plot kinds
        if self.hue is not None:
            leg = ax.legend(loc="best", scatterpoints=1)
            if hasattr(self.hue, "name"):
                leg.set_title(self.hue.name,
                              prop={"size": mpl.rcParams["axes.labelsize"]})
        ax.xaxis.grid(False)
        ax.set_xticks(self.positions)
        ax.set_xticklabels(self.x_order)
        if hasattr(self.x, "name"):
            ax.set_xlabel(self.x.name)
        if self.y_count:
            ax.set_ylabel("count")
        else:
            if hasattr(self.y, "name"):
                ax.set_ylabel(self.y.name)

        if self.hline is not None:
            ymin, ymax = ax.get_ylim()
            if self.hline > ymin and self.hline < ymax:
                ax.axhline(self.hline, c="#666666")

    def barplot(self, ax):
        """Draw the plot with a bar representation."""
        for i, (pos, height, ci) in enumerate(self.plot_data):

            color = self.palette if self.x_palette else self.palette[i]
            ecolor = self.err_palette[i]
            label = self.hue_order[i]

            # The main plot
            ax.bar(pos, height, self.bar_widths, color=color,
                   label=label, align="center")

            # The error bars
            for x, (low, high) in zip(pos, ci):
                ax.plot([x, x], [low, high], linewidth=2.5, color=ecolor)

        # Set the x limits
        offset = .5
        xlim = self.positions.min() - offset, self.positions.max() + offset
        ax.set_xlim(xlim)

    def pointplot(self, ax):
        """Draw the plot with a point representation."""
        for i, (pos, height, ci) in enumerate(self.plot_data):

            color = self.palette if self.x_palette else self.palette[i]
            err_palette =  self.err_palette
            label = self.hue_order[i]

            # The error bars
            for j, (x, (low, high)) in enumerate(zip(pos, ci)):
                ecolor = err_palette[j] if self.x_palette else err_palette[i]
                ax.plot([x, x], [low, high], linewidth=2.5, color=ecolor)

            # The main plot
            ax.scatter(pos, height, s=75, color=color, label=label)

            # The join line
            if self.join:
                ax.plot(pos, height, color=color, linewidth=2.5)

        # Set the x limits
        xlim = (self.positions.min() + self.offset.min() - .3,
                self.positions.max() + self.offset.max() + .3)
        ax.set_xlim(xlim)

# TODO dropna

def barplot(x, y=None, hue=None, data=None, estimator=np.mean, hline=0,
            ci=95, n_boot=1000, units=None, x_order=None, hue_order=None,
            color=None, palette=None, label=None, ax=None):

    plotter = _DiscretePlotter(x, y, hue, data, units, x_order, hue_order,
                               color, palette, "bar", 0, False, hline,
                               estimator, ci, n_boot)

    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax


def pointplot(x, y, hue=None, data=None, estimator=np.mean, hline=True,
              ci=95, n_boot=1000, units=None, x_order=None, hue_order=None,
              dodge=0, color=None, palette=None, join=True, label=None, ax=None):

    plotter = _DiscretePlotter(x, y, hue, data, units, x_order, hue_order,
                               color, palette, "point", dodge, join, hline,
                               estimator, ci, n_boot)

    if ax is None:
        ax = plt.gca()
    plotter.plot(ax)
    return ax


def _regress_fast(grid, x, y, units, ci, n_boot):
    """Low-level regression and prediction using linear algebra."""
    X = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    reg_func = lambda _x, _y: np.linalg.pinv(_x).dot(_y)
    y_hat = grid.dot(reg_func(X, y))
    if ci is None:
        return y_hat, None

    beta_boots = moss.bootstrap(X, y, func=reg_func,
                                n_boot=n_boot, units=units).T
    y_hat_boots = grid.dot(beta_boots).T
    return y_hat, y_hat_boots


def _regress_poly(grid, x, y, units, order, ci, n_boot):
    """Regression using numpy polyfit for higher-order trends."""
    reg_func = lambda _x, _y: np.polyval(np.polyfit(_x, _y, order), grid)
    y_hat = reg_func(x, y)
    if ci is None:
        return y_hat, None

    y_hat_boots = moss.bootstrap(x, y, func=reg_func,
                                 n_boot=n_boot, units=units)
    return y_hat, y_hat_boots


def _regress_statsmodels(grid, x, y, units, model, ci, n_boot, **kwargs):
    """More general regression function using statsmodels objects."""
    X = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    reg_func = lambda _x, _y: model(_y, _x, **kwargs).fit().predict(grid)
    y_hat = reg_func(X, y)
    if ci is None:
        return y_hat, None

    y_hat_boots = moss.bootstrap(X, y, func=reg_func,
                                 n_boot=n_boot, units=units)
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


def _point_est(x, y, estimator, ci, units, n_boot):
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
            _units = units
            if units is not None:
                _units = units[x == val]
            _ci = moss.ci(moss.bootstrap(_y, func=estimator, n_boot=n_boot,
                                         units=_units), ci)
            cis.append(_ci)

    return vals, points, cis


def _regress_out(a, b):
    """Regress b from a keeping a's original mean."""
    a_mean = a.mean()
    a = a - a_mean
    b = b - b.mean()
    b = np.c_[b]
    a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
    return (a_prime + a_mean).reshape(a.shape)


def regplot(x, y, data=None, x_estimator=None, x_bins=None, x_ci=95,
            scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
            order=1, logistic=False, lowess=False, robust=False,
            x_partial=None, y_partial=None,
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
    scatter : boolean, optional
        Draw the scatter plot or point estimates with CIs representing the
        observed data.
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
    lowess : boolean, optional
        Plot a lowess model (locally weighted nonparametric regression).
    robust : boolean, optional
        Fit a robust linear regression, which may be useful when the data
        appear to have outliers.
    {x, y}_partial : matrix or string(s) , optional
        Matrix with same first dimension as `x`, or column name(s) in `data`.
        These variables are treated as confounding and are removed from
        the `x` or `y` variables before plotting.
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
        if units is not None:
            units = data[units].values

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

    # This is a heuristic but unlikely to be wrong
    if x_partial is not None:
        if data is not None and len(x_partial) != len(x):
            x_partial = data[x_partial]
    if y_partial is not None:
        if data is not None and len(y_partial) != len(y):
            y_partial = data[y_partial]

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
    if x_partial is not None:
        x = _regress_out(x, x_partial)
    if y_partial is not None:
        y = _regress_out(y, y_partial)

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
        y = + np.random.uniform(-y_jitter, y_jitter, y.size)

    # Get some defaults
    scatter_color = scatter_kws.pop("c", color)
    scatter_color = scatter_kws.pop("color", scatter_color)
    lw = scatter_kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.75)
    lw = scatter_kws.pop("linewidth", lw)

    # Draw the datapoints either as a scatter or point estimate with CIs
    if scatter:
        if x_estimator is None:
            alpha = scatter_kws.pop("alpha", .8)
            ax.scatter(x_scatter, y,
                       color=scatter_color, alpha=alpha, **scatter_kws)
        else:
            if x_bins is None:
                x_discrete = x
            point_data = _point_est(x_discrete, y,
                                    x_estimator, x_ci, units, n_boot)
            for x_val, height, ci_bounds in zip(*point_data):
                size = scatter_kws.pop("s", 50)
                size = scatter_kws.pop("size", size)
                ax.scatter(x_val, height, size, color=scatter_color,
                           **scatter_kws)
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
    if sum((order > 1, logistic, robust, lowess)) > 1:
        raise ValueError("Mutually exclusive regression options were used.")

    # Fit the regression and bootstrap the prediction.
    # This gets delegated to one of several functions depending on the options.
    if order > 1:
        y_hat, y_hat_boots = _regress_poly(grid, x, y, units,
                                           order, ci, n_boot)

    elif logistic:
        binomial = sm.families.Binomial()
        y_hat, y_hat_boots = _regress_statsmodels(grid, x, y, units,
                                                  sm.GLM, ci, n_boot,
                                                  family=binomial)
    elif lowess:
        ci = None
        grid, y_hat = sm.nonparametric.lowess(y, x).T
    elif robust:
        y_hat, y_hat_boots = _regress_statsmodels(grid, x, y, units, sm.RLM,
                                                  ci, n_boot)
    else:
        y_hat, y_hat_boots = _regress_fast(grid, x, y, units, ci, n_boot)

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
