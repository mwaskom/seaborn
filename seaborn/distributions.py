"""Plotting functions for visualizing distributions."""
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.collections import LineCollection
import warnings

try:
    import statsmodels.nonparametric.api as smnp
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

from ._core import (
    VectorPlotter,
)
from .utils import _kde_support, remove_na
from .palettes import color_palette, light_palette, dark_palette, blend_palette
from ._decorators import _deprecate_positional_args


__all__ = ["distplot", "kdeplot", "rugplot"]


class _DistributionPlotter(VectorPlotter):

    semantics = "x", "y", "hue"

    wide_structure = {
        "x": "values", "hue": "columns",
    }


class _HistPlotter(_DistributionPlotter):

    pass


class _KDEPlotter(_DistributionPlotter):

    pass


class _RugPlotter(_DistributionPlotter):

    def __init__(
        self,
        data=None,
        variables={},
        height=None,
    ):

        super().__init__(data=data, variables=variables)

        self.height = height

    def plot(self, ax, kws):

        # TODO we need to abstract this logic
        scout, = ax.plot([], [], **kws)

        kws = kws.copy()
        kws["color"] = kws.pop("color", scout.get_color())

        scout.remove()

        # TODO handle more gracefully
        alias_map = dict(linewidth="lw", linestyle="ls", color="c")
        for attr, alias in alias_map.items():
            if alias in kws:
                kws[attr] = kws.pop(alias)
        kws.setdefault("linewidth", 1)

        # ---

        # TODO expand the plot margins to account for the height of
        # the rug (as an option?)

        # ---

        if "x" in self.variables:
            self._plot_single_rug("x", ax, kws)
        if "y" in self.variables:
            self._plot_single_rug("y", ax, kws)

    def _plot_single_rug(self, var, ax, kws):

        vector = self.plot_data[var]
        n = len(vector)

        if "hue" in self.variables:
            colors = self._hue_map(self.plot_data["hue"])
            kws.pop("color", None)  # TODO simplify
        else:
            colors = None

        if var == "x":

            trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
            xy_pairs = np.column_stack([
                np.repeat(vector, 2), np.tile([0, self.height], n)
            ])

        if var == "y":

            trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
            xy_pairs = np.column_stack([
                np.tile([0, self.height], n), np.repeat(vector, 2)
            ])

        line_segs = xy_pairs.reshape([n, 2, 2])
        ax.add_collection(LineCollection(
            line_segs, transform=trans, colors=colors, **kws
        ))

        ax.autoscale_view(scalex=var == "x", scaley=var == "y")


@_deprecate_positional_args
def _new_rugplot(
    *,
    x=None,
    height=.05, axis="x", ax=None,
    data=None, y=None, hue=None,
    palette=None, hue_order=None, hue_norm=None,
    a=None,
    **kwargs
):

    # Handle deprecation of `a``
    if a is not None:
        msg = "The `a` parameter is now called `x`. Please update your code."
        warnings.warn(msg, FutureWarning)
        x = a
        del a

    # TODO Handle deprecation of "axis"
    # TODO Handle deprecation of "vertical"
    if kwargs.pop("vertical", axis == "y"):
        x, y = None, x

    # ----------

    variables = _RugPlotter.get_variables(locals())

    p = _RugPlotter(
        data=data,
        variables=variables,
        height=height,
    )
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * stats.iqr(a) / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


@_deprecate_positional_args
def distplot(
    *,
    x=None,
    bins=None, hist=True, kde=True, rug=False, fit=None,
    hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
    color=None, vertical=False, norm_hist=False, axlabel=None,
    label=None, ax=None, a=None,
):
    """Flexibly plot a univariate distribution of observations.

    This function combines the matplotlib ``hist`` function (with automatic
    calculation of a good default bin size) with the seaborn :func:`kdeplot`
    and :func:`rugplot` functions. It can also fit ``scipy.stats``
    distributions and plot the estimated PDF over the data.

    Parameters
    ----------

    x : Series, 1d-array, or list.
        Observed data. If this is a Series object with a ``name`` attribute,
        the name will be used to label the data axis.
    bins : argument for matplotlib hist(), or None, optional
        Specification of hist bins. If unspecified, as reference rule is used
        that tries to find a useful default.
    hist : bool, optional
        Whether to plot a (normed) histogram.
    kde : bool, optional
        Whether to plot a gaussian kernel density estimate.
    rug : bool, optional
        Whether to draw a rugplot on the support axis.
    fit : random variable object, optional
        An object with `fit` method, returning a tuple that can be passed to a
        `pdf` method a positional arguments following a grid of values to
        evaluate the pdf on.
    hist_kws : dict, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.hist`.
    kde_kws : dict, optional
        Keyword arguments for :func:`kdeplot`.
    rug_kws : dict, optional
        Keyword arguments for :func:`rugplot`.
    color : matplotlib color, optional
        Color to plot everything but the fitted curve in.
    vertical : bool, optional
        If True, observed values are on y-axis.
    norm_hist : bool, optional
        If True, the histogram height shows a density rather than a count.
        This is implied if a KDE or fitted density is plotted.
    axlabel : string, False, or None, optional
        Name for the support axis label. If None, will try to get it
        from a.name if False, do not set a label.
    label : string, optional
        Legend label for the relevant component of the plot.
    ax : matplotlib axis, optional
        If provided, plot on this axis.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    See Also
    --------
    kdeplot : Show a univariate or bivariate distribution with a kernel
              density estimate.
    rugplot : Draw small vertical lines to show each observation in a
              distribution.

    Examples
    --------

    Show a default plot with a kernel density estimate and histogram with bin
    size determined automatically with a reference rule:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns, numpy as np
        >>> sns.set(); np.random.seed(0)
        >>> x = np.random.randn(100)
        >>> ax = sns.distplot(x=x)

    Use Pandas objects to get an informative axis label:

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> x = pd.Series(x, name="x variable")
        >>> ax = sns.distplot(x=x)

    Plot the distribution with a kernel density estimate and rug plot:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x=x, rug=True, hist=False)

    Plot the distribution with a histogram and maximum likelihood gaussian
    distribution fit:

    .. plot::
        :context: close-figs

        >>> from scipy.stats import norm
        >>> ax = sns.distplot(x=x, fit=norm, kde=False)

    Plot the distribution on the vertical axis:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x=x, vertical=True)

    Change the color of all the plot elements:

    .. plot::
        :context: close-figs

        >>> sns.set_color_codes()
        >>> ax = sns.distplot(x=x, color="y")

    Pass specific parameters to the underlying plot functions:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x=x, rug=True, rug_kws={"color": "g"},
        ...                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
        ...                   hist_kws={"histtype": "step", "linewidth": 3,
        ...                             "alpha": 1, "color": "g"})

    """
    # Handle deprecation of ``a```
    if a is not None:
        msg = "The `a` parameter is now called `x`. Please update your code."
        warnings.warn(msg)
    else:
        a = x  # TODO refactor

    # Default to drawing on the currently-active axes
    if ax is None:
        ax = plt.gca()

    # Intelligently label the support axis
    label_ax = bool(axlabel)
    if axlabel is None and hasattr(a, "name"):
        axlabel = a.name
        if axlabel is not None:
            label_ax = True

    # Make a a 1-d float array
    a = np.asarray(a, np.float)
    if a.ndim > 1:
        a = a.squeeze()

    # Drop null values from array
    a = remove_na(a)

    # Decide if the hist is normed
    norm_hist = norm_hist or kde or (fit is not None)

    # Handle dictionary defaults
    hist_kws = {} if hist_kws is None else hist_kws.copy()
    kde_kws = {} if kde_kws is None else kde_kws.copy()
    rug_kws = {} if rug_kws is None else rug_kws.copy()
    fit_kws = {} if fit_kws is None else fit_kws.copy()

    # Get the color from the current color cycle
    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()

    # Plug the label into the right kwarg dictionary
    if label is not None:
        if hist:
            hist_kws["label"] = label
        elif kde:
            kde_kws["label"] = label
        elif rug:
            rug_kws["label"] = label
        elif fit:
            fit_kws["label"] = label

    if hist:
        if bins is None:
            bins = min(_freedman_diaconis_bins(a), 50)
        hist_kws.setdefault("alpha", 0.4)
        hist_kws.setdefault("density", norm_hist)

        orientation = "horizontal" if vertical else "vertical"
        hist_color = hist_kws.pop("color", color)
        ax.hist(a, bins, orientation=orientation,
                color=hist_color, **hist_kws)
        if hist_color != color:
            hist_kws["color"] = hist_color

    if kde:
        kde_color = kde_kws.pop("color", color)
        kdeplot(x=a, vertical=vertical, ax=ax, color=kde_color, **kde_kws)
        if kde_color != color:
            kde_kws["color"] = kde_color

    if rug:
        rug_color = rug_kws.pop("color", color)
        axis = "y" if vertical else "x"
        rugplot(x=a, axis=axis, ax=ax, color=rug_color, **rug_kws)
        if rug_color != color:
            rug_kws["color"] = rug_color

    if fit is not None:

        def pdf(x):
            return fit.pdf(x, *params)

        fit_color = fit_kws.pop("color", "#282828")
        gridsize = fit_kws.pop("gridsize", 200)
        cut = fit_kws.pop("cut", 3)
        clip = fit_kws.pop("clip", (-np.inf, np.inf))
        bw = stats.gaussian_kde(a).scotts_factor() * a.std(ddof=1)
        x = _kde_support(a, bw, gridsize, cut, clip)
        params = fit.fit(a)
        y = pdf(x)
        if vertical:
            x, y = y, x
        ax.plot(x, y, color=fit_color, **fit_kws)
        if fit_color != "#282828":
            fit_kws["color"] = fit_color

    if label_ax:
        if vertical:
            ax.set_ylabel(axlabel)
        else:
            ax.set_xlabel(axlabel)

    return ax


def _univariate_kdeplot(data, shade, vertical, kernel, bw, gridsize, cut,
                        clip, legend, ax, cumulative=False, **kwargs):
    """Plot a univariate kernel density estimate on one of the axes."""

    # Sort out the clipping
    if clip is None:
        clip = (-np.inf, np.inf)

    # Preprocess the data
    data = remove_na(data)

    # Calculate the KDE

    if np.nan_to_num(data.var()) == 0:
        # Don't try to compute KDE on singular data
        msg = "Data must have variance to compute a kernel density estimate."
        warnings.warn(msg, UserWarning)
        x, y = np.array([]), np.array([])

    elif _has_statsmodels:
        # Prefer using statsmodels for kernel flexibility
        x, y = _statsmodels_univariate_kde(data, kernel, bw,
                                           gridsize, cut, clip,
                                           cumulative=cumulative)
    else:
        # Fall back to scipy if missing statsmodels
        if kernel != "gau":
            kernel = "gau"
            msg = "Kernel other than `gau` requires statsmodels."
            warnings.warn(msg, UserWarning)
        if cumulative:
            raise ImportError("Cumulative distributions are currently "
                              "only implemented in statsmodels. "
                              "Please install statsmodels.")
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
    legend = label is not None and legend
    label = "_nolegend_" if label is None else label

    # Use the active color cycle to find the plot color
    facecolor = kwargs.pop("facecolor", None)
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)
    facecolor = color if facecolor is None else facecolor

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    shade_kws = dict(
        facecolor=facecolor,
        alpha=kwargs.get("alpha", 0.25),
        clip_on=kwargs.get("clip_on", True),
        zorder=kwargs.get("zorder", 1),
    )
    if shade:
        if vertical:
            ax.fill_betweenx(y, 0, x, **shade_kws)
        else:
            ax.fill_between(x, 0, y, **shade_kws)

    # Set the density axis minimum to 0
    if vertical:
        ax.set_xlim(0, auto=None)
    else:
        ax.set_ylim(0, auto=None)

    # Draw the legend here
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        ax.legend(loc="best")

    return ax


def _statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip,
                                cumulative=False):
    """Compute a univariate kernel density estimate using statsmodels."""
    # statsmodels 0.8 fails on int type data
    data = data.astype(np.float64)

    fft = kernel == "gau"
    kde = smnp.KDEUnivariate(data)

    try:
        kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    except RuntimeError as err:  # GH#1990
        if stats.iqr(data) > 0:
            raise err
        msg = "Default bandwidth for data is 0; skipping density estimation."
        warnings.warn(msg, UserWarning)
        return np.array([]), np.array([])

    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density
    return grid, y


def _scipy_univariate_kde(data, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using scipy."""
    kde = stats.gaussian_kde(data, bw_method=bw)
    if isinstance(bw, str):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)() * np.std(data)
    grid = _kde_support(data, bw, gridsize, cut, clip)
    y = kde(grid)
    return grid, y


def _bivariate_kdeplot(x, y, filled, fill_lowest,
                       kernel, bw, gridsize, cut, clip,
                       axlabel, cbar, cbar_ax, cbar_kws, ax, **kwargs):
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    # Calculate the KDE
    if _has_statsmodels:
        xx, yy, z = _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
    else:
        xx, yy, z = _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)

    scout, = ax.plot([], [])
    default_color = scout.get_color()
    scout.remove()

    cmap = kwargs.pop("cmap", None)
    color = kwargs.pop("color", None)
    if cmap is None and "colors" not in kwargs:
        if color is None:
            color = default_color
        if filled:
            cmap = light_palette(color, as_cmap=True)
        else:
            cmap = dark_palette(color, as_cmap=True)
    if isinstance(cmap, str):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
        else:
            cmap = mpl.cm.get_cmap(cmap)

    label = kwargs.pop("label", None)

    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx, yy, z, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels

    if cbar:
        cbar_kws = {} if cbar_kws is None else cbar_kws
        ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    if label is not None:
        legend_color = cmap(.95) if color is None else color
        if filled:
            ax.fill_between([], [], color=legend_color, label=label)
        else:
            ax.plot([], [], color=legend_color, label=label)

    return ax


def _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels."""
    # statsmodels 0.8 fails on int type data
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if isinstance(bw, str):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T, bw_method=bw)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, str):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = ("Cannot specify a different bandwidth for each dimension "
               "with the scipy backend. You should install statsmodels.")
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


@_deprecate_positional_args
def kdeplot(
    *,
    x=None, y=None,
    shade=False, vertical=False, kernel="gau",
    bw="scott", gridsize=100, cut=3, clip=None, legend=True,
    cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None,
    cbar_kws=None, ax=None,
    data=None, data2=None,  # TODO move data once * is enforced
    **kwargs,
):
    """Fit and plot a univariate or bivariate kernel density estimate.

    Parameters
    ----------
    x : 1d array-like
        Input data.
    y: 1d array-like, optional
        Second input data. If present, a bivariate KDE will be estimated.
    shade : bool, optional
        If True, shade in the area under the KDE curve (or draw with filled
        contours when data is bivariate).
    vertical : bool, optional
        If True, density is on x-axis.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with. Bivariate KDE can only use
        gaussian kernel.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor,
        or scalar for each dimension of the bivariate plot. Note that the
        underlying computational libraries have different interperetations
        for this parameter: ``statsmodels`` uses it directly, but ``scipy``
        treats it as a scaling factor for the standard deviation of the
        data.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    legend : bool, optional
        If True, add a legend or label the axes when possible.
    cumulative : bool, optional
        If True, draw the cumulative distribution estimated by the kde.
    shade_lowest : bool, optional
        If True, shade the lowest contour of a bivariate KDE plot. Not
        relevant when drawing a univariate plot or when ``shade=False``.
        Setting this to ``False`` can be useful when you want multiple
        densities on the same Axes.
    cbar : bool, optional
        If True and drawing a bivariate KDE plot, add a colorbar.
    cbar_ax : matplotlib axes, optional
        Existing axes to draw the colorbar onto, otherwise space is taken
        from the main axes.
    cbar_kws : dict, optional
        Keyword arguments for ``fig.colorbar()``.
    ax : matplotlib axes, optional
        Axes to plot on, otherwise uses current axes.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``plt.plot()`` or
        ``plt.contour{f}`` depending on whether a univariate or bivariate
        plot is being drawn.

    Returns
    -------
    ax : matplotlib Axes
        Axes with plot.

    See Also
    --------
    distplot: Flexibly plot a univariate distribution of observations.
    jointplot: Plot a joint dataset with bivariate and marginal distributions.

    Examples
    --------

    Plot a basic univariate density:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(10)
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> mean, cov = [0, 2], [(1, .5), (.5, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, size=50).T
        >>> ax = sns.kdeplot(x=x)

    Shade under the density curve and use a different color:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, shade=True, color="r")

    Plot a bivariate density:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, y=y)

    Use filled contours:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, y=y, shade=True)

    Use more contour levels and a different color palette:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, y=y, n_levels=30, cmap="Purples_d")

    Use a narrower bandwith:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, bw=.15)

    Plot the density on the vertical axis:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=y, vertical=True)

    Limit the density curve within the range of the data:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, cut=0)

    Add a colorbar for the contours:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x=x, y=y, cbar=True)

    Plot two shaded bivariate densities:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> setosa = iris.loc[iris.species == "setosa"]
        >>> virginica = iris.loc[iris.species == "virginica"]
        >>> ax = sns.kdeplot(x=setosa.sepal_width, y=setosa.sepal_length,
        ...                  cmap="Reds", shade=True, shade_lowest=False)
        >>> ax = sns.kdeplot(x=virginica.sepal_width, y=virginica.sepal_length,
        ...                  cmap="Blues", shade=True, shade_lowest=False)

    """
    # Handle deprecation of `data` as name for x variable
    # TODO this can be removed once refactored to do centralized preprocessing
    # of input variables, because a vector input to `data` will be treated like
    # an input to `x`. Warning is probably not necessary.
    x_passed_as_data = (
        x is None
        and data is not None
        and np.ndim(data) == 1
    )
    if x_passed_as_data:
        x = data

    # Handle deprecation of `data2` as name for y variable
    if data2 is not None:
        msg = "The `data2` param is now named `y`; please update your code."
        warnings.warn(msg)
        y = data2

    # TODO replace this preprocessing with central refactoring
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(y, list):
        y = np.asarray(y)

    bivariate = x is not None and y is not None
    if bivariate and cumulative:
        raise TypeError("Cumulative distribution plots are not"
                        "supported for bivariate distributions.")

    if ax is None:
        ax = plt.gca()

    if bivariate:
        ax = _bivariate_kdeplot(x, y, shade, shade_lowest,
                                kernel, bw, gridsize, cut, clip, legend,
                                cbar, cbar_ax, cbar_kws, ax, **kwargs)
    else:
        ax = _univariate_kdeplot(x, shade, vertical, kernel, bw,
                                 gridsize, cut, clip, legend, ax,
                                 cumulative=cumulative, **kwargs)

    return ax


@_deprecate_positional_args
def rugplot(
    *,
    x=None,
    height=.05, axis="x", ax=None,
    a=None,
    **kwargs
):
    """Plot datapoints in an array as sticks on an axis.

    Parameters
    ----------
    x : vector
        1D array of observations.
    height : scalar, optional
        Height of ticks as proportion of the axis.
    axis : {'x' | 'y'}, optional
        Axis to draw rugplot on.
    ax : matplotlib axes, optional
        Axes to draw plot into; otherwise grabs current axes.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``LineCollection``.

    Returns
    -------
    ax : matplotlib axes
        The Axes object with the plot on it.

    """
    # Handle deprecation of ``a```
    if a is not None:
        msg = "The `a` parameter is now called `x`. Please update your code."
        warnings.warn(msg, FutureWarning)
    else:
        a = x  # TODO refactor

    # Default to drawing on the currently active axes
    if ax is None:
        ax = plt.gca()
    a = np.asarray(a)
    vertical = kwargs.pop("vertical", axis == "y")

    alias_map = dict(linewidth="lw", linestyle="ls", color="c")
    for attr, alias in alias_map.items():
        if alias in kwargs:
            kwargs[attr] = kwargs.pop(alias)
    kwargs.setdefault("linewidth", 1)

    if vertical:
        trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
        xy_pairs = np.column_stack([np.tile([0, height], len(a)),
                                    np.repeat(a, 2)])
    else:
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        xy_pairs = np.column_stack([np.repeat(a, 2),
                                    np.tile([0, height], len(a))])
    line_segs = xy_pairs.reshape([len(a), 2, 2])
    ax.add_collection(LineCollection(line_segs, transform=trans, **kwargs))

    ax.autoscale_view(scalex=not vertical, scaley=vertical)

    return ax
