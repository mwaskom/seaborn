"""Plotting functions for visualizing distributions."""
from numbers import Number
from functools import partial
import warnings

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.collections import LineCollection

from ._core import (
    VectorPlotter,
)
from ._statistics import (
    KDE,
)
from .utils import _kde_support, _normalize_kwargs, remove_na
from .palettes import light_palette
from ._decorators import _deprecate_positional_args


__all__ = ["distplot", "kdeplot", "rugplot"]


class _DistributionPlotter(VectorPlotter):

    semantics = "x", "y", "hue"

    wide_structure = {"x": "values", "hue": "columns"}
    flat_structure = {"x": "values"}

    def __init__(
        self,
        data=None,
        variables={},
    ):

        super().__init__(data=data, variables=variables)

    def _add_legend(self, ax, artist, hue_attrs, artist_kws, legend_kws):

        artist_kws = artist_kws.copy()

        if isinstance(hue_attrs, str):
            hue_attrs = [hue_attrs]

        handles = []
        labels = []
        for level in self._hue_map.levels:
            val = self._hue_map(level)
            for attr in hue_attrs:
                artist_kws[attr] = val
            handles.append(artist(**artist_kws))
            labels.append(level)

        ax.legend(handles, labels, title=self.variables["hue"], **legend_kws)


class _HistPlotter(_DistributionPlotter):

    pass


class _KDEPlotter(_DistributionPlotter):

    # TODO we maybe need a different category for variables that do not
    # map to semantics of the plot, like weights
    semantics = _DistributionPlotter.semantics + ("weights",)

    def plot_univariate(
        self,
        hue_method,
        common_norm,
        common_grid,
        fill,
        legend,
        estimate_kws,
        fill_kws,
        line_kws,
        ax,
    ):

        # Preprocess the matplotlib keyward dictionaries
        line_kws = _normalize_kwargs(line_kws, mpl.lines.Line2D)
        fill_kws = _normalize_kwargs(
            fill_kws, mpl.collections.PolyCollection
        )

        # Set shared default values for the matplotlib attributes
        fill_kws.setdefault("alpha", .25)
        fill_kws.setdefault("linewidth", 0)

        # Input checking
        hue_method_options = {"layer", "stack", "fill"}
        if hue_method not in hue_method_options:
            msg = (
                f"hue_method must be one of {hue_method_options}, "
                f"but {hue_method} was passed."
            )
            raise ValueError(msg)

        # Control the interaction with autoscaling by defining sticky_edges
        # i.e. we don't want autoscale margins below the density curve
        sticky_y = (0, 1) if hue_method == "fill" else (0, np.inf)

        # Identify the axis with the data values
        data_variable = {"x", "y"}.intersection(self.variables).pop()

        # Check for log scaling on the data axis
        data_axis = getattr(ax, f"{data_variable}axis")
        log_scale = data_axis.get_scale() == "log"

        # Initialize the estimator object
        estimator = KDE(**estimate_kws)

        if "hue" in self.variables:

            # Access and clean the data
            all_observations = remove_na(self.plot_data[data_variable])

            # Compute the reletive proportion of each hue level
            rows = all_observations.index
            hue_props = (self.plot_data.loc[rows, "hue"]
                         .value_counts(normalize=True))

            # Always share the evaluation grid when stacking
            if hue_method in ("stack", "fill"):
                common_grid = True

            # Define a single grid of support for the PDFs
            if common_grid:
                if log_scale:
                    all_observations = np.log10(all_observations)
                estimator.define_support(all_observations)

        else:

            common_norm = False

        # We will do two loops through the semantic subsets
        # The first is to estimate the density of observations in each subset
        densities = {}

        for sub_vars, sub_data in self._semantic_subsets("hue"):

            # Extract the data points from this sub set and remove nulls
            observations = remove_na(sub_data[data_variable])

            observation_variance = observations.var()
            if not observation_variance or np.isnan(observation_variance):
                msg = "Dataset has 0 variance; skipping density estimate."
                warnings.warn(msg, UserWarning)
                continue

            # Extract the weights for this subset of observations
            if "weights" in self.variables:
                weights = sub_data["weights"]
            else:
                weights = None

            # If data axis is log scaled, fit the KDE in logspace
            if log_scale:
                observations = np.log10(observations)

            # Estimate the density of observations at this level
            density, support = estimator(observations, weights=weights)

            if log_scale:
                support = np.power(10, support)

            # Apply a scaling factor so that the integral over all subsets is 1
            if common_norm:
                density *= hue_props[sub_vars["hue"]]

            # Store the density for this level
            key = tuple(sub_vars.items())
            densities[key] = pd.Series(density, index=support)

        # Implement the density stacking (and filling)
        if hue_method in ("stack", "fill"):

            # The densities share a support grid, so we can make a dataframe
            densities = pd.DataFrame(densities).iloc[:, ::-1]
            norm_constant = densities.sum(axis="columns")

            # Take the cumulative sum to stack
            densities = densities.cumsum(axis="columns")

            # Normalize by row sum to fill
            if hue_method == "fill":
                densities = densities.div(norm_constant, axis="index")

            # Define where each segment starts
            baselines = densities.shift(1, axis=1).fillna(0)

        else:

            # All densities will start at 0
            baselines = {k: np.zeros_like(v) for k, v in densities.items()}

        # Filled plots should not have any margins
        if hue_method == "fill":
            sticky_x = densities.index.min(), densities.index.max()
        else:
            sticky_x = []

        # Now iterate through again and draw the densities
        # We go backwards so stacked densities read from top-to-bottom
        for sub_vars, _ in self._semantic_subsets("hue", reverse=True):

            # Extract the support grid and density curve for this level
            key = tuple(sub_vars.items())
            try:
                density = densities[key]
            except KeyError:
                continue
            support = density.index
            fill_from = baselines[key]

            # Modify the matplotlib attributes from semantic mapping
            if "hue" in self.variables:
                line_kws["color"] = self._hue_map(sub_vars["hue"])
                fill_kws["facecolor"] = self._hue_map(sub_vars["hue"])

            # Plot a curve with observation values on the x axis
            if "x" in self.variables:

                line, = ax.plot(support, density, **line_kws)
                line.sticky_edges.x[:] = sticky_x
                line.sticky_edges.y[:] = sticky_y

                if fill:
                    fill_kws.setdefault("facecolor", line.get_color())
                    fill = ax.fill_between(
                        support, fill_from, density, **fill_kws
                    )
                    fill.sticky_edges.x[:] = sticky_x
                    fill.sticky_edges.y[:] = sticky_y

            # Plot a curve with observation values on the y axis
            else:

                line, = ax.plot(density, support, **line_kws)
                line.sticky_edges.x[:] = sticky_y
                line.sticky_edges.y[:] = sticky_x

                if fill:
                    fill_kws.setdefault("facecolor", line.get_color())
                    fill = ax.fill_between(
                        density, fill_from, support, **fill_kws
                    )
                    fill.sticky_edges.x[:] = sticky_y
                    fill.sticky_edges.y[:] = sticky_x

        # --- Finalize the plot ----
        default_x = default_y = ""
        if data_variable == "x":
            default_y = "Density"
        if data_variable == "y":
            default_x = "Density"
        self._add_axis_labels(ax, default_x, default_y)

        if "hue" in self.variables and legend:

            # TODO what i would like in the case of shaded densities
            # is to have the legend artist be filled at the alpha of
            # the fill between and with an edge that looks like either the
            # line or the line of the fill
            # This is possible to hack here, or we might want to just draw
            # the shaded densities with a fill_between artist.
            # I am punting for now

            fill_kws = fill_kws.copy()
            if fill:
                artist = partial(mpl.patches.Patch)
                hue_attrs = ["facecolor", "edgecolor"]
                artist_kws = fill_kws
            else:
                artist = partial(mpl.lines.Line2D, [], [])
                hue_attrs = "color"
                artist_kws = line_kws

            self._add_legend(
                ax, artist, hue_attrs, artist_kws, {},
            )

    def plot_bivariate(
        self,
        common_norm,
        fill,
        levels,
        thresh,
        legend,
        log_scale,
        color,
        cbar,
        cbar_ax,
        cbar_kws,
        estimate_kws,
        contour_kws,
        ax,
    ):

        contour_kws = contour_kws.copy()

        estimator = KDE(**estimate_kws)

        if "hue" in self.variables:

            # Compute the reletive proportion of each hue level
            rows = remove_na(self.plot_data[["x", "y"]]).index
            hue_props = (self.plot_data.loc[rows, "hue"]
                         .value_counts(normalize=True))

        else:

            common_norm = False

        # Loop through the subsets and estimate the KDEs
        densities, supports = {}, {}

        for sub_vars, sub_data in self._semantic_subsets("hue"):

            # Extract the data points from this sub set and remove nulls
            observations = remove_na(sub_data[["x", "y"]])

            observation_variance = observations.var().any()
            if not observation_variance or np.isnan(observation_variance):
                msg = "Dataset has 0 variance; skipping density estimate."
                warnings.warn(msg, UserWarning)
                continue

            # Extract the weights for this subset of observations
            if "weights" in self.variables:
                weights = sub_data["weights"]
            else:
                weights = None

            # If data axis is log scaled, fit the KDE in logspace
            if log_scale is not None:
                if log_scale[0]:
                    observations["x"] = np.log10(observations["x"])
                if log_scale[1]:
                    observations["y"] = np.log10(observations["y"])

            # Check that KDE will not error out
            variance = observations[["x", "y"]].var()
            if not variance.all() or variance.isna().any():
                msg = "Dataset has 0 variance; skipping density estimate."
                warnings.warn(msg, UserWarning)
                continue

            # Estimate the density of observations at this level
            observations = observations["x"], observations["y"]
            density, support = estimator(*observations, weights=weights)

            # Transform the support grid back to the original scale
            if log_scale is not None:
                xx, yy = support
                if log_scale[0]:
                    xx = np.power(10, xx)
                if log_scale[1]:
                    yy = np.power(10, yy)
                support = xx, yy

            # Apply a scaling factor so that the integral over all subsets is 1
            if common_norm:
                density *= hue_props[sub_vars["hue"]]

            key = tuple(sub_vars.items())
            densities[key] = density
            supports[key] = support

        # Define a grid of iso-proportion levels
        if isinstance(levels, Number):
            levels = np.linspace(thresh, 1, levels)
        else:
            if min(levels) < 0 or max(levels) > 1:
                raise ValueError("levels must be in [0, 1]")

        # Transfrom from iso-proportions to iso-densities
        if common_norm:
            common_levels = self._find_contour_levels(
                list(densities.values()), levels,
            )
            draw_levels = {k: common_levels for k in densities}
        else:
            draw_levels = {
                k: self._find_contour_levels(d, levels)
                for k, d in densities.items()
            }

        # Get a default single color from the attribute cycle
        scout, = ax.plot([], color=color)
        default_color = scout.get_color()
        scout.remove()

        # Apply a common color-mapping to single color specificiations
        color_map = partial(light_palette, reverse=True, as_cmap=True)

        # Define the coloring of the contours
        if "hue" in self.variables:
            for param in ["cmap", "colors"]:
                if param in contour_kws:
                    msg = f"{param} parameter ignored when using hue mapping."
                    warnings.warn(msg, UserWarning)
                    contour_kws.pop(param)
        else:
            coloring_given = set(contour_kws) & {"cmap", "colors"}
            if fill and not coloring_given:
                cmap = color_map(default_color)
                contour_kws["cmap"] = cmap
            if not fill and not coloring_given:
                contour_kws["colors"] = [default_color]

        # Choose the function to plot with
        # TODO could add a pcolormesh based option as well
        if fill:
            contour_func = ax.contourf
        else:
            contour_func = ax.contour

        # Loop through the subsets again and plot the data
        for sub_vars, _ in self._semantic_subsets("hue"):

            if "hue" in sub_vars:
                color = self._hue_map(sub_vars["hue"])
                if fill:
                    contour_kws["cmap"] = color_map(color)
                else:
                    contour_kws["colors"] = [color]

            key = tuple(sub_vars.items())
            if key not in densities:
                continue
            density = densities[key]
            xx, yy = supports[key]

            label = contour_kws.pop("label", None)

            cset = contour_func(
                xx, yy, density,
                levels=draw_levels[key],
                **contour_kws,
            )

            if "hue" not in self.variables:
                cset.collections[0].set_label(label)

        # Add a color bar representing the contour heights
        # Note: this shows iso densities, not iso proportions
        if cbar:
            # TODO what to do about hue here?
            # TODO maybe use the legend instead?
            cbar_kws = {} if cbar_kws is None else cbar_kws
            ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

        # --- Finalize the plot
        self._add_axis_labels(ax)

        if "hue" in self.variables and legend:

            # TODO if possible, I would like to move the contour
            # intensity information into the legend too and label the
            # iso proportions rather than the raw density values

            artist_kws = {}
            if fill:
                artist = partial(mpl.patches.Patch)
                hue_attrs = "facecolor", "edgecolor"
            else:
                artist = partial(mpl.lines.Line2D, [], [])
                hue_attrs = "color"

            self._add_legend(
                ax, artist, hue_attrs, artist_kws, {},
            )

    def _find_contour_levels(self, density, isoprop):
        """Return contour levels to draw density at given iso-propotions."""
        isoprop = np.asarray(isoprop)
        values = np.ravel(density)
        sorted_values = np.sort(values)[::-1]
        normalized_values = np.cumsum(sorted_values) / values.sum()
        idx = np.searchsorted(normalized_values, 1 - isoprop)
        levels = np.take(sorted_values, idx, mode="clip")
        return levels


@_deprecate_positional_args
def kdeplot(
    x=None,  # Allow positional x, because behavior will not change with reorg
    *,
    y=None,
    shade=None,  # Note "soft" deprecation, explained below
    vertical=False,  # Deprecated
    kernel=None,  # Deprecated
    bw=None,  # Deprecated
    gridsize=200,  # TODO maybe depend on uni/bivariate?
    cut=3, clip=None, legend=True, cumulative=False,
    shade_lowest=None,  # Deprecated, controlled with levels now
    cbar=False, cbar_ax=None, cbar_kws=None,
    ax=None,

    # New params
    weights=None,  # TODO note that weights is grouped with semantics
    hue=None, palette=None, hue_order=None, hue_norm=None, hue_method="layer",
    common_norm=True, common_grid=False,
    levels=10, thresh=.05,  # TODO rethink names
    bw_method="scott", bw_adjust=1, log_scale=None,
    color=None, fill=None, fill_kws=None,

    # Renamed params
    data=None, data2=None,

    **kwargs,
):

    # Handle deprecation of `data2` as name for y variable
    if data2 is not None:

        y = data2

        # If `data2` is present, we need to check for the `data` kwarg being
        # used to pass a vector for `x`. We'll reassign the vectors and warn.
        # We need this check because just passing a vector to `data` is now
        # technically valid.

        x_passed_as_data = (
            x is None
            and data is not None
            and np.ndim(data) == 1
        )

        if x_passed_as_data:
            msg = "Use `x` and `y` rather than `data` `and `data2`"
            x = data
        else:
            msg = "The `data2` param is now named `y`; please update your code"

        warnings.warn(msg, FutureWarning)

    # Handle deprecation of `vertical`
    if vertical:
        msg = (
            "The `vertical` parameter is deprecated and will be removed in a "
            "future version. Assign the data to the `y` variable instead."
        )
        warnings.warn(msg, FutureWarning)
        x, y = y, x

    # Handle deprecation of `bw`
    if bw is not None:
        msg = (
            "The `bw` parameter is deprecated in favor of `bw_method` and "
            f"`bw_adjust`. Using {bw} for `bw_method`, but please "
            "see the docs for the new parameters and update your code."
        )
        warnings.warn(msg, FutureWarning)
        bw_method = bw

    # Handle deprecation of `kernel`
    if kernel is not None:
        msg = (
            "Support for alternate kernels has been removed. "
            "Using Gaussian kernel."
        )
        warnings.warn(msg, UserWarning)

    # Handle deprecation of shade_lowest
    # TODO check that we still use `thresh` before merging
    if shade_lowest is not None:
        if shade_lowest:
            thresh = 0
        msg = (
            "`shade_lowest` is now deprecated in favor of `thresh`. "
            f"Setting `thresh={thresh}`, but please update your code."
        )
        warnings.warn(msg, UserWarning)

    # Handle `n_levels`
    # This was never in the formal API but it was processed, and appeared in an
    # example. We can treat as an alias for `levels` now and deprecate later.
    levels = kwargs.pop("n_levels", levels)

    # Handle "soft" deprecation of shade `shade` is not really the right
    # terminology here, but unlike some of the other deprecated parameters it
    # is probably very commonly used and much hard to remove. This is therefore
    # going to be a longer process where, first, `fill` will be introduced and
    # be used throughout the documentation. In 0.12, when kwarg-only
    # enforcement hits, we can remove the shade/shade_lowest out of the
    # function signature all together and pull them out of the kwargs. Then we
    # can actually fire a FutureWarning, and eventually remove.
    if shade is not None:
        fill = shade

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    p = _KDEPlotter(
        data=data,
        variables=_KDEPlotter.get_semantics(locals()),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if fill_kws is None:
        fill_kws = {}

    if ax is None:
        ax = plt.gca()

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    # Check for a specification that lacks x/y data and return early
    any_data = bool({"x", "y"} & set(p.variables))
    if not any_data:
        return ax

    # Determine the kind of plot to use
    univariate = bool({"x", "y"} - set(p.variables))

    if univariate:

        data_variable = (set(p.variables) & {"x", "y"}).pop()

        # Catch some inputs we cannot do anything with
        data_var_type = p.var_types[data_variable]
        if data_var_type != "numeric":
            msg = (
                f"kdeplot requires a numeric '{data_variable}' variable, "
                f"but a {data_var_type} was passed."
            )
            raise TypeError(msg)

        # Possibly log scale the data axis
        if log_scale is not None:
            set_scale = getattr(ax, f"set_{data_variable}scale")
            if log_scale is True:
                set_scale("log")
            else:
                set_scale("log", **{f"base{data_variable}": log_scale})

        # Set defaults that depend on other parameters
        if fill is None:
            fill = hue_method in ("stack", "fill")

        kwargs["color"] = color

        p.plot_univariate(
            hue_method=hue_method,
            common_norm=common_norm,
            common_grid=common_grid,
            fill=fill,
            legend=legend,
            estimate_kws=estimate_kws,
            fill_kws=fill_kws,
            line_kws=kwargs,
            ax=ax
        )

    else:

        # TODO input checking?

        # Possibly log-scale one or both axes
        if log_scale is not None:
            # Allow single value or x, y tuple
            try:
                scalex, scaley = log_scale
            except TypeError:
                scalex = scaley = log_scale
                log_scale = scalex, scaley  # Tupelize for downstream

            for axis, scale in zip("xy", (scalex, scaley)):
                if scale:
                    set_scale = getattr(ax, f"set_{axis}scale")
                    if scale is True:
                        set_scale("log")
                    else:
                        set_scale("log", **{f"base{axis}": scale})

        p.plot_bivariate(
            common_norm=common_norm,
            fill=fill,
            levels=levels,
            thresh=thresh,
            legend=legend,
            log_scale=log_scale,
            color=color,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            estimate_kws=estimate_kws,
            contour_kws=kwargs,
            ax=ax,
        )

    return ax


class _RugPlotter(_DistributionPlotter):

    def __init__(
        self,
        data=None,
        variables={},
    ):

        super().__init__(data=data, variables=variables)

    def plot(self, height, expand_margins, legend, ax, kws):

        kws = _normalize_kwargs(kws, mpl.lines.Line2D)

        # TODO we need to abstract this logic
        scout, = ax.plot([], [], **kws)
        kws["color"] = kws.pop("color", scout.get_color())
        scout.remove()

        kws.setdefault("linewidth", 1)

        if expand_margins:
            xmarg, ymarg = ax.margins()
            if "x" in self.variables:
                ymarg += height * 2
            if "y" in self.variables:
                xmarg += height * 2
            ax.margins(x=xmarg, y=ymarg)

        if "hue" in self.variables:
            kws.pop("c", None)
            kws.pop("color", None)

        if "x" in self.variables:
            self._plot_single_rug("x", height, ax, kws)
        if "y" in self.variables:
            self._plot_single_rug("y", height, ax, kws)

        # --- Finalize the plot
        self._add_axis_labels(ax)
        if "hue" in self.variables and legend:
            # TODO ideally i'd like the legend artist to look like a rug
            legend_artist = partial(mpl.lines.Line2D, [], [])
            self._add_legend(
                ax, legend_artist, "color", {}, {},
            )

    def _plot_single_rug(self, var, height, ax, kws):
        """Draw a rugplot along one axis of the plot."""
        vector = self.plot_data[var]
        n = len(vector)

        # We'll always add a single collection with varying colors
        if "hue" in self.variables:
            colors = self._hue_map(self.plot_data["hue"])
        else:
            colors = None

        # Build the array of values for the LineCollection
        if var == "x":

            trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
            xy_pairs = np.column_stack([
                np.repeat(vector, 2), np.tile([0, height], n)
            ])

        if var == "y":

            trans = tx.blended_transform_factory(ax.transAxes, ax.transData)
            xy_pairs = np.column_stack([
                np.tile([0, height], n), np.repeat(vector, 2)
            ])

        # Draw the lines on the plot
        line_segs = xy_pairs.reshape([n, 2, 2])
        ax.add_collection(LineCollection(
            line_segs, transform=trans, colors=colors, **kws
        ))

        ax.autoscale_view(scalex=var == "x", scaley=var == "y")


@_deprecate_positional_args
def rugplot(
    x=None,  # Allow positional x, because behavior won't change
    *,
    height=.025, axis=None, ax=None,

    # New parameters
    data=None, y=None, hue=None,
    palette=None, hue_order=None, hue_norm=None,
    expand_margins=True,
    legend=True,  # TODO or maybe default to False?

    # Renamed parameter
    a=None,

    **kwargs
):

    # A note: if we want to add a style semantic to rugplot,
    # we could make an option that draws the rug using scatterplot

    # Handle deprecation of `a``
    if a is not None:
        msg = "The `a` parameter is now called `x`. Please update your code."
        warnings.warn(msg, FutureWarning)
        x = a
        del a

    # Handle deprecation of "axis"
    if axis is not None:
        msg = (
            "The `axis` variable is no longer used and will be removed. "
            "Instead, assign variables directly to `x` or `y`."
        )
        warnings.warn(msg, FutureWarning)

    # Handle deprecation of "vertical"
    if kwargs.pop("vertical", axis == "y"):
        x, y = None, x
        msg = (
            "Using `vertical=True` to control the orientation of the plot  "
            "is deprecated. Instead, assign the data directly to `y`. "
        )
        warnings.warn(msg, FutureWarning)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    variables = _RugPlotter.get_semantics(locals())

    p = _RugPlotter(data=data, variables=variables)
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is None:
        ax = plt.gca()

    p.plot(height, expand_margins, legend, ax, kwargs)

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
        a = x

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
