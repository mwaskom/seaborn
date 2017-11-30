from __future__ import division
from itertools import product
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .external.six import string_types

from . import utils
from .utils import categorical_order, get_color_cycle, sort_df
from .algorithms import bootstrap
from .palettes import color_palette


__all__ = ["lineplot"]


class _BasicPlotter(object):

    # TODO use different lists for mpl 1 and 2?
    # We could use "line art glyphs" (e.g. "P") on mpl 2
    default_markers = ["o", "s", "D", "v", "^", "p"]
    marker_scales = {"o": 1, "s": .85, "D": .9, "v": 1.3, "^": 1.3, "p": 1.25}
    default_dashes = ["", (4, 1.5), (1, 1),
                      (3, 1, 1.5, 1), (5, 1, 1, 1), (5, 1, 2, 1, 2, 1)]

    def establish_variables(self, x=None, y=None,
                            hue=None, size=None, style=None,
                            data=None):
        """Parse the inputs to define data for plotting."""
        # Initialize label variables
        x_label = y_label = hue_label = size_label = style_label = None

        # Option 1:
        # We have a wide-form datast
        # --------------------------

        if x is None and y is None:

            self.input_format = "wide"

            # Option 1a:
            # The input data is a Pandas DataFrame
            # ------------------------------------
            # We will assign the index to x, the values to y,
            # and the columns names to both hue and style

            # TODO accept a dict and try to coerce to a dataframe?

            if isinstance(data, pd.DataFrame):

                # Enforce numeric values
                try:
                    data.astype(np.float)
                except ValueError:
                    err = "A wide-form input must have only numeric values."
                    raise ValueError(err)

                plot_data = data.copy()
                plot_data.loc[:, "x"] = data.index
                plot_data = pd.melt(plot_data, "x",
                                    var_name="hue", value_name="y")
                plot_data["style"] = plot_data["hue"]

                x_label = getattr(data.index, "name", None)
                hue_label = style_label = getattr(plot_data.columns,
                                                  "name", None)

            # Option 1b:
            # The input data is an array or list
            # ----------------------------------

            else:

                if not len(data):

                    plot_data = pd.DataFrame(columns=["x", "y"])

                elif np.isscalar(np.asarray(data)[0]):

                    # The input data is a flat list(like):
                    # We assign a numeric index for x and use the values for y

                    x = getattr(data, "index", np.arange(len(data)))
                    plot_data = pd.DataFrame(dict(x=x, y=data))

                elif hasattr(data, "shape"):

                    # The input data is an array(like):
                    # We either use the index or assign a numeric index to x,
                    # the values to y, and id keys to both hue and style

                    plot_data = pd.DataFrame(data)
                    plot_data.loc[:, "x"] = plot_data.index
                    plot_data = pd.melt(plot_data, "x",
                                        var_name="hue",
                                        value_name="y")
                    plot_data["style"] = plot_data["hue"]

                else:

                    # The input data is a nested list: We will either use the
                    # index or assign a numeric index for x, use the values
                    # for y, and use numeric hue/style identifiers.

                    plot_data = []
                    for i, data_i in enumerate(data):
                        x = getattr(data_i, "index", np.arange(len(data_i)))
                        n = getattr(data_i, "name", i)
                        data_i = dict(x=x, y=data_i, hue=n, style=n, size=None)
                        plot_data.append(pd.DataFrame(data_i))
                    plot_data = pd.concat(plot_data)

        # Option 2:
        # We have long-form data
        # ----------------------

        elif x is not None and y is not None:

            self.input_format = "long"

            # Use variables as from the dataframe if specified
            if data is not None:
                x = data.get(x, x)
                y = data.get(y, y)
                hue = data.get(hue, hue)
                size = data.get(size, size)
                style = data.get(style, style)

            # Validate the inputs
            for input in [x, y, hue, size, style]:
                if isinstance(input, string_types):
                    err = "Could not interpret input '{}'".format(input)
                    raise ValueError(err)

            # Extract variable names
            x_label = getattr(x, "name", None)
            y_label = getattr(y, "name", None)
            hue_label = getattr(hue, "name", None)
            size_label = getattr(size, "name", None)
            style_label = getattr(style, "name", None)

            # Reassemble into a DataFrame
            plot_data = dict(x=x, y=y, hue=hue, style=style, size=size)
            plot_data = pd.DataFrame(plot_data)

        # Option 3:
        # Only one variable argument
        # --------------------------

        else:
            err = ("Either both or neither of `x` and `y` must be specified "
                   "(but try passing to `data`, which is more flexible).")
            raise ValueError(err)

        # ---- Post-processing

        # Assign default values for missing attribute variables
        for attr in ["hue", "style", "size"]:
            if attr not in plot_data:
                plot_data[attr] = None

        self.x_label = x_label
        self.y_label = y_label
        self.hue_label = hue_label
        self.size_label = size_label
        self.style_label = style_label
        self.plot_data = plot_data

        return plot_data

    def categorical_to_palette(self, data, order, palette):
        """Determine colors when the hue variable is qualitative."""
        # -- Identify the order and name of the levels

        if order is None:
            levels = categorical_order(data)
        else:
            levels = order
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            palette = dict(zip(levels, colors))

        return levels, palette

    def numeric_to_palette(self, data, order, palette, limits):
        """Determine colors when the hue variable is quantitative."""
        levels = list(np.sort(data.unique()))

        # TODO do we want to do something complicated to ensure contrast
        # at the extremes of the colormap against the background?

        # Identify the colormap to use
        if palette is None:
            cmap = mpl.cm.get_cmap(plt.rcParams["image.cmap"])
        elif isinstance(palette, mpl.colors.Colormap):
            cmap = palette
        else:
            try:
                cmap = mpl.cm.get_cmap(palette)
            except (ValueError, TypeError):
                err = "Palette {} not understood"
                raise ValueError(err)

        if limits is None:
            limits = data.min(), data.max()

        hue_min, hue_max = limits
        hue_min = data.min() if hue_min is None else hue_min
        hue_max = data.max() if hue_max is None else hue_max

        limits = hue_min, hue_max
        normalize = mpl.colors.Normalize(hue_min, hue_max, clip=True)
        palette = {l: cmap(normalize(l)) for l in levels}

        return levels, palette, cmap, limits

    def color_lookup(self, key):
        """Return the color corresponding to the hue level."""
        if self.hue_type == "numeric":
            norm = mpl.colors.Normalize(*self.hue_limits, clip=True)
            return self.cmap(norm(key))
        elif self.hue_type == "categorical":
            return self.palette[key]

    def size_lookup(self, key):
        """Return the size corresponding to the size level."""
        if self.size_type == "numeric":
            norm = mpl.colors.Normalize(*self.size_limits, clip=True)
            min_size, max_size = self.size_range
            return min_size + norm(key) * (max_size - min_size)
        elif self.size_type == "categorical":
            return self.sizes[key]

    def style_to_attributes(self, levels, style, defaults, name):
        """Convert a style argument to a dict of matplotlib attributes."""
        if style is True:
            attrdict = dict(zip(levels, defaults))
        elif style and isinstance(style, dict):
            attrdict = style
        elif style:
            attrdict = dict(zip(levels, style))
        else:
            attrdict = {}

        if attrdict:
            missing_levels = set(levels) - set(attrdict)
            if any(missing_levels):
                err = "These `style` levels are missing {}: {}"
                raise ValueError(err.format(name, missing_levels))

        return attrdict

    def _empty_data(self, data):
        """Test if a series is completely missing."""
        return data.isnull().all()

    def _semantic_type(self, data):
        """Determine if data should considered numeric or categorical."""
        if self.input_format == "wide":
            return "categorical"
        else:
            try:
                data.astype(np.float)
                return "numeric"
            except ValueError:
                return "categorical"


class _LinePlotter(_BasicPlotter):

    def __init__(self,
                 x=None, y=None, hue=None, size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_limits=None,
                 sizes=None, size_order=None, size_limits=None,
                 dashes=None, markers=None, style_order=None,
                 units=None, estimator=None, ci=None, n_boot=None,
                 sort=True, errstyle=None, legend=None):

        plot_data = self.establish_variables(x, y, hue, size, style, data)

        self.parse_hue(plot_data["hue"], palette, hue_order, hue_limits)
        self.parse_size(plot_data["size"], sizes, size_order, size_limits)
        self.parse_style(plot_data["style"], markers, dashes, style_order)

        self.sort = sort
        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot
        self.errstyle = errstyle

        self.legend = legend

    def subset_data(self):
        """Return (x, y) data for each subset defined by semantics."""
        data = self.plot_data
        all_true = pd.Series(True, data.index)

        iter_levels = product(self.hue_levels,
                              self.size_levels,
                              self.style_levels)

        for hue, size, style in iter_levels:

            hue_rows = all_true if hue is None else data["hue"] == hue
            size_rows = all_true if size is None else data["size"] == size
            style_rows = all_true if style is None else data["style"] == style

            rows = hue_rows & size_rows & style_rows
            subset_data = data.loc[rows, ["x", "y"]].dropna()

            if not len(subset_data):
                continue

            if self.sort:
                subset_data = sort_df(subset_data, ["x", "y"])

            yield (hue, size, style), subset_data

    def parse_hue(self, data, palette, order, limits):
        """Determine what colors to use given data characteristics."""
        if self._empty_data(data):

            # Set default values when not using a hue mapping
            levels = [None]
            palette = {}
            var_type = None
            cmap = None

        else:

            # Determine what kind of hue mapping we want
            var_type = self._semantic_type(data)

            # Override depending on the type of the palette argument
            if isinstance(palette, (dict, list)):
                var_type = "categorical"

        # -- Option 1: categorical color palette

        if var_type == "categorical":

            cmap = None
            levels, palette = self.categorical_to_palette(
                data, order, palette
            )

        # -- Option 2: sequential color palette

        elif var_type == "numeric":

            levels, palette, cmap, limits = self.numeric_to_palette(
                data, order, palette, limits
            )

        self.hue_levels = levels
        self.hue_limits = limits
        self.hue_type = var_type
        self.palette = palette
        self.cmap = cmap

    def parse_size(self, data, sizes, order, limits):
        """Determine the linewidths given data characteristics."""
        if self._empty_data(data):
            levels = [None]
            sizes = {}
            var_type = None
            width_range = None

        else:

            var_type = self._semantic_type(data)
            if var_type == "categorical":
                levels = categorical_order(data)
                numbers = np.arange(0, len(levels))[::-1]
            elif var_type == "numeric":
                levels = numbers = np.sort(data.unique())

            if isinstance(sizes, (dict, list)):

                # Use literal size values
                if isinstance(sizes, list):
                    if len(sizes) != len(levels):
                        err = "The `sizes` list has wrong number of levels"
                        raise ValueError(err)
                    sizes = dict(zip(levels, sizes))

                missing = set(levels) - set(sizes)
                if any(missing):
                    err = "Missing sizes for the following levels: {}"
                    raise ValueError(err.format(missing))

                width_range = min(sizes.values()), max(sizes.values())
                try:
                    limits = min(sizes.keys()), max(sizes.keys())
                except TypeError:
                    pass

            else:

                # Infer the range of sizes to use
                if sizes is None:
                    default = plt.rcParams["lines.linewidth"]
                    min_width, max_width = default * .5, default * 2
                else:
                    try:
                        min_width, max_width = sizes
                    except (TypeError, ValueError):
                        err = "sizes argument {} not understood".format(sizes)
                        raise ValueError(err)
                width_range = min_width, max_width

                # Infer the range of numeric values to map to sizes
                if limits is None:
                    s_min, s_max = numbers.min(), numbers.max()
                else:
                    s_min, s_max = limits
                    s_min = numbers.min() if s_min is None else s_min
                    s_max = numbers.max() if s_max is None else s_max

                # Map the numeric labels into the range of sizes
                # TODO rework to use size_lookup from above
                limits = s_min, s_max
                normalize = mpl.colors.Normalize(s_min, s_max, clip=True)
                sizes = {l: min_width + normalize(n) * (max_width - min_width)
                         for l, n in zip(levels, numbers)}

        self.sizes = sizes
        self.size_type = var_type
        self.size_levels = levels
        self.size_limits = limits
        self.size_range = width_range

    def parse_style(self, data, markers, dashes, order):
        """Determine the markers and line dashes."""

        if self._empty_data(data):

            levels = [None]
            dashes = {}
            markers = {}

        else:

            if order is None:
                levels = categorical_order(data)
            else:
                levels = order

            markers = self.style_to_attributes(
                levels, markers, self.default_markers, "markers"
            )

            dashes = self.style_to_attributes(
                levels, dashes, self.default_dashes, "dashes"
            )

        self.style_levels = levels
        self.dashes = dashes
        self.markers = markers

    def aggregate(self, vals, grouper, func, ci):
        """Compute an estimate and confidence interval using grouper."""
        n_boot = self.n_boot

        # Define a "null" CI for when we only have one value
        null_ci = pd.Series(index=["low", "high"], dtype=np.float)

        # Function to bootstrap in the context of a pandas group by
        def bootstrapped_cis(vals):

            if len(vals) == 1:
                return null_ci

            boots = bootstrap(vals, func=func, n_boot=n_boot)
            cis = utils.ci(boots, ci)
            return pd.Series(cis, ["low", "high"])

        # Group and get the aggregation estimate
        grouped = vals.groupby(grouper, sort=self.sort)
        est = grouped.agg(func)

        # Exit early if we don't want a confidence interval
        if ci is None:
            return est.index, est, None

        # Compute the error bar extents
        if ci == "sd":
            sd = grouped.std()
            cis = pd.DataFrame(np.c_[est - sd, est + sd],
                               index=est.index,
                               columns=["low", "high"]).stack()
        else:
            cis = grouped.apply(bootstrapped_cis)

        # Unpack the CIs into "wide" format for plotting
        if cis.notnull().any():
            cis = cis.unstack().reindex(est.index)
        else:
            cis = None

        return est.index, est, cis

    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""

        # Draw a test line, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the lineplot semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all lines
        # will have the same color, but they will have the color that
        # ax.plot() would have used for a single line, and calling lineplot
        # will advance the axes property cycle.

        scout, = ax.plot([], [], **kws)

        orig_color = kws.pop("color", scout.get_color())
        orig_marker = kws.pop("marker", scout.get_marker())
        orig_linewidth = kws.pop("linewidth",
                                 kws.pop("lw", scout.get_linewidth()))

        orig_dashes = kws.pop("dashes", "")

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        scout.remove()

        # Loop over the semantic subsets and draw a line for each

        for semantics, subset_data in self.subset_data():

            hue, size, style = semantics

            x, y = subset_data["x"], subset_data["y"]

            if self.estimator is not None:
                x, y, y_ci = self.aggregate(y, x, self.estimator, self.ci)
            else:
                y_ci = None

            kws["color"] = self.palette.get(hue, orig_color)
            kws["dashes"] = self.dashes.get(style, orig_dashes)
            kws["marker"] = self.markers.get(style, orig_marker)
            kws["linewidth"] = self.sizes.get(size, orig_linewidth)

            # --- Draw the main line

            # TODO when not estimating, use units to get multiple lines
            # with the same semantics?

            line, = ax.plot(x.values, y.values, **kws)
            line_color = line.get_color()
            line_alpha = line.get_alpha()

            # --- Draw the confidence intervals

            if y_ci is not None:

                if self.errstyle == "band":

                    ax.fill_between(x, y_ci["low"], y_ci["high"],
                                    color=line_color, alpha=.2)

                elif self.errstyle == "bars":

                    ci_xy = np.empty((len(x), 2, 2))
                    ci_xy[:, :, 0] = x[:, np.newaxis]
                    ci_xy[:, :, 1] = y_ci.values
                    lines = LineCollection(ci_xy,
                                           color=line_color,
                                           alpha=line_alpha)
                    ax.add_collection(lines)
                    ax.autoscale_view()

                else:
                    err = "`errstyle` must by 'band' or 'bars', not {}"
                    raise ValueError(err.format(self.errstyle))

        # TODO this should go in its own method?
        if self.x_label is not None:
            x_visible = any(t.get_visible() for t in ax.get_xticklabels())
            ax.set_xlabel(self.x_label, visible=x_visible)
        if self.y_label is not None:
            y_visible = any(t.get_visible() for t in ax.get_yticklabels())
            ax.set_ylabel(self.y_label, visible=y_visible)

        # Add legend
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

    def add_legend_data(self, ax):
        """Add labeled artists to represent the different plot semantics."""
        verbosity = self.legend
        if verbosity not in ["brief", "full"]:
            err = "`legend` must be 'brief', 'full', or False"
            raise ValueError(err)

        keys = []
        legend_data = {}

        def update(var_name, val_name, **kws):

            key = var_name, val_name
            if key in legend_data:
                legend_data[key].update(**kws)
            else:
                keys.append(key)
                legend_data[key] = dict(**kws)

        ticker = mpl.ticker.MaxNLocator(nbins=3)

        # -- Add a legend for hue semantics

        if verbosity == "brief" and self.hue_type == "numeric":
            hue_levels = (ticker.tick_values(*self.hue_limits)
                                .astype(self.plot_data["hue"].dtype))
        else:
            hue_levels = self.hue_levels

        for level in hue_levels:
            if level is not None:
                color = self.color_lookup(level)
                update(self.hue_label, level, color=color)

        # -- Add a legend for size semantics

        if verbosity == "brief" and self.size_type == "numeric":
            size_levels = (ticker.tick_values(*self.size_limits)
                                 .astype(self.plot_data["size"].dtype))
        else:
            size_levels = self.size_levels

        for level in size_levels:
            if level is not None:
                linewidth = self.size_lookup(level)
                update(self.size_label, level, linewidth=linewidth)

        # -- Add a legend for style semantics

        for level in self.style_levels:
            if level is not None:
                update(self.style_label, level,
                       marker=self.markers.get(level, ""),
                       dashes=self.dashes.get(level, ""))

        for key in keys:
            _, label = key
            kws = legend_data[key]
            kws.setdefault("color", ".2")
            ax.plot([], [], label=label, **kws)


class _ScatterPlotter(_BasicPlotter):

    def __init__(self):
        pass

    def plot(self, ax=None):
        pass


_basic_docs = dict(

    # ---  Introductory prose
    main_api_narrative=dedent("""\
    The relationship between ``x`` and ``y`` can be shown for different subsets
    of the data using the ``hue``, ``size``, and ``style`` parameters. These
    parameters control what visual semantics are used to identify the different
    subsets. It is possible to show up to three dimensions independently by
    using all three semantic types, but this style of plot can be hard to
    interpret and is often ineffective. Using redundant semantics (i.e. both
    ``hue`` and ``style`` for the same variable) can be helpful for making
    graphics more accessible.\
    """),

    # --- Shared function parameters
    data_vars=dedent("""\
    x, y : names of variables in ``data`` or vector data, optional
        Input data variables; must be numeric. Can pass data directly or
        reference columns in ``data``.\
    """),
    data=dedent("""\
    data : DataFrame, array, or list of arrays, optional
        Input data structure. If ``x`` and ``y`` are specified as names, this
        should be a "long-form" DataFrame containing those columns. Otherwise
        it is treated as "wide-form" data and grouping variables are ignored.
        See the examples for the various ways this parameter can be specified
        and the different effects of each.\
    """),
    palette=dedent("""\
    palette : string, list, dict, or matplotlib colormap
        An object that determines how colors are chosen when ``hue`` is used.
        It can be the name of a seaborn palette or matplotlib colormap, a list
        of colors (anything matplotlib understands), a dict mapping levels
        of the ``hue`` variable to colors, or a matplotlib colormap object.\
    """),
    hue_order=dedent("""\
    hue_order : list, optional
        Specified order for the appearance of the ``hue`` variable levels,
        otherwise they are determined from the data. Not relevant when the
        ``hue`` variable is numeric.\
    """),
    hue_limits=dedent("""\
    hue_limits : tuple, optional
        Limits in data units to use for the colormap applied to the ``hue``
        variable when it is numeric. Not relevant if it is categorical.\
    """),
    sizes=dedent("""
    sizes : list, dict, or tuple, optional
        An object that determines how sizes are chosen when ``size`` is used.
        It can always be a list of size values or a dict mapping levels of the
        ``size`` variable to sizes. When ``size``  is numeric, it can also be
        a tuple specifying the minimum and maximum size to use such that other
        values are normalized within this range.\
    """),
    size_order=dedent("""\
    size_order : list, optional
        Specified order for appearance of the ``size`` variable levels,
        otherwise they are determined from the data. Not relevant when the
        ``size`` variable is numeric.\
    """),
    size_limits=dedent("""\
    size_limits : tuple, optional
        Limits in data units to use for the size normalization when the
        ``size`` variable is numeric.\
    """),
    style_order=dedent("""\
    style_order : list, optional
        Specified order for appearance of the ``style`` variable levels
        otherwise they are determined from the data. Not relevant when the
        ``style`` variable is numeric.\
    """),
    units=dedent("""\
    units : {long_form_var}
        Grouping variable identifying sampling units. Currently has no effect.\
    """),
    estimator=dedent("""\
    estimator : name of pandas method or callable or None, optional
        Method for aggregating across multiple observations of the ``y``
        variable at the same ``x`` level. If ``None``, all observations will
        be drawn.\
    """),
    ci=dedent("""\
    ci : int or "sd" or None, optional
        Size of the confidence interval to draw when aggregating with an
        estimator. "sd" means to draw the standard deviation of the data.
        Setting to ``None`` will skip bootstrapping.\
    """),
    n_boot=dedent("""\
    n_boot : int, optional
        Number of bootstraps to use for computing the confidence interval.\
    """),
    legend=dedent("""\
    legend : "brief", "full", or False, optional
        How to draw the legend. If "brief", numeric ``hue`` and ``size``
        variables will be represented with a sample of evenly spaced values.
        If "full", every group will get an entry in the legend. If ``False``,
        no legend data is added and no legend is drawn.\
    """),
    ax_in=dedent("""\
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
    ax_out=dedent("""\
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
    """),

    # --- Repeated phrases
    long_form_var="name of variables in ``data`` or vector data, optional",


)


def lineplot(x=None, y=None, hue=None, size=None, style=None, data=None,
             palette=None, hue_order=None, hue_limits=None,
             sizes=None, size_order=None, size_limits=None,
             dashes=True, markers=None, style_order=None,
             units=None, estimator="mean", ci=95, n_boot=1000,
             sort=True, errstyle="band",
             legend="brief", ax=None, **kwargs):

    p = _LinePlotter(
        x=x, y=y, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_limits=hue_limits,
        sizes=sizes, size_order=size_order, size_limits=size_limits,
        dashes=dashes, markers=markers, style_order=style_order,
        units=units, estimator=estimator, ci=ci, n_boot=n_boot,
        sort=sort, errstyle=errstyle, legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax


lineplot.__doc__ = dedent("""\
    Draw a plot with numeric x and y values where the points are connected.

    {main_api_narrative}

    By default, the plot aggregates over multiple ``y`` values at each value of
    ``x`` and shows an estimate of the central tendency and a confidence
    interval for that estimate.

    Parameters
    ----------
    {data_vars}
    hue : {long_form_var}
        Grouping variable that will produce lines with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : {long_form_var}
        Gropuing variable that will produce lines with different widths.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : {long_form_var}
        Grouping variable that will produce lines with different dashes
        and/or markers. Can have a numeric dtype but will always be treated
        as categorical.
    {data}
    {palette}
    {hue_order}
    {hue_limits}
    {sizes}
    {size_order}
    {size_limits}
    dashes : boolean, list, or dictionary, optional
        Object determining how to draw the lines for different levels of the
        ``style`` variable. Setting to ``True`` will use default dash codes, or
        you can pass a list of dash codes or a dictionary mapping levels of the
        ``style`` variable to dash codes. Setting to ``False`` will use solid
        lines for all subsets. Dashes are specified as in matplotlib: a tuple
        of ``(segment, gap)`` lengths, or an empty string to draw a solid line.
    markers : boolean, list, or dictionary, optional
        Object determining how to draw the markers for different levels of the
        ``style`` variable. Setting to ``True`` will use default markers, or
        you can pass a list of markers or a dictionary mapping levels of the
        ``style`` variable to markers. Setting to ``False`` will draw
        marker-less lines.  Markers are specified as in matplotlib.
    {style_order}
    {units}
    {estimator}
    {ci}
    {n_boot}
    sort : boolean, optional
        If True, the data will be sorted by the x and y variables, otherwise
        lines will connect points in the order they appear in the dataset.
    errstyle : "band" or "bars", optional
        Whether to draw the confidence intervals with translucent error bands
        or discrete error bars.
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed down to ``plt.plot`` at draw time.

    Returns
    -------
    {ax_out}

    See Also
    --------
    scatterplot : Show the relationship between two variables without
                  emphasizing continuity of the ``x`` variable.
    pointplot : Show the relationship between two variables when one is
                categorical.

    Examples
    --------

    Draw a single line plot with error bands showing a confidence interval:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set()
        >>> import matplotlib.pyplot as plt
        >>> fmri = sns.load_dataset("fmri")
        >>> ax = sns.lineplot(x="timepoint", y="signal", data=fmri)

    Group by another variable and show the groups with different colors:


    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   data=fmri)

    Show the grouping variable with both color and line dashing:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="event", style="event", data=fmri)

    Use color and line dashing to represent two different grouping variables:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="region", style="event", data=fmri)

    Use markers instead of the dashes to identify groups:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="event", style="event",
        ...                   markers=True, dashes=False, data=fmri)

    Show error bars instead of error bands and plot the standard error:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   errstyle="bars", ci=68, data=fmri)

    Use a quantitative color mapping:

    .. plot::
        :context: close-figs

        >>> dots = sns.load_dataset("dots").query("align == 'dots'")
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   data=dots)

    Change the data limits over which the colormap is normalized:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   hue_limits=(0, 100), data=dots)

    Use a different color palette:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   palette="viridis_r", data=dots)

    Use specific color values, treating the hue variable as categorical:

    .. plot::
        :context: close-figs

        >>> palette = sns.color_palette("mako_r", 6)
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   palette=palette, data=dots)

    Change the width of the lines with a quantitative variable:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   size="coherence", hue="choice",
        ...                   legend="full", data=dots)

    Change the range of line widths used to normalize the size variable:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   size="coherence", hue="choice",
        ...                   sizes=(.2, 1), data=dots)

    Plot from a wide-form DataFrame:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; plt.close("all")
        >>> index = pd.date_range("1 1 2000", periods=100,
        ...                       freq="m", name="date")
        >>> data = np.random.randn(100, 4).cumsum(axis=0)
        >>> wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
        >>> ax = sns.lineplot(data=wide_df)

    Plot from a list of Series:

    .. plot::
        :context: close-figs

        >>> list_data = [wide_df.loc[:"2005", "a"], wide_df.loc["2003":, "b"]]
        >>> ax = sns.lineplot(data=list_data)

    Plot a single Series, pass kwargs to ``plt.plot``:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(data=wide_df["a"], color="coral", label="line")

    Draw lines at points as they appear in the dataset:

    .. plot::
        :context: close-figs

        >>> x, y = np.random.randn(2, 5000).cumsum(axis=1)
        >>> ax = sns.lineplot(x=x, y=y, sort=False, lw=1)


    """).format(**_basic_docs)


def scatterplot(x=None, y=None, hue=None, style=None, size=None, data=None,
                palette=None, hue_order=None, hue_limits=None,
                markers=None, style_order=None,
                sizes=None, size_order=None, size_limits=None,
                x_bins=None, y_bins=None,
                estimator=None, ci=95, n_boot=1000, units=None,
                errstyle="bars", alpha="auto",
                x_jitter=None, y_jitter=None,
                ax=None, **kwargs):

    # TODO auto alpha

    pass
