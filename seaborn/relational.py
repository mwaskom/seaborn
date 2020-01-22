from __future__ import division
from itertools import product
from textwrap import dedent
from distutils.version import LooseVersion
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils
from .utils import (categorical_order, get_color_cycle, ci_to_errsize, sort_df,
                    remove_na, locator_to_legend_entries)
from .algorithms import bootstrap
from .palettes import (color_palette, cubehelix_palette,
                       _parse_cubehelix_args, QUAL_PALETTES)
from .axisgrid import FacetGrid, _facet_docs


__all__ = ["relplot", "scatterplot", "lineplot"]


class _RelationalPlotter(object):

    if LooseVersion(mpl.__version__) >= "2.0":
        default_markers = ["o", "X", "s", "P", "D", "^", "v", "p"]
    else:
        default_markers = ["o", "s", "D", "^", "v", "p"]
    default_dashes = ["", (4, 1.5), (1, 1),
                      (3, 1, 1.5, 1), (5, 1, 1, 1),
                      (5, 1, 2, 1, 2, 1)]

    def establish_variables(self, x=None, y=None,
                            hue=None, size=None, style=None,
                            units=None, data=None):
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
                units = data.get(units, units)

            # Validate the inputs
            for var in [x, y, hue, size, style, units]:
                if isinstance(var, str):
                    err = "Could not interpret input '{}'".format(var)
                    raise ValueError(err)

            # Extract variable names
            x_label = getattr(x, "name", None)
            y_label = getattr(y, "name", None)
            hue_label = getattr(hue, "name", None)
            size_label = getattr(size, "name", None)
            style_label = getattr(style, "name", None)

            # Reassemble into a DataFrame
            plot_data = dict(
                x=x, y=y,
                hue=hue, style=style, size=size,
                units=units
            )
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
        for attr in ["hue", "style", "size", "units"]:
            if attr not in plot_data:
                plot_data[attr] = None

        # Determine which semantics have (some) data
        plot_valid = plot_data.notnull().any()
        semantics = ["x", "y"] + [
            name for name in ["hue", "size", "style"]
            if plot_valid[name]
        ]

        self.x_label = x_label
        self.y_label = y_label
        self.hue_label = hue_label
        self.size_label = size_label
        self.style_label = style_label
        self.plot_data = plot_data
        self.semantics = semantics

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

    def numeric_to_palette(self, data, order, palette, norm):
        """Determine colors when the hue variable is quantitative."""
        levels = list(np.sort(remove_na(data.unique())))

        # TODO do we want to do something complicated to ensure contrast
        # at the extremes of the colormap against the background?

        # Identify the colormap to use
        palette = "ch:" if palette is None else palette
        if isinstance(palette, mpl.colors.Colormap):
            cmap = palette
        elif str(palette).startswith("ch:"):
            args, kwargs = _parse_cubehelix_args(palette)
            cmap = cubehelix_palette(0, *args, as_cmap=True, **kwargs)
        elif isinstance(palette, dict):
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
        else:
            try:
                cmap = mpl.cm.get_cmap(palette)
            except (ValueError, TypeError):
                err = "Palette {} not understood"
                raise ValueError(err)

        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, tuple):
            norm = mpl.colors.Normalize(*norm)
        elif not isinstance(norm, mpl.colors.Normalize):
            err = "``hue_norm`` must be None, tuple, or Normalize object."
            raise ValueError(err)

        if not norm.scaled():
            norm(np.asarray(data.dropna()))

        # TODO this should also use color_lookup, but that needs the
        # class attributes that get set after using this function...
        if not isinstance(palette, dict):
            palette = dict(zip(levels, cmap(norm(levels))))
        # palette = {l: cmap(norm([l, 1]))[0] for l in levels}

        return levels, palette, cmap, norm

    def color_lookup(self, key):
        """Return the color corresponding to the hue level."""
        if self.hue_type == "numeric":
            normed = self.hue_norm(key)
            if np.ma.is_masked(normed):
                normed = np.nan
            return self.cmap(normed)
        elif self.hue_type == "categorical":
            return self.palette[key]

    def size_lookup(self, key):
        """Return the size corresponding to the size level."""
        if self.size_type == "numeric":
            min_size, max_size = self.size_range
            val = self.size_norm(key)
            if np.ma.is_masked(val):
                return 0
            return min_size + val * (max_size - min_size)
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
            data["units"] = data.units.fillna("")
            subset_data = data.loc[rows, ["units", "x", "y"]].dropna()

            if not len(subset_data):
                continue

            if self.sort:
                subset_data = sort_df(subset_data, ["units", "x", "y"])

            if self.units is None:
                subset_data = subset_data.drop("units", axis=1)

            yield (hue, size, style), subset_data

    def parse_hue(self, data, palette, order, norm):
        """Determine what colors to use given data characteristics."""
        if self._empty_data(data):

            # Set default values when not using a hue mapping
            levels = [None]
            limits = None
            norm = None
            palette = {}
            var_type = None
            cmap = None

        else:

            # Determine what kind of hue mapping we want
            var_type = self._semantic_type(data)

            # Override depending on the type of the palette argument
            if palette in QUAL_PALETTES:
                var_type = "categorical"
            elif norm is not None:
                var_type = "numeric"
            elif isinstance(palette, (dict, list)):
                var_type = "categorical"

        # -- Option 1: categorical color palette

        if var_type == "categorical":

            cmap = None
            limits = None
            levels, palette = self.categorical_to_palette(
                # List comprehension here is required to
                # overcome differences in the way pandas
                # externalizes numpy datetime64
                list(data), order, palette
            )

        # -- Option 2: sequential color palette

        elif var_type == "numeric":

            data = pd.to_numeric(data)

            levels, palette, cmap, norm = self.numeric_to_palette(
                data, order, palette, norm
            )
            limits = norm.vmin, norm.vmax

        self.hue_levels = levels
        self.hue_norm = norm
        self.hue_limits = limits
        self.hue_type = var_type
        self.palette = palette
        self.cmap = cmap

        # Update data as it may have changed dtype
        self.plot_data["hue"] = data

    def parse_size(self, data, sizes, order, norm):
        """Determine the linewidths given data characteristics."""

        # TODO could break out two options like parse_hue does for clarity

        if self._empty_data(data):
            levels = [None]
            limits = None
            norm = None
            sizes = {}
            var_type = None
            width_range = None

        else:

            var_type = self._semantic_type(data)

            # Override depending on the type of the sizes argument
            if norm is not None:
                var_type = "numeric"
            elif isinstance(sizes, (dict, list)):
                var_type = "categorical"

            if var_type == "categorical":
                levels = categorical_order(data, order)
                numbers = np.arange(1, 1 + len(levels))[::-1]

            elif var_type == "numeric":
                data = pd.to_numeric(data)
                levels = numbers = np.sort(remove_na(data.unique()))

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
                    limits = None

            else:

                # Infer the range of sizes to use
                if sizes is None:
                    min_width, max_width = self._default_size_range
                else:
                    try:
                        min_width, max_width = sizes
                    except (TypeError, ValueError):
                        err = "sizes argument {} not understood".format(sizes)
                        raise ValueError(err)
                width_range = min_width, max_width

                if norm is None:
                    norm = mpl.colors.Normalize()
                elif isinstance(norm, tuple):
                    norm = mpl.colors.Normalize(*norm)
                elif not isinstance(norm, mpl.colors.Normalize):
                    err = ("``size_norm`` must be None, tuple, "
                           "or Normalize object.")
                    raise ValueError(err)

                norm.clip = True
                if not norm.scaled():
                    norm(np.asarray(numbers))
                limits = norm.vmin, norm.vmax

                scl = norm(numbers)
                widths = np.asarray(min_width + scl * (max_width - min_width))
                if scl.mask.any():
                    widths[scl.mask] = 0
                sizes = dict(zip(levels, widths))
                # sizes = {l: min_width + norm(n) * (max_width - min_width)
                #          for l, n in zip(levels, numbers)}

            if var_type == "categorical":
                # Don't keep a reference to the norm, which will avoid
                # downstream  code from switching to numerical interpretation
                norm = None

        self.sizes = sizes
        self.size_type = var_type
        self.size_levels = levels
        self.size_norm = norm
        self.size_limits = limits
        self.size_range = width_range

        # Update data as it may have changed dtype
        self.plot_data["size"] = data

    def parse_style(self, data, markers, dashes, order):
        """Determine the markers and line dashes."""

        if self._empty_data(data):

            levels = [None]
            dashes = {}
            markers = {}

        else:

            if order is None:
                # List comprehension here is required to
                # overcome differences in the way pandas
                # coerces numpy datatypes
                levels = categorical_order(list(data))
            else:
                levels = order

            markers = self.style_to_attributes(
                levels, markers, self.default_markers, "markers"
            )

            dashes = self.style_to_attributes(
                levels, dashes, self.default_dashes, "dashes"
            )

        paths = {}
        filled_markers = []
        for k, m in markers.items():
            if not isinstance(m, mpl.markers.MarkerStyle):
                m = mpl.markers.MarkerStyle(m)
            paths[k] = m.get_path().transformed(m.get_transform())
            filled_markers.append(m.is_filled())

        # Mixture of filled and unfilled markers will show line art markers
        # in the edge color, which defaults to white. This can be handled,
        # but there would be additional complexity with specifying the
        # weight of the line art markers without overwhelming the filled
        # ones with the edges. So for now, we will disallow mixtures.
        if any(filled_markers) and not all(filled_markers):
            err = "Filled and line art markers cannot be mixed"
            raise ValueError(err)

        self.style_levels = levels
        self.dashes = dashes
        self.markers = markers
        self.paths = paths

    def _empty_data(self, data):
        """Test if a series is completely missing."""
        return data.isnull().all()

    def _semantic_type(self, data):
        """Determine if data should considered numeric or categorical."""
        if self.input_format == "wide":
            return "categorical"
        elif isinstance(data, pd.Series) and data.dtype.name == "category":
            return "categorical"
        else:
            try:
                float_data = data.astype(np.float)
                values = np.unique(float_data.dropna())
                # TODO replace with isin when pinned np version >= 1.13
                if np.all(np.in1d(values, np.array([0., 1.]))):
                    return "categorical"
                return "numeric"
            except (ValueError, TypeError):
                return "categorical"

    def label_axes(self, ax):
        """Set x and y labels with visibility that matches the ticklabels."""
        if self.x_label is not None:
            x_visible = any(t.get_visible() for t in ax.get_xticklabels())
            ax.set_xlabel(self.x_label, visible=x_visible)
        if self.y_label is not None:
            y_visible = any(t.get_visible() for t in ax.get_yticklabels())
            ax.set_ylabel(self.y_label, visible=y_visible)

    def add_legend_data(self, ax):
        """Add labeled artists to represent the different plot semantics."""
        verbosity = self.legend
        if verbosity not in ["brief", "full"]:
            err = "`legend` must be 'brief', 'full', or False"
            raise ValueError(err)

        legend_kwargs = {}
        keys = []

        title_kws = dict(color="w", s=0, linewidth=0, marker="", dashes="")

        def update(var_name, val_name, **kws):

            key = var_name, val_name
            if key in legend_kwargs:
                legend_kwargs[key].update(**kws)
            else:
                keys.append(key)

                legend_kwargs[key] = dict(**kws)

        # -- Add a legend for hue semantics
        if verbosity == "brief" and self.hue_type == "numeric":
            if isinstance(self.hue_norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=3)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=3)
            hue_levels, hue_formatted_levels = locator_to_legend_entries(
                locator, self.hue_limits, self.plot_data["hue"].dtype
            )
        else:
            hue_levels = hue_formatted_levels = self.hue_levels

        # Add the hue semantic subtitle
        if self.hue_label is not None:
            update((self.hue_label, "title"), self.hue_label, **title_kws)

        # Add the hue semantic labels
        for level, formatted_level in zip(hue_levels, hue_formatted_levels):
            if level is not None:
                color = self.color_lookup(level)
                update(self.hue_label, formatted_level, color=color)

        # -- Add a legend for size semantics

        if verbosity == "brief" and self.size_type == "numeric":
            if isinstance(self.size_norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=3)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=3)
            size_levels, size_formatted_levels = locator_to_legend_entries(
                locator, self.size_limits, self.plot_data["size"].dtype)
        else:
            size_levels = size_formatted_levels = self.size_levels

        # Add the size semantic subtitle
        if self.size_label is not None:
            update((self.size_label, "title"), self.size_label, **title_kws)

        # Add the size semantic labels
        for level, formatted_level in zip(size_levels, size_formatted_levels):
            if level is not None:
                size = self.size_lookup(level)
                update(
                    self.size_label, formatted_level, linewidth=size, s=size)

        # -- Add a legend for style semantics

        # Add the style semantic title
        if self.style_label is not None:
            update((self.style_label, "title"), self.style_label, **title_kws)

        # Add the style semantic labels
        for level in self.style_levels:
            if level is not None:
                update(self.style_label, level,
                       marker=self.markers.get(level, ""),
                       dashes=self.dashes.get(level, ""))

        func = getattr(ax, self._legend_func)

        legend_data = {}
        legend_order = []

        for key in keys:

            _, label = key
            kws = legend_kwargs[key]
            kws.setdefault("color", ".2")
            use_kws = {}
            for attr in self._legend_attributes + ["visible"]:
                if attr in kws:
                    use_kws[attr] = kws[attr]
            artist = func([], [], label=label, **use_kws)
            if self._legend_func == "plot":
                artist = artist[0]
            legend_data[key] = artist
            legend_order.append(key)

        self.legend_data = legend_data
        self.legend_order = legend_order


class _LinePlotter(_RelationalPlotter):

    _legend_attributes = ["color", "linewidth", "marker", "dashes"]
    _legend_func = "plot"

    def __init__(self,
                 x=None, y=None, hue=None, size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None,
                 sizes=None, size_order=None, size_norm=None,
                 dashes=None, markers=None, style_order=None,
                 units=None, estimator=None, ci=None, n_boot=None, seed=None,
                 sort=True, err_style=None, err_kws=None, legend=None):

        plot_data = self.establish_variables(
            x, y, hue, size, style, units, data
        )

        self._default_size_range = (
            np.r_[.5, 2] * mpl.rcParams["lines.linewidth"]
        )

        self.parse_hue(plot_data["hue"], palette, hue_order, hue_norm)
        self.parse_size(plot_data["size"], sizes, size_order, size_norm)
        self.parse_style(plot_data["style"], markers, dashes, style_order)

        self.units = units
        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot
        self.seed = seed
        self.sort = sort
        self.err_style = err_style
        self.err_kws = {} if err_kws is None else err_kws

        self.legend = legend

    def aggregate(self, vals, grouper, units=None):
        """Compute an estimate and confidence interval using grouper."""
        func = self.estimator
        ci = self.ci
        n_boot = self.n_boot
        seed = self.seed

        # Define a "null" CI for when we only have one value
        null_ci = pd.Series(index=["low", "high"], dtype=np.float)

        # Function to bootstrap in the context of a pandas group by
        def bootstrapped_cis(vals):

            if len(vals) <= 1:
                return null_ci

            boots = bootstrap(vals, func=func, n_boot=n_boot, seed=seed)
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

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        scout, = ax.plot([], [], **kws)

        orig_color = kws.pop("color", scout.get_color())
        orig_marker = kws.pop("marker", scout.get_marker())
        orig_linewidth = kws.pop("linewidth",
                                 kws.pop("lw", scout.get_linewidth()))

        orig_dashes = kws.pop("dashes", "")

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        scout.remove()

        # Set default error kwargs
        err_kws = self.err_kws.copy()
        if self.err_style == "band":
            err_kws.setdefault("alpha", .2)
        elif self.err_style == "bars":
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))

        # Loop over the semantic subsets and draw a line for each

        for semantics, data in self.subset_data():

            hue, size, style = semantics
            x, y, units = data["x"], data["y"], data.get("units", None)

            if self.estimator is not None:
                if self.units is not None:
                    err = "estimator must be None when specifying units"
                    raise ValueError(err)
                x, y, y_ci = self.aggregate(y, x, units)
            else:
                y_ci = None

            kws["color"] = self.palette.get(hue, orig_color)
            kws["dashes"] = self.dashes.get(style, orig_dashes)
            kws["marker"] = self.markers.get(style, orig_marker)
            kws["linewidth"] = self.sizes.get(size, orig_linewidth)

            line, = ax.plot([], [], **kws)
            line_color = line.get_color()
            line_alpha = line.get_alpha()
            line_capstyle = line.get_solid_capstyle()
            line.remove()

            # --- Draw the main line

            x, y = np.asarray(x), np.asarray(y)

            if self.units is None:
                line, = ax.plot(x, y, **kws)

            else:
                for u in units.unique():
                    rows = np.asarray(units == u)
                    ax.plot(x[rows], y[rows], **kws)

            # --- Draw the confidence intervals

            if y_ci is not None:

                low, high = np.asarray(y_ci["low"]), np.asarray(y_ci["high"])

                if self.err_style == "band":

                    ax.fill_between(x, low, high, color=line_color, **err_kws)

                elif self.err_style == "bars":

                    y_err = ci_to_errsize((low, high), y)
                    ebars = ax.errorbar(x, y, y_err, linestyle="",
                                        color=line_color, alpha=line_alpha,
                                        **err_kws)

                    # Set the capstyle properly on the error bars
                    for obj in ebars.get_children():
                        try:
                            obj.set_capstyle(line_capstyle)
                        except AttributeError:
                            # Does not exist on mpl < 2.2
                            pass

        # Finalize the axes details
        self.label_axes(ax)
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()


class _ScatterPlotter(_RelationalPlotter):

    _legend_attributes = ["color", "s", "marker"]
    _legend_func = "scatter"

    def __init__(self,
                 x=None, y=None, hue=None, size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None,
                 sizes=None, size_order=None, size_norm=None,
                 dashes=None, markers=None, style_order=None,
                 x_bins=None, y_bins=None,
                 units=None, estimator=None, ci=None, n_boot=None,
                 alpha=None, x_jitter=None, y_jitter=None,
                 legend=None):

        plot_data = self.establish_variables(
            x, y, hue, size, style, units, data
        )

        self._default_size_range = (
            np.r_[.5, 2] * np.square(mpl.rcParams["lines.markersize"])
        )

        self.parse_hue(plot_data["hue"], palette, hue_order, hue_norm)
        self.parse_size(plot_data["size"], sizes, size_order, size_norm)
        self.parse_style(plot_data["style"], markers, None, style_order)
        self.units = units

        self.alpha = alpha

        self.legend = legend

    def plot(self, ax, kws):

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        scout = ax.scatter([], [], **kws)
        s = kws.pop("s", scout.get_sizes())
        c = kws.pop("c", scout.get_facecolors())
        scout.remove()

        kws.pop("color", None)  # TODO is this optimal?

        kws.setdefault("linewidth", .75)  # TODO scale with marker size?
        kws.setdefault("edgecolor", "w")

        if self.markers:
            # Use a representative marker so scatter sets the edgecolor
            # properly for line art markers. We currently enforce either
            # all or none line art so this works.
            example_marker = list(self.markers.values())[0]
            kws.setdefault("marker", example_marker)

        # TODO this makes it impossible to vary alpha with hue which might
        # otherwise be useful? Should we just pass None?
        kws["alpha"] = 1 if self.alpha == "auto" else self.alpha

        # Assign arguments for plt.scatter and draw the plot

        data = self.plot_data[self.semantics].dropna()
        if not data.size:
            return

        x = data["x"]
        y = data["y"]

        if self.palette:
            c = [self.palette.get(val) for val in data["hue"]]

        if self.sizes:
            s = [self.sizes.get(val) for val in data["size"]]

        args = np.asarray(x), np.asarray(y), np.asarray(s), np.asarray(c)
        points = ax.scatter(*args, **kws)

        # Update the paths to get different marker shapes. This has to be
        # done here because plt.scatter allows varying sizes and colors
        # but only a single marker shape per call.

        if self.paths:
            p = [self.paths.get(val) for val in data["style"]]
            points.set_paths(p)

        # Finalize the axes details
        self.label_axes(ax)
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()


_relational_docs = dict(

    # ---  Introductory prose
    main_api_narrative=dedent("""\
    The relationship between ``x`` and ``y`` can be shown for different subsets
    of the data using the ``hue``, ``size``, and ``style`` parameters. These
    parameters control what visual semantics are used to identify the different
    subsets. It is possible to show up to three dimensions independently by
    using all three semantic types, but this style of plot can be hard to
    interpret and is often ineffective. Using redundant semantics (i.e. both
    ``hue`` and ``style`` for the same variable) can be helpful for making
    graphics more accessible.

    See the :ref:`tutorial <relational_tutorial>` for more information.\
    """),

    relational_semantic_narrative=dedent("""\
    The default treatment of the ``hue`` (and to a lesser extent, ``size``)
    semantic, if present, depends on whether the variable is inferred to
    represent "numeric" or "categorical" data. In particular, numeric variables
    are represented with a sequential colormap by default, and the legend
    entries show regular "ticks" with values that may or may not exist in the
    data. This behavior can be controlled through various parameters, as
    described and illustrated below.\
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
    hue_norm=dedent("""\
    hue_norm : tuple or Normalize object, optional
        Normalization in data units for colormap applied to the ``hue``
        variable when it is numeric. Not relevant if it is categorical.\
    """),
    sizes=dedent("""\
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
    size_norm=dedent("""\
    size_norm : tuple or Normalize object, optional
        Normalization in data units for scaling plot objects when the
        ``size`` variable is numeric.\
    """),
    markers=dedent("""\
    markers : boolean, list, or dictionary, optional
        Object determining how to draw the markers for different levels of the
        ``style`` variable. Setting to ``True`` will use default markers, or
        you can pass a list of markers or a dictionary mapping levels of the
        ``style`` variable to markers. Setting to ``False`` will draw
        marker-less lines.  Markers are specified as in matplotlib.\
    """),
    style_order=dedent("""\
    style_order : list, optional
        Specified order for appearance of the ``style`` variable levels
        otherwise they are determined from the data. Not relevant when the
        ``style`` variable is numeric.\
    """),
    units=dedent("""\
    units : {long_form_var}
        Grouping variable identifying sampling units. When used, a separate
        line will be drawn for each unit with appropriate semantics, but no
        legend entry will be added. Useful for showing distribution of
        experimental replicates when exact identities are not needed.
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
    seed=dedent("""\
    seed : int, numpy.random.Generator, or numpy.random.RandomState, optional
        Seed or random number generator for reproducible bootstrapping.\
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

_relational_docs.update(_facet_docs)


def lineplot(x=None, y=None, hue=None, size=None, style=None, data=None,
             palette=None, hue_order=None, hue_norm=None,
             sizes=None, size_order=None, size_norm=None,
             dashes=True, markers=None, style_order=None,
             units=None, estimator="mean", ci=95, n_boot=1000, seed=None,
             sort=True, err_style="band", err_kws=None,
             legend="brief", ax=None, **kwargs):

    p = _LinePlotter(
        x=x, y=y, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        dashes=dashes, markers=markers, style_order=style_order,
        units=units, estimator=estimator, ci=ci, n_boot=n_boot, seed=seed,
        sort=sort, err_style=err_style, err_kws=err_kws, legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)
    return ax


lineplot.__doc__ = dedent("""\
    Draw a line plot with possibility of several semantic groupings.

    {main_api_narrative}

    {relational_semantic_narrative}

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
        Grouping variable that will produce lines with different widths.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : {long_form_var}
        Grouping variable that will produce lines with different dashes
        and/or markers. Can have a numeric dtype but will always be treated
        as categorical.
    {data}
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    dashes : boolean, list, or dictionary, optional
        Object determining how to draw the lines for different levels of the
        ``style`` variable. Setting to ``True`` will use default dash codes, or
        you can pass a list of dash codes or a dictionary mapping levels of the
        ``style`` variable to dash codes. Setting to ``False`` will use solid
        lines for all subsets. Dashes are specified as in matplotlib: a tuple
        of ``(segment, gap)`` lengths, or an empty string to draw a solid line.
    {markers}
    {style_order}
    {units}
    {estimator}
    {ci}
    {n_boot}
    {seed}
    sort : boolean, optional
        If True, the data will be sorted by the x and y variables, otherwise
        lines will connect points in the order they appear in the dataset.
    err_style : "band" or "bars", optional
        Whether to draw the confidence intervals with translucent error bands
        or discrete error bars.
    err_kws : dict of keyword arguments
        Additional paramters to control the aesthetics of the error bars. The
        kwargs are passed either to :meth:`matplotlib.axes.Axes.fill_between`
        or :meth:`matplotlib.axes.Axes.errorbar`, depending on ``err_style``.
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed down to
        :meth:`matplotlib.axes.Axes.plot`.

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
        ...                   err_style="bars", ci=68, data=fmri)

    Show experimental replicates instead of aggregating:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   units="subject", estimator=None, lw=1,
        ...                   data=fmri.query("region == 'frontal'"))

    Use a quantitative color mapping:

    .. plot::
        :context: close-figs

        >>> dots = sns.load_dataset("dots").query("align == 'dots'")
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   data=dots)

    Use a different normalization for the colormap:

    .. plot::
        :context: close-figs

        >>> from matplotlib.colors import LogNorm
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   hue_norm=LogNorm(), data=dots)

    Use a different color palette:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   palette="ch:2.5,.25", data=dots)

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
        ...                   sizes=(.25, 2.5), data=dots)

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

    Plot a single Series, pass kwargs to :meth:`matplotlib.axes.Axes.plot`:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(data=wide_df["a"], color="coral", label="line")

    Draw lines at points as they appear in the dataset:

    .. plot::
        :context: close-figs

        >>> x, y = np.random.randn(2, 5000).cumsum(axis=1)
        >>> ax = sns.lineplot(x=x, y=y, sort=False, lw=1)

    Use :func:`relplot` to combine :func:`lineplot` and :class:`FacetGrid`:
    This allows grouping within additional categorical variables. Using
    :func:`relplot` is safer than using :class:`FacetGrid` directly, as it
    ensures synchronization of the semantic mappings across facets.

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                  col="region", hue="event", style="event",
        ...                  kind="line", data=fmri)

    """).format(**_relational_docs)


def scatterplot(x=None, y=None, hue=None, style=None, size=None, data=None,
                palette=None, hue_order=None, hue_norm=None,
                sizes=None, size_order=None, size_norm=None,
                markers=True, style_order=None,
                x_bins=None, y_bins=None,
                units=None, estimator=None, ci=95, n_boot=1000,
                alpha="auto", x_jitter=None, y_jitter=None,
                legend="brief", ax=None, **kwargs):

    p = _ScatterPlotter(
        x=x, y=y, hue=hue, style=style, size=size, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        markers=markers, style_order=style_order,
        x_bins=x_bins, y_bins=y_bins,
        estimator=estimator, ci=ci, n_boot=n_boot,
        alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax


scatterplot.__doc__ = dedent("""\
    Draw a scatter plot with possibility of several semantic groupings.

    {main_api_narrative}

    {relational_semantic_narrative}

    Parameters
    ----------
    {data_vars}
    hue : {long_form_var}
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : {long_form_var}
        Grouping variable that will produce points with different sizes.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : {long_form_var}
        Grouping variable that will produce points with different markers.
        Can have a numeric dtype but will always be treated as categorical.
    {data}
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    {markers}
    {style_order}
    {{x,y}}_bins : lists or arrays or functions
        *Currently non-functional.*
    {units}
        *Currently non-functional.*
    {estimator}
        *Currently non-functional.*
    {ci}
        *Currently non-functional.*
    {n_boot}
        *Currently non-functional.*
    alpha : float
        Proportional opacity of the points.
    {{x,y}}_jitter : booleans or floats
        *Currently non-functional.*
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed down to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    lineplot : Show the relationship between two variables connected with
               lines to emphasize continuity.
    swarmplot : Draw a scatter plot with one categorical variable, arranging
                the points to show the distribution of values.

    Examples
    --------

    Draw a simple scatter plot between two variables:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set()
        >>> import matplotlib.pyplot as plt
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.scatterplot(x="total_bill", y="tip", data=tips)

    Group by another variable and show the groups with different colors:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip", hue="time",
        ...                      data=tips)

    Show the grouping variable by varying both color and marker:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="time", style="time", data=tips)

    Vary colors and markers to show two different grouping variables:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="day", style="time", data=tips)

    Show a quantitative variable by varying the size of the points:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip", size="size",
        ...                      data=tips)

    Also show the quantitative variable by also using continuous colors:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      data=tips)

    Use a different continuous color map:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      palette=cmap,
        ...                      data=tips)

    Change the minimum and maximum point size and show all sizes in legend:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      sizes=(20, 200), palette=cmap,
        ...                      legend="full", data=tips)

    Use a narrower range of color map intensities:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      sizes=(20, 200), hue_norm=(0, 7),
        ...                      legend="full", data=tips)

    Vary the size with a categorical variable, and use a different palette:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="day", size="smoker",
        ...                      palette="Set2",
        ...                      data=tips)

    Use a specific set of markers:

    .. plot::
        :context: close-figs

        >>> markers = {{"Lunch": "s", "Dinner": "X"}}
        >>> ax = sns.scatterplot(x="total_bill", y="tip", style="time",
        ...                      markers=markers,
        ...                      data=tips)

    Control plot attributes using matplotlib parameters:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      s=100, color=".2", marker="+",
        ...                      data=tips)

    Pass data vectors instead of names in a data frame:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> ax = sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width,
        ...                      hue=iris.species, style=iris.species)

    Pass a wide-form dataset and plot against its index:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; plt.close("all")
        >>> index = pd.date_range("1 1 2000", periods=100,
        ...                       freq="m", name="date")
        >>> data = np.random.randn(100, 4).cumsum(axis=0)
        >>> wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
        >>> ax = sns.scatterplot(data=wide_df)

    Use :func:`relplot` to combine :func:`scatterplot` and :class:`FacetGrid`:
    This allows grouping within additional categorical variables. Using
    :func:`relplot` is safer than using :class:`FacetGrid` directly, as it
    ensures synchronization of the semantic mappings across facets.

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip",
        ...                  col="time", hue="day", style="day",
        ...                  kind="scatter", data=tips)


    """).format(**_relational_docs)


def relplot(x=None, y=None, hue=None, size=None, style=None, data=None,
            row=None, col=None, col_wrap=None, row_order=None, col_order=None,
            palette=None, hue_order=None, hue_norm=None,
            sizes=None, size_order=None, size_norm=None,
            markers=None, dashes=None, style_order=None,
            legend="brief", kind="scatter",
            height=5, aspect=1, facet_kws=None, **kwargs):

    if kind == "scatter":

        plotter = _ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers

    elif kind == "line":

        plotter = _LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes

    else:
        err = "Plot kind {} not recognized".format(kind)
        raise ValueError(err)

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = ("relplot is a figure-level function and does not accept "
               "target axes. You may wish to try {}".format(kind + "plot"))
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # Use the full dataset to establish how to draw the semantics
    p = plotter(
        x=x, y=y, hue=hue, size=size, style=style, data=data,
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        markers=markers, dashes=dashes, style_order=style_order,
        legend=legend,
    )

    palette = p.palette if p.palette else None
    hue_order = p.hue_levels if any(p.hue_levels) else None
    hue_norm = p.hue_norm if p.hue_norm is not None else None

    sizes = p.sizes if p.sizes else None
    size_order = p.size_levels if any(p.size_levels) else None
    size_norm = p.size_norm if p.size_norm is not None else None

    markers = p.markers if p.markers else None
    dashes = p.dashes if p.dashes else None
    style_order = p.style_levels if any(p.style_levels) else None

    plot_kws = dict(
        palette=palette, hue_order=hue_order, hue_norm=p.hue_norm,
        sizes=sizes, size_order=size_order, size_norm=p.size_norm,
        markers=markers, dashes=dashes, style_order=style_order,
        legend=False,
    )
    plot_kws.update(kwargs)
    if kind == "scatter":
        plot_kws.pop("dashes")

    # Set up the FacetGrid object
    facet_kws = {} if facet_kws is None else facet_kws
    g = FacetGrid(
        data=data, row=row, col=col, col_wrap=col_wrap,
        row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet_kws
    )

    # Draw the plot
    g.map_dataframe(func, x, y,
                    hue=hue, size=size, style=style,
                    **plot_kws)

    # Show the legend
    if legend:
        p.add_legend_data(g.axes.flat[0])
        if p.legend_data:
            g.add_legend(legend_data=p.legend_data,
                         label_order=p.legend_order)

    return g


relplot.__doc__ = dedent("""\
    Figure-level interface for drawing relational plots onto a FacetGrid.

    This function provides access to several different axes-level functions
    that show the relationship between two variables with semantic mappings
    of subsets. The ``kind`` parameter selects the underlying axes-level
    function to use:

    - :func:`scatterplot` (with ``kind="scatter"``; the default)
    - :func:`lineplot` (with ``kind="line"``)

    Extra keyword arguments are passed to the underlying function, so you
    should refer to the documentation for each to see kind-specific options.

    {main_api_narrative}

    {relational_semantic_narrative}

    After plotting, the :class:`FacetGrid` with the plot is returned and can
    be used directly to tweak supporting plot details or add other layers.

    Note that, unlike when using the underlying plotting functions directly,
    data must be passed in a long-form DataFrame with variables specified by
    passing strings to ``x``, ``y``, and other parameters.

    Parameters
    ----------
    x, y : names of variables in ``data``
        Input data variables; must be numeric.
    hue : name in ``data``, optional
        Grouping variable that will produce elements with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : name in ``data``, optional
        Grouping variable that will produce elements with different sizes.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : name in ``data``, optional
        Grouping variable that will produce elements with different styles.
        Can have a numeric dtype but will always be treated as categorical.
    {data}
    row, col : names of variables in ``data``, optional
        Categorical variables that will determine the faceting of the grid.
    {col_wrap}
    row_order, col_order : lists of strings, optional
        Order to organize the rows and/or columns of the grid in, otherwise the
        orders are inferred from the data objects.
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    {legend}
    kind : string, optional
        Kind of plot to draw, corresponding to a seaborn relational plot.
        Options are {{``scatter`` and ``line``}}.
    {height}
    {aspect}
    facet_kws : dict, optional
        Dictionary of other keyword arguments to pass to :class:`FacetGrid`.
    kwargs : key, value pairings
        Other keyword arguments are passed through to the underlying plotting
        function.

    Returns
    -------
    g : :class:`FacetGrid`
        Returns the :class:`FacetGrid` object with the plot on it for further
        tweaking.

    Examples
    --------

    Draw a single facet to use the :class:`FacetGrid` legend placement:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns
        >>> sns.set(style="ticks")
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.relplot(x="total_bill", y="tip", hue="day", data=tips)

    Facet on the columns with another variable:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip",
        ...                 hue="day", col="time", data=tips)

    Facet on the columns and rows:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="day",
        ...                 col="time", row="sex", data=tips)

    "Wrap" many column facets into multiple rows:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="time",
        ...                 col="day", col_wrap=2, data=tips)

    Use multiple semantic variables on each facet with specified attributes:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="time", size="size",
        ...                 palette=["b", "r"], sizes=(10, 100),
        ...                 col="time", data=tips)

    Use a different kind of plot:

    .. plot::
        :context: close-figs

        >>> fmri = sns.load_dataset("fmri")
        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                 hue="event", style="event", col="region",
        ...                 kind="line", data=fmri)

    Change the size of each facet:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                 hue="event", style="event", col="region",
        ...                 height=5, aspect=.7, kind="line", data=fmri)

    """).format(**_relational_docs)
