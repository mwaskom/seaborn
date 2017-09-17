from itertools import product
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .external.six import string_types

from . import utils
from .utils import categorical_order, remove_na, hue_type, get_color_cycle
from .algorithms import bootstrap
from .palettes import color_palette, husl_palette


__all__ = ["lineplot"]


class _BasicPlotter(object):

    # TODO use different lists for mpl 1 and 2?
    default_markers = ["o", "s", "D", "v", "^", "p"]
    marker_scales = {"o": 1, "s": .85, "D": .9, "v": 1.3, "^": 1.3, "p": 1.25}
    default_dashes = [(np.inf, 1), (4, 1), (1, 1),
                      (4, 1, 2, 1), (5, 1, 1, 1), (5, 1, 2, 1, 2, 1)]

    def establish_variables(self, x=None, y=None,
                            hue=None, style=None, size=None,
                            data=None):

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

            if isinstance(data, pd.DataFrame):

                # Enforce numeric values
                try:
                    data.astype(np.float)
                except ValueError:
                    raise ValueError("A wide-form dataframe must have only "
                                     "numeric values.")

                plot_data = pd.melt(data.assign(x=data.index), "x",
                                    var_name="hue", value_name="y")
                plot_data["style"] = plot_data["hue"]

            # TODO accept a dict and try to coerce to a dataframe?

            # Option 1b:
            # The input data is an array or list
            # ----------------------------------

            else:

                if hasattr(data, "shape"):

                    # The input data is an array(like):
                    # We assign a numeric index to x, the values to y, and
                    # numeric values to both hue and style

                    plot_data = pd.DataFrame(data)
                    plot_data = plot_data.assign(x=plot_data.index)
                    plot_data = pd.melt(plot_data, "x",
                                        var_name="hue", value_name="y")
                    plot_data["style"] = plot_data["hue"]

                elif np.isscalar(data[0]):

                    # The input data is a flat list(like):
                    # We assign a numeric index for x and use the values for y

                    plot_data = pd.DataFrame(dict(x=np.arange(len(data)),
                                                  y=data))

                else:

                    # The input data is a nested list: We will assign a numeric
                    # index for x, use the values for, y and use numieric
                    # hue/style identifiers for each entry.

                    plot_data = pd.concat([
                        pd.DataFrame(dict(x=np.arange(len(data_i)),
                                          y=data_i, hue=i, style=i, size=None))
                        for i, data_i in enumerate(data)
                    ])

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
                style = data.get(style, style)
                size = data.get(size, size)

            # Validate the inputs
            for input in [x, y, hue, style, size]:
                if isinstance(input, string_types):
                    err = "Could not interpret input '{}'".format(input)
                    raise ValueError(err)

            # Reassemble into a DataFrame
            plot_data = dict(x=x, y=y, hue=hue, style=style, size=size)
            plot_data = pd.DataFrame(plot_data)

        # Option 3:
        # Only one variable arugment
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

        if not plot_data.size:
            err = "Input data (after processing) was empty"
            raise ValueError(err)

        self.plot_data = plot_data


class _LinePlotter(_BasicPlotter):

    def __init__(self,
                 x=None, y=None, hue=None, style=None, size=None, data=None,
                 palette=None, clim=None,
                 dashes=None, markers=None, slim=None,
                 estimator=None, ci=None, n_boot=None,
                 sort=True, errstyle=None):

        self.establish_variables(x, y, hue, style, size, data)
        self.determine_attributes(palette, clim, markers, dashes, slim)

        self.sort = sort
        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot
        self.errstyle = errstyle

    def determine_attributes(self,
                             palette=None, clim=None,
                             markers=None, dashes=None,
                             slim=None):

        data = self.plot_data

        self.parse_hue(self.plot_data["hue"], palette, clim)

        if data["size"].isnull().all():
            size_levels = [None]
            sizes = {}

        else:
            size_levels = categorical_order(data["size"])

            if slim is None:
                smin, smax = 1, 3
            else:
                smin, smax = slim
            smax -= smin
            norm = mpl.colors.Normalize(data["size"].min(), data["size"].max())
            sizes = {s: smin + (norm(s) * smax) for s in size_levels}

        if data["style"].isnull().all():
            style_levels = [None]
            dashes = {}
            markers = {}

        else:

            style_levels = categorical_order(data["style"])

            if dashes is True:
                # TODO error on too many levels
                dashes = dict(zip(style_levels, self.default_dashes))
            elif dashes and isinstance(dashes, dict):
                # TODO error on missing levels
                pass
            elif dashes:
                dashes = dict(zip(style_levels, dashes))
            else:
                dashes = {}

            if markers is True:
                # TODO error on too many levels
                markers = dict(zip(style_levels, self.default_markers))
            elif markers and isinstance(markers, dict):
                # TODO error on missing levels
                pass
            elif markers:
                markers = dict(zip(style_levels, markers))
            else:
                markers = {}

        # TODO This doesn't work when attributes share a variable
        attributes = product(self.hue_levels,
                             style_levels, size_levels)

        self.attributes = attributes
        self.dashes = dashes
        self.markers = markers
        self.sizes = sizes

    def parse_hue(self, data, palette, clim):
        """Determine what color palette to use given data characteristics."""

        if data.isnull().all():

            # -- Set default values when not using a hue mapping

            hue_levels = [None]
            palette = {}
            palette_type = None
            cmap = None

        else:

            # -- Determine what kind of hue mapping we want

            # (Default to categorical for wide-form inputs because the hue
            # variable will usually be default integer index values)

            if self.input_format == "wide":
                palette_type = "categorical"
            else:
                palette_type = hue_type(data)

        # -- Option 1: categorical color palette

        if palette_type == "categorical":

            hue_levels = categorical_order(data)
            n_colors = len(hue_levels)
            cmap = None

            if isinstance(palette, dict):

                missing = set(hue_levels) - set(palette)
                if any(missing):
                    msg = ("The palette dictionary is missing keys: {}"
                           .format(missing))
                    raise ValueError(msg)

            elif palette is None:

                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(n_colors=n_colors)
                else:
                    colors = husl_palette(n_colors, l=.7)

            else:
                colors = color_palette(palette, n_colors)

            palette = dict(zip(hue_levels, colors))

        # -- Option 2: sequential color palette

        elif palette_type is not None:

            if palette is None:
                cmap = mpl.cm.get_cmap(plt.rcParams["image.cmap"])
            elif isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = mpl.cm.get_cmap(palette)

            # -- Option 2a: a discrete colormap

            if palette_type == "discrete":

                if clim is None:
                    hue_levels = np.arange(data.min(), data.max() + 1)
                else:
                    hue_levels = np.arange(clim[0], clim[1] + 1)

                vals = np.linspace(0, 1, len(hue_levels))
                colors = cmap(vals)
                palette = dict(zip(hue_levels, colors))

            # -- Option 2b: a continuous colormap

            elif palette_type == "continuous":

                hue_levels = data.unique()
                vals = mpl.colors.Normalize(*clim)(data)
                colors = cmap(vals)
                palette = dict(zip(hue_levels, colors))

        self.hue_levels = hue_levels
        self.palette = palette
        self.palette_type = palette_type
        self.cmap = cmap

    def estimate(self, vals, grouper, func, ci):

        n_boot = self.n_boot

        def bootstrapped_cis(vals):
            if len(vals) == 1:
                return None
            boots = bootstrap(vals, func=func, n_boot=n_boot)
            cis = utils.ci(boots, ci)
            return pd.Series(cis, ["low", "high"])

        # TODO handle ci="sd"

        grouped = vals.groupby(grouper, sort=self.sort)
        est = grouped.apply(func)
        if ci is None:
            return est.index, est, None
        cis = grouped.apply(bootstrapped_cis)
        if cis.notnull().any():
            cis = cis.unstack()
        else:
            cis = None

        return est.index, est, cis

    def plot(self, ax, kws):

        orig_color = kws.pop("color", None)
        orig_dashes = kws.pop("dashes", (np.inf, 1))
        orig_marker = kws.pop("marker", None)
        orig_linewidth = kws.pop("linewidth", kws.pop("lw", None))

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        data = self.plot_data
        all_true = pd.Series(True, data.index)

        for hue, style, size in self.attributes:

            rows = (
                all_true
                & (all_true if hue is None else data["hue"] == hue)
                & (all_true if style is None else data["style"] == style)
                & (all_true if size is None else data["size"] == size)
            )

            subset_data = data.loc[rows]

            # TODO dumb way to handle shared attributes
            if not len(subset_data):
                continue

            subset_data = remove_na(subset_data)

            if self.sort:
                subset_data = subset_data.sort_values(["x", "y"])

            x, y = subset_data["x"], subset_data["y"]

            if self.estimator is not None:
                x, y, y_ci = self.estimate(y, x, self.estimator, self.ci)
            else:
                y_ci = None

            # TODO convert from None to (inf, 0) for dash spec?

            kws["color"] = self.palette.get(hue, orig_color)
            kws["dashes"] = self.dashes.get(style, orig_dashes)
            kws["marker"] = self.markers.get(style, orig_marker)
            kws["linewidth"] = self.sizes.get(size, orig_linewidth)

            # TODO handle marker size adjustment

            line, = ax.plot(x, y, **kws)
            line_color = line.get_color()
            line_alpha = line.get_alpha()

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


class _ScatterPlotter(_BasicPlotter):

    def __init__(self):
        pass

    def plot(self, ax=None):
        pass


_basic_docs = dict(

    main_api_narrative=dedent("""\
    """),

)


def lineplot(x=None, y=None, hue=None, style=None, size=None, data=None,
             palette=None, clim=None, dashes=True, markers=None, slim=None,
             estimator=np.mean, ci=95, n_boot=1000, sort=True, errstyle="band",
             ax=None, **kwargs):

    p = _LinePlotter(
        x=x, y=y, hue=hue, style=style, size=size, data=data,
        palette=palette, clim=clim,
        dashes=dashes, markers=markers,
        slim=slim,
        estimator=estimator, ci=ci, n_boot=n_boot,
        sort=sort, errstyle=errstyle
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax, p  # TODO


def scatterplot(x=None, y=None, hue=None, style=None, size=None, data=None,
                palette=None, clim=None, markers=None, slim=None,
                x_bins=None, y_bins=None, estimator=None, ci=95, n_boot=1000,
                errstyle="bars", alpha="auto", x_jitter=None, y_jitter=None,
                ax=None, **kwargs):

    pass
