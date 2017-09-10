from itertools import product
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .external.six import string_types

from . import utils
from .utils import categorical_order, hue_type, get_color_cycle
from .algorithms import bootstrap
from .palettes import color_palette, husl_palette


class _BasicPlotter(object):

    # TODO use different lists for mpl 1 and 2
    default_markers = ["o", "s", "D", "v", "^", "p"]
    marker_scales = {"o": 1, "s": .85, "D": .9, "v": 1.3, "^": 1.3, "p": 1.25}
    default_dashes = [(np.inf, 1), (5, 1), (4, 1, 2, 1),
                      (2, 1), (5, 1, 1, 1), (5, 1, 2, 1, 2, 1)]

    def establish_variables(self, x=None, y=None,
                            hue=None, style=None, size=None,
                            data=None):

        # Option 1:
        # We have a wide-form datast
        # --------------------------

        if x is None and y is None:

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

            # Option 1b:
            # The input data is an array or list
            # ----------------------------------

            else:

                # The input data is an array:
                # We will assign a numeric index to x, the values to y, and
                # numeric values to both hue and style

                if hasattr(data, "shape"):

                    plot_data = pd.DataFrame(data)
                    plot_data = pd.melt(plot_data.assign(x="data.index"), "x",
                                        var_name="hue", value_name="y")
                    plot_data["style"] = plot_data["hue"]

                # The input data is a flat list:
                # We will assign a numeric index for x and use the values for y

                elif np.isscalar(data[0]):

                    plot_data = pd.DataFrame(dict(x=np.arange(len(data)),
                                                  y=data))

                # The input data is a nested list:
                # We will assign a numeric index for x, use the values for, y
                # and use numieric hue/style identifiers for each entry.

                else:

                    plot_data = pd.concat([
                        pd.DataFrame(dict(x=np.arange(len(data_i)),
                                          y=data_i, hue=i, style=i))
                        for i, data_i in enumerate(data)
                    ])

        # Option 2:
        # We have long-form data
        # ----------------------

        else:

            # See if we need to get variables from `data`
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

            plot_data = dict(x=x, y=y, hue=hue, style=style, size=size)
            plot_data = pd.DataFrame(plot_data)

        self.plot_data = plot_data


class _LinePlotter(_BasicPlotter):

    def __init__(self,
                 x=None, y=None, hue=None, style=None, size=None, data=None,
                 palette=None, clims=None,
                 markers=None, dashes=None, slims=None,
                 estimator=None, ci=None, n_boot=None,
                 sort=True, errstyle="band"):

        self.establish_variables(x, y, hue, style, size, data)
        self.determine_attributes(palette, clims, markers, dashes, slims)

        self.sort = sort
        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot

    def determine_attributes(self,
                             palette=None, clims=None,
                             markers=None, dashes=None,
                             slims=None):

        data = self.plot_data

        if data["hue"].isnull().all():
            hue_levels = [None]
            palette = {}
        else:
            type = hue_type(data["hue"])

            if type == "categorical":

                hue_levels = categorical_order(data["hue"])
                n_colors = len(hue_levels)

                if isinstance(palette, dict):
                    missing = set(hue_levels) - set(palette)
                    if any(missing):
                        msg = ("palette dictionary is missing keys: {}"
                               .format(missing))
                        raise ValueError(msg)
                else:
                    if palette is None:

                        # Determine whether the current palette will have
                        # enough values If not, we'll default to the husl
                        # palette so each is distinct
                        if n_colors <= len(get_color_cycle()):
                            colors = color_palette(n_colors=n_colors)
                        else:
                            colors = husl_palette(n_colors, l=.7)
                    else:
                        colors = color_palette(palette, n_colors)
                    palette = dict(zip(hue_levels, colors))

            else:

                if palette is None:
                    cmap_name = plt.rcParams["image.cmap"]

                elif isinstance(palette, mpl.colors.Colormap):

                    pass  # TODO

        if data["size"].isnull().all():
            size_levels = [None]
            sizes = {}

        else:
            size_levels = categorical_order(data["size"])

            if slims is None:
                smin, smax = 1, 3
            else:
                smin, smax = slims
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

        self.attributes = product(hue_levels, style_levels, size_levels)

        self.palette = palette
        self.dashes = dashes
        self.markers = markers
        self.sizes = sizes

    def estimate(self, vals, grouper, func, ci):

        n_boot = self.n_boot

        def bootstrapped_cis(vals):
            boots = bootstrap(vals, n_boot=n_boot)
            cis = utils.ci(boots, ci)
            return pd.Series(cis, ["low", "high"])

        # TODO handle ci="sd"

        grouped = vals.groupby(grouper)
        est = grouped.apply(func)
        cis = grouped.apply(bootstrapped_cis)
        return est.index, est, cis.unstack()

    def plot(self, ax, kws):

        orig_color = kws.pop("color", None)
        orig_dashes = kws.pop("dashes", (np.inf, 1))
        orig_marker = kws.pop("marker", None)
        orig_linewidth = kws.pop("linewidth", kws.pop("lw", None))

        kws.setdefault("markeredgewidth", .75)
        kws.setdefault("markeredgecolor", "w")

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

            if self.sort:
                subset_data = subset_data.sort_values(["x", "y"])

            x, y = subset_data["x"], subset_data["y"]

            if self.estimator is not None:
                x, y, y_ci = self.estimate(y, x, self.estimator, self.ci)
            else:
                y_ci = None

            kws["color"] = self.palette.get(hue, orig_color)
            kws["dashes"] = self.dashes.get(style, orig_dashes)
            kws["marker"] = self.markers.get(style, orig_marker)
            kws["linewidth"] = self.sizes.get(size, orig_linewidth)

            # TODO handle marker size adjustment

            if y_ci is not None:
                ax.fill_between(x, y_ci["low"], y_ci["high"],
                                color=kws["color"], alpha=.2)

            ax.plot(x, y, **kws)


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
             palette=None, clims=None, markers=None, dashes=None, slims=None,
             estimator=None, ci=95, n_boot=1000, sort=True, errstyle="bars",
             ax=None, **kwargs):

    p = _LinePlotter(
        x=x, y=y, hue=hue, style=style, size=size, data=data,
        palette=palette, clims=clims,
        markers=markers, dashes=dashes,
        slims=slims,
        estimator=estimator, ci=ci, n_boot=n_boot,
        sort=sort, errstyle=errstyle
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax, p  # TODO


def scatterplot(x=None, y=None, hue=None, style=None, size=None, data=None,
                palette=None, clims=None, markers=None, slims=None,
                x_bins=None, n_bins=None, estimator=None, ci=95, n_boot=1000,
                errstyle="bars", alpha="auto",
                ax=None, **kwargs):

    pass
