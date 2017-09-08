from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .external.six import string_types

from .utils import categorical_order, hue_type, get_color_cycle
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
                # We will assign a numeric index for x, and use the values for y

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
            plot_data = {k: v for k, v in plot_data.items() if v is not None}
            plot_data = pd.DataFrame(plot_data)

        self.plot_data = plot_data

    def determine_attributes(self, palette=None, clims=None, markers=None,
                             dashes=None, slims=None):

        hue_subset_masks = []

        if "hue" in self.plot_data:

            hue_data = self.plot_data["hue"]
            type = hue_type(hue_data)

            if type == "categorical":

                hue_levels = categorical_order(hue_data)
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
                    self.palette = dict(zip(hue_levels, colors))

                
                for hue in hue_levels:
                    hue_subset_masks.append(
                        (hue, self.plot_data["hue"] == hue)
                    )
            else:

                if palette is None:
                    cmap_name = plt.rcParams["image.cmap"]

                elif isinstance(palette, mpl.colors.Colormap):

                    pass  # TODO

        self.subset_masks = hue_subset_masks
        

class _LinePlotter(_BasicPlotter):

    def __init__(self,
                 x=None, y=None, hue=None, style=None, size=None, data=None,
                 x_estimator=None, x_ci=95, y_estimator=None, y_ci=None,
                 palette=None, clims=None, markers=None, dashes=None, slims=None,
                 sort=True, errstyle="bars"):

        self.establish_variables(x, y, hue, style, size, data)
        self.determine_attributes(palette, clims, markers, dashes, slims)

        self.sort = sort


    def plot(self, ax, kws):

        orig_color = kws.pop("color", None)

        for (hue, subset_mask) in self.subset_masks:

            subset_data = self.plot_data.loc[subset_mask]

            if self.sort:
                subset_data = subset_data.sort_values(["x", "y"])

            kws["color"] = self.palette[hue]

            ax.plot(subset_data["x"], subset_data["y"],
                    **kws)

        


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
             x_estimator=None, x_ci=95, y_estimator=None, y_ci=None,
             palette=None, clims=None, markers=None, dashes=None, slims=None,
             sort=True, errstyle="bars",
             ax=None, **kwargs):

    p = _LinePlotter(
        x=x, y=y, hue=hue, style=style, size=size, data=data,
        x_estimator=x_estimator, x_ci=x_ci,
        y_estimator=y_estimator, y_ci=y_ci,
        palette=palette, clims=clims,
        markers=markers, dashes=dashes,
        slims=slims,
        sort=sort, errstyle=errstyle
    )

    if ax is None:
        ax = plt.gca()

    p.plot(ax, kwargs)

    return ax, p  # TODO
                     


def scatterplot(x=None, y=None, hue=None, style=None, size=None, data=None,
                x_estimator=None, x_ci=95, y_estimator=None, y_ci=None,
                palette=None, clims=None, markers=None, slims=None,
                errstyle="bars", alpha="auto",
                ax=None, **kwargs):

    pass
