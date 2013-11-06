"""Functions that alter the matplotlib rc dictionary on the fly."""
import numpy as np
import matplotlib as mpl

from . import palettes

def set(context="notebook", style="darkgrid", palette="deep", font="Arial"):
    """Set new RC params in one step."""
    set_axes_style(style, context)
    set_color_palette(palette)

    # Set the constant defaults
    mpl.rc("font", family=font)
    mpl.rc("legend", frameon=False, numpoints=1)
    mpl.rc("lines", markeredgewidth=0)
    mpl.rc("figure", figsize=(8, 5.5))
    mpl.rc("image", cmap="cubehelix")
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def reset_orig():
    """Restore all RC params to original settings (respects custom rc)."""
    mpl.rcParams.update(mpl.rcParamsOrig)


class _AxesStyle(dict):
    """Light wrapper on a dict to set style temporarily."""
    def __enter__(self):
        """Open the context."""
        rc = mpl.rcParams
        self._orig_style = {k: rc[k] for k in _style_keys}
        set_style(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        set_style(self._orig_style)


class _PlottingContext(dict):
    """Light wrapper on a dict to set context temporarily."""
    def __enter__(self):
        """Open the context."""
        rc = mpl.rcParams
        self._orig_context = {k: rc[k] for k in _context_keys}
        set_context(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        set_context(self._orig_context)


def axes_style(style=None, rc=None):
    """Return a parameter dict for the aesthetic style of the plots.

    This affects things like the color of the axes, whether a grid is
    enabled by default, and other aesthetic elements.

    This function returns an object that can be used in a ``with`` statement
    to temporarily change the style parameters.

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.

    Examples
    --------
    >>> st = axes_style("whitegrid")

    >>> set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    >>> import matplotlib.pyplot as plt
    >>> with axes_style("white"):
    ...     f, ax = plt.subplots()
    ...     ax.plot(x, y)               # doctest: +SKIP

    See Also
    --------
    set_style : set the matplotlib parameters for a seaborn theme
    plotting_context : return a parameter dict to to scale plot elements
    color_palette : define the color palette for a plot

    """
    # Validate the arguments
    if not {"darkgrid", "whitegrid", "nogrid"} & {style}:
        raise ValueError("Style %s not recognized" % style)

    if not {"notebook", "talk", "paper", "poster"} & {context}:
        raise ValueError("Context %s is not recognized" % context)

    # Determine the axis parameters
    if style == "darkgrid":
        lw = .8 if context == "paper" else 1.5
        ax_params = {"axes.facecolor": "#EAEAF2",
                     "axes.edgecolor": "white",
                     "axes.linewidth": 0,
                     "axes.grid": True,
                     "axes.axisbelow": True,
                     "grid.color": "w",
                     "grid.linestyle": "-",
                     "grid.linewidth": lw}
        _blank_ticks(ax_params)

    elif style == "whitegrid":
        glw = .8 if context == "paper" else 1.5
        ax_params = {"axes.facecolor": "white",
                     "axes.edgecolor": "#CCCCCC",
                     "axes.linewidth": glw + .2,
                     "axes.grid": True,
                     "axes.axisbelow": True,
                     "grid.color": "#CCCCCC",
                     "grid.linestyle": "-",
                     "grid.linewidth": glw}
        _blank_ticks(ax_params)

    elif style == "nogrid":
        ax_params = {"axes.grid": False,
                     "axes.facecolor": "white",
                     "axes.edgecolor": "black",
                     "axes.linewidth": 1}
        _restore_ticks(ax_params)

    mpl.rcParams.update(ax_params)

    # Determine the font sizes
    if context == "talk":
        font_params = {"axes.labelsize": 16,
                       "axes.titlesize": 19,
                       "xtick.labelsize": 14,
                       "ytick.labelsize": 14,
                       "legend.fontsize": 13,
                       }

    elif context == "notebook":
        font_params = {"axes.labelsize": 11,
                       "axes.titlesize": 12,
                       "xtick.labelsize": 10,
                       "ytick.labelsize": 10,
                       "legend.fontsize": 9,
                       }

    elif context == "poster":
        font_params = {"axes.labelsize": 18,
                       "axes.titlesize": 22,
                       "xtick.labelsize": 16,
                       "ytick.labelsize": 16,
                       "legend.fontsize": 16,
                       }

    elif context == "paper":
        font_params = {"axes.labelsize": 8,
                       "axes.titlesize": 12,
                       "xtick.labelsize": 8,
                       "ytick.labelsize": 8,
                       "legend.fontsize": 8,
                       }

    mpl.rcParams.update(font_params)

    # Set other parameters
    mpl.rcParams.update({
        "lines.linewidth": 1.1 if context == "paper" else 1.4,
        "patch.linewidth": .1 if context == "paper" else .3
        })


def set_color_palette(name, n_colors=6, desat=None):
    """Set the matplotlib color cycle in one of a variety of ways.

    Parameters
    ----------
    name : hls | husl | matplotlib colormap | seaborn color palette
        Palette definition. Should be something that :func:`color_palette`
        can process.
    n_colors : int
        Number of colors in the cycle.
    desat : float
        desaturation factor for each color

    """
    colors = utils.color_palette(name, n_colors, desat)
    mpl.rcParams["axes.color_cycle"] = colors
    mpl.rcParams["patch.facecolor"] = colors[0]
