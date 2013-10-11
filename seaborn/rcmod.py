"""Functions that alter the matplotlib rc dictionary on the fly."""
import matplotlib as mpl
from seaborn import utils


def set(context="notebook", style="darkgrid", palette="deep", font="Arial"):
    """Set new RC params in one step."""
    set_axes_style(style, context)
    set_color_palette(palette)
    mpl.rc("font", family=font)
    mpl.rc("legend", frameon=False, numpoints=1)
    mpl.rc("lines", markeredgewidth=0)
    mpl.rc("figure", figsize=(8, 5.5))
    mpl.rc("image", cmap="CMRmap")


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_axes_style(style, context):
    """Set the axis style.

    Parameters
    ----------
    style : darkgrid | whitegrid | nogrid
        Style of axis background.
    context: notebook | talk | paper | poster
        Intended context for resulting figures.

    """
    # Validate the arguments
    if not {"darkgrid", "whitegrid", "nogrid"} & {style}:
        raise ValueError("Style %s not recognized" % style)

    if not {"notebook", "talk", "paper", "poster"} & {context}:
        raise ValueError("Context %s is not recognized" % context)

    # Determine the axis parameters
    if style == "darkgrid":
        lw = .8 if context  == "paper" else 1.5
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
                     "axes.edgecolor": "white",
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


def set_color_palette(name, n_colors=8, desat=None):
    """Set the matplotlib color cycle in one of a variety of ways.

    Parameters
    ----------
    name : hls | husl | matplotlib colormap | seaborn color palette
        palette name
    n_colors : int
        only relevant for hls or matplotlib palettes
    desat : float
        desaturation factor for each color

    """
    colors = utils.color_palette(name, n_colors, desat)
    mpl.rcParams["axes.color_cycle"] = colors
    mpl.rcParams["patch.facecolor"] = colors[0]


def _blank_ticks(params):
    """Turn off x and y ticks in a param dict (but not labels)."""
    for axis in ["x", "y"]:
        for step in ["major", "minor"]:
            params["%stick.%s.size" % (axis, step)] = 0


def _restore_ticks(params):
    """Reset x and y ticks in a param dict to matplotlib defaults."""
    for axis in ["x", "y"]:
        for step, size in zip(["major", "minor"], [4, 2]):
            params["%stick.%s.size" % (axis, step)] = size
