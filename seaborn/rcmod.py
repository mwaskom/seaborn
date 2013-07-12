"""Functions that alter the matplotlib rc dictionary on the fly."""
import matplotlib as mpl
from seaborn import utils


def set(context="notebook", style="darkgrid", palette="deep", font="Arial"):
    """Set new RC params in one step."""
    context_setting(context)
    axes_style(style)
    set_color_palette(palette)
    params = {"figure.figsize": (8, 5.5),
              "lines.linewidth": 1.4,
              "patch.linewidth": .3,
              "font.family": font}
    mpl.rcParams.update(params)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def axes_style(style):
    """Set the axis style.

    Parameters
    ----------
    style : darkgrid | whitegrid | nogrid
        Style of axis background.

    """
    grid_params = {"axes.grid": True,
                   "axes.axisbelow": True}

    if style == "darkgrid":
        grid_params.update({"axes.facecolor": "#EAEAF2",
                            "axes.linewidth": 0,
                            "grid.color": "w",
                            "grid.linestyle": "-",
                            "grid.linewidth": 1.5})
        _blank_ticks(grid_params)
        mpl.rcParams.update(grid_params)

    elif style == "whitegrid":
        grid_params.update({"axes.facecolor": "white",
                            "axes.linewidth": 1,
                            "grid.color": "#222222",
                            "grid.linestyle": ":",
                            "grid.linewidth": .8})
        _restore_ticks(grid_params)
        mpl.rcParams.update(grid_params)

    elif style == "nogrid":
        params = {"axes.grid": False,
                  "axes.facecolor": "white",
                  "axes.linewidth": 1}
        _restore_ticks(params)
        mpl.rcParams.update(params)

    else:
        raise ValueError("Style %s not recognized" % style)


def context_setting(context):
    """Set some visual parameters based on intended context.

    Currently just changes font sizes

    Parameters
    ----------
    context: notebook | talk | paper
        Intended context for resulting figures.

    """
    if context == "talk":
        params = {"axes.labelsize": 16,
                  "axes.titlesize": 19,
                  "xtick.labelsize": 14,
                  "ytick.labelsize": 14,
                  "legend.fontsize": 13,
                  }

    elif context == "notebook":
        params = {"axes.labelsize": 11,
                  "axes.titlesize": 12,
                  "xtick.labelsize": 10,
                  "ytick.labelsize": 10,
                  }

    elif context == "poster":
        params = {"axes.labelsize": 18,
                  "axes.titlesize": 22,
                  "xtick.labelsize": 16,
                  "ytick.labelsize": 16,
                  }

    elif context == "paper":
        params = {"axes.labelsize": 10,
                  "axes.titlesize": 13,
                  "xtick.labelsize": 10,
                  "ytick.labelsize": 10,
                  }

    else:
        raise ValueError("Context %s is not recognized" % context)

    mpl.rcParams.update(params)


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
