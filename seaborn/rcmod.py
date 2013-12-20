"""Functions that alter the matplotlib rc dictionary on the fly."""
import contextlib
import matplotlib as mpl
from seaborn import utils


def set(context="notebook", style="darkgrid", palette="deep", font="Arial"):
    """Set new RC params in one step."""
    set_axes_style(style, context, font)
    set_color_palette(palette)

def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def set_axes_style(style, context, font="Arial"):
    """Set the axis style.

    Parameters
    ----------
    style : darkgrid | whitegrid | nogrid | ticks
        Style of axis background.
    context: notebook | talk | paper | poster
        Intended context for resulting figures.
    font : matplotlib font spec
        Font to use for text in the figures.

    """
    # Validate the arguments
    if not {"darkgrid", "whitegrid", "nogrid", "ticks"} & {style}:
        raise ValueError("Style %s not recognized" % style)

    if not {"notebook", "talk", "paper", "poster"} & {context}:
        raise ValueError("Context %s is not recognized" % context)

    # Determine the axis parameters
    # -----------------------------

    # Turn ticks off; they get turned back on in 'ticks' style
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)
    mpl.rc("xtick.minor", size=0)
    mpl.rc("ytick.minor", size=0)

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

    elif style == "whitegrid":
        glw = .8 if context == "paper" else 1.5
        ax_params = {"axes.facecolor": "white",
                     "axes.edgecolor": "#CCCCCC",
                     "axes.linewidth": glw + .2,
                     "axes.grid": True,
                     "axes.axisbelow": True,
                     "grid.color": "#DDDDDD",
                     "grid.linestyle": "-",
                     "grid.linewidth": glw}

    elif style == "nogrid":
        ax_params = {"axes.grid": False,
                     "axes.facecolor": "white",
                     "axes.edgecolor": "black",
                     "axes.linewidth": 1}

    elif style == "ticks":
        ticksize = 3. if context == "paper" else 6.
        tickwidth = .5 if context == "paper" else 1
        ax_params = {"axes.grid": False,
                     "axes.facecolor": "white",
                     "axes.edgecolor": "black",
                     "axes.linewidth": 1,
                     "xtick.direction": "out",
                     "ytick.direction": "out",
                     "xtick.major.width": tickwidth,
                     "ytick.major.width": tickwidth,
                     "xtick.minor.width": tickwidth,
                     "xtick.minor.width": tickwidth,
                     "xtick.major.size": ticksize,
                     "xtick.minor.size": ticksize / 2,
                     "ytick.major.size": ticksize,
                     "ytick.minor.size": ticksize / 2}

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
                       "legend.fontsize": 10,
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
        "patch.linewidth": .1 if context == "paper" else .3,
        "xtick.major.pad": 3.5 if context == "paper" else 7,
        "ytick.major.pad": 3.5 if context == "paper" else 7,
        })

    # Set the constant defaults
    mpl.rc("font", family=font)
    mpl.rc("legend", frameon=False, numpoints=1)
    mpl.rc("lines", markeredgewidth=0, solid_capstyle="round")
    mpl.rc("figure", figsize=(8, 5.5))
    mpl.rc("image", cmap="cubehelix")


def set_color_palette(name, n_colors=6, desat=None):
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


@contextlib.contextmanager
def palette_context(palette, n_colors=6, desat=None):
    """Context manager for temporarily setting the color palette."""
    orig_palette = mpl.rcParams["axes.color_cycle"]
    set_color_palette(palette, n_colors, desat)
    yield
    set_color_palette(orig_palette, len(orig_palette))
