"""Functions that alter the matplotlib rc dictionary on the fly."""
import warnings
from numpy import isreal
import matplotlib as mpl

from . import palettes

from . import palettes

style_keys = (
    "axes.facecolor",
    "axes.edgecolor",
    "axes.grid",
    "axes.axisbelow",
    "axes.linewidth",

    "grid.color",
    "grid.linestyle",

    "xtick.direction",
    "ytick.direction",
    "xtick.major.size",
    "ytick.major.size",
    "xtick.minor.size",
    "ytick.minor.size",

    "legend.frameon",
    "legend.numpoints",
    "legend.scatterpoints",

    "lines.solid_capstyle",

    "image.cmap",
    )

context_keys = (
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",

    "grid.linewidth",
    "lines.linewidth",
    "patch.linewidth",
    "lines.markeredgewidth",

    "xtick.major.width",
    "ytick.major.width",
    "xtick.minor.width",
    "ytick.minor.width",

    "xtick.major.pad",
    "ytick.major.pad"
    )


def set(context="notebook", style="darkgrid", palette="deep", font="Arial",
        gridweight=None):
    """Set new RC params in one step."""
    set_axes_style(style, context, font=font, gridweight=gridweight)
    set_color_palette(palette)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def reset_orig():
    """Restore all RC params to original settings (respects custom rc)."""
    mpl.rcParams.update(mpl.rcParamsOrig)


class _AxesStyle(dict):

    def __enter__(self):
        """Open the context."""
        rc = mpl.rcParams
        self._orig_style = {k: rc[k] for k in style_keys}
        set_style(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        set_style(self._orig_style)


def axes_style(style=None, rc=None):

    if style is None:
        style_dict = {k: mpl.rcParams[k] for k in style_keys}

    elif isinstance(style, dict):
        style_dict = style

    else:

        # Backwards compatibility
        if style == "nogrid":
            style = "white"
            warnings.warn("The 'nogrid' style is now named 'white'. "
                          "Please update your code", UserWarning)

        styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
        if style not in styles:
            raise ValueError("style must be one of %s" % ", ".join(styles))

        # Common parameters
        style_dict = {
            "legend.frameon": False,
            "legend.numpoints": 1,
            "legend.scatterpoints": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.axisbelow": True,
            "image.cmap": "Greys",
            "grid.linestyle": "-",
            "lines.solid_capstyle": "round",
            }

        # Set grid on or off
        if "grid" in style:
            style_dict.update({
                "axes.grid": True,
                })
        else:
            style_dict.update({
                "axes.grid": False,
                })

        # Set the color of the background, spines, and grids
        if style.startswith("dark"):
            style_dict.update({
                "axes.facecolor": "#EAEAF2",
                "axes.edgecolor": "white",
                "axes.linewidth": 0,
                "grid.color": "white",
                })

        elif style == "whitegrid":
            style_dict.update({
                "axes.facecolor": "white",
                "axes.edgecolor": ".7",
                "axes.linewidth": 0,
                "grid.color": ".8",
                })

        elif style in ["white", "ticks"]:
            style_dict.update({
                "axes.facecolor": "white",
                "axes.edgecolor": ".2",
                "axes.linewidth": 1.25,
                "grid.color": ".8",
                })

        # Show or hide the axes ticks
        if style == "ticks":
            style_dict.update({
                "axes.facecolor": "white",
                "axes.edgecolor": ".2",
                "grid.color": ".8",
                "xtick.major.size": 6,
                "ytick.major.size": 6,
                "xtick.minor.size": 3,
                "ytick.minor.size": 3,
                })
        else:
            style_dict.update({
                "xtick.major.size": 0,
                "ytick.major.size": 0,
                "xtick.minor.size": 0,
                "ytick.minor.size": 0,
                })

    # Override these settings with the provided rc dictionary
    if rc is not None:
        rc = {k: v for k, v in rc.items() if k in style_keys}
        style_dict.update(rc)

    # Wrap in an _AxesStyle object so this can be used in a with statement
    style_object = _AxesStyle(style_dict)

    return style_object


def set_style(style, rc=None):

    style_object = axes_style(style, rc)
    mpl.rcParams.update(style_object)


def set_axes_style(style, context, font, gridweight):
    """Set the axis style.

    Parameters
    ----------
    style : darkgrid | whitegrid | nogrid | ticks
        Style of axis background.

    """

    # Validate the arguments
    if not {"darkgrid", "whitegrid", "nogrid", "ticks"} & {style}:
        raise ValueError("Style %s not recognized" % style)

    if not {"notebook", "talk", "paper", "poster"} & {context}:
        raise ValueError("Context %s is not recognized" % context)

    if not isreal(gridweight) and \
       (not {"None", "extra heavy", "heavy",
             "medium", "light"} & {gridweight}):
        raise ValueError("Gridweight %s is not recognized" % gridweight)

    # Determine the axis parameters
    # -----------------------------

    # Turn ticks off; they get turned back on in 'ticks' style
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)
    mpl.rc("xtick.minor", size=0)
    mpl.rc("ytick.minor", size=0)

    # select grid line width:
    gridweights = {
        'extra heavy': 1.5,
        'heavy': 1.1,
        'medium': 0.8,
        'light': 0.5,
    }
    if gridweight is None:
        if context == "paper":
            glw = gridweights["medium"]
        else:
            glw = gridweights['extra heavy']
    elif isreal(gridweight):
        glw = gridweight
    else:
        glw = gridweights[gridweight]

    if style == "darkgrid":
        lw = .8 if context == "paper" else 1.5
        ax_params = {"axes.facecolor": "#EAEAF2",
                     "axes.edgecolor": "white",
                     "axes.linewidth": 0,
                     "axes.grid": True,
                     "axes.axisbelow": True,
                     "grid.color": "w",
                     "grid.linestyle": "-",
                     "grid.linewidth": glw}

    elif style == "whitegrid":
        lw = 1.0 if context == "paper" else 1.7
        ax_params = {"axes.facecolor": "white",
                     "axes.edgecolor": "#CCCCCC",
                     "axes.linewidth": lw,
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
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)
    mpl.rc("xtick.minor", size=0)
    mpl.rc("ytick.minor", size=0)


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
        Factor to desaturate each color by.

    Examples
    --------
    >>> set_palette("Reds")

    >>> set_palette("Set1", 8, .75)

    See Also
    --------
    color_palette : build a color palette or set the color cycle temporarily
                    in a ``with`` statement.
    set_context : set parameters to scale plot elements
    set_style : set the default parameters for figure style

    """
    colors = palettes.color_palette(name, n_colors, desat)
    mpl.rcParams["axes.color_cycle"] = list(colors)
    mpl.rcParams["patch.facecolor"] = colors[0]


def palette_context(palette, n_colors=6, desat=None):
    """Context manager for temporarily setting the color palette."""
    warnings.warn("palette_context is deprecated, use color_palette directly.",
                  UserWarning)
    return palettes.color_palette(palette, n_colors, desat)
