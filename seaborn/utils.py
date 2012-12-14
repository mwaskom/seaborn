"""Small plotting-related utility functions."""
from __future__ import division
import colorsys
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplcol


def color_palette(name=None, n_colors=8, desat=None, h=.01, l=.6, s=.65):
    """Return matplotlib color codes for a given palette.

    Parameters
    ----------
    name: None or string
        Name of palette or None to return current color list
    n_colors : int
        number of colors in the palette
    desat : float
        desaturation factor for each color
    h : float
        first hue
    l : float
        lightness
    s : float
        saturation

    Returns
    -------
    palette : list of colors
        color palette

    """
    if name is None:
        return mpl.rcParams["axes.color_cycle"]

    palettes = dict(
        default=["b", "g", "r", "c", "m", "y", "k"],
        pastel=["#92C6FF", "#97F0AA", "#FF9F9A", "#D0BBFF", "#FFFEA3"],
        bright=["#003FFF", "#03ED3A", "#E8000B", "#00D7FF", "#FFB400"],
        muted=["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"],
        deep=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
        dark=["#001C7F", "#017517", "#8C0900", "#7600A1", "#007364"],
        colorblind=["#0072B2", "#009E73", "#D55E00", "#F0E442",
                    "#CC79A7", "#56B4E9", "#E69F00"],
    )

    if name == "hls":
        palette = hls_palette(n_colors, h, l, s)
    else:
        try:
            palette =  palettes[name]
        except KeyError:
            bins = np.linspace(0, 1, n_colors + 2)[1:-1]
            cmap = getattr(mpl.cm, name)
            palette = map(tuple, cmap(bins)[:, :3])
        except KeyError:
            raise ValueError("%s is not a valid palette name" % name)

    if desat is not None:
        palette = [desaturate(c, desat) for c in palette]

    return palette


def hls_palette(n_colors=6, h=.01, l=.6, s=.65):
    """Get a set of evenly spaced colors in HLS hue space.

    Parameters
    ----------

    n_colors : int
        number of colors in the palette
    h : float
        first hue
    l : float
        lightness
    s : float
        saturation

    Returns
    -------
    palette : list of tuples
        color palette

    """
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    hues += h
    hues -= hues.astype(int)
    palette = [colorsys.hls_to_rgb(h_i, l, s) for h_i in hues]
    return palette


def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.

    Parameters
    ----------
    cis: 2 x n sequence
        sequence of confidence interval limits
    heights : n sequence
        sequence of plot heights

    Returns
    -------
    errsize : 2 x n array
        sequence of error size relative to height values in correct
        format as argument for plt.bar

    """
    errsize = []
    for i, (low, high) in enumerate(np.transpose(cis)):
        h = heights[i]
        elow = h - low
        ehigh = high - h
        errsize.append([elow, ehigh])

    errsize = np.asarray(errsize).T
    return errsize


def pmf_hist(a, bins=10):
    """Return arguments to plt.bar for pmf-like histogram of an array.

    Parameters
    ----------
    a: array-like
        array to make histogram of
    bins: int
        number of bins

    Returns
    -------
    x: array
        left x position of bars
    h: array
        height of bars
    w: float
        width of bars

    """
    n, x = np.histogram(a, bins)
    h = n / n.sum()
    w = x[1] - x[0]
    return x[:-1], h, w


def desaturate(color, pct, space="hsv"):
    """Decrease the saturation channel of a color by some percent.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    pct : float
        saturation channel of color will be multiplied by this value
    space : hsv | hls
        intermediate color space to max saturation channel

    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation

    """
    # Check inputs
    if not 0 < pct < 1:
        raise ValueError("Pct must be between 0 and 1")

    # Get rgb tuple rep
    rgb = mplcol.colorConverter.to_rgb(color)

    # Get the parameters to map in and out of hue-based space
    sat_chan, map_in, map_out = _hue_space_params(space)

    # Map into the space, desaturate, map back out and return
    inter_rep = list(map_in(*rgb))
    inter_rep[sat_chan] *= pct
    new_color = map_out(*inter_rep)
    return new_color


def saturate(color, space="hsv"):
    """Return a fully saturated color with the same hue.

    Parameters
    ----------
    color :  matplotlib color
        hex, rgb-tuple, or html color name
    space : hsv | hls
        intermediate color space to max saturation channel

    Returns
    -------
    new_color : rgb tuple
        saturated color code in RGB tuple representation

    """
    # Get rgb tuple rep
    rgb = mplcol.colorConverter.to_rgb(color)

    # Get the parameters to map in and out of hue-based space
    sat_chan, map_in, map_out = _hue_space_params(space)

    # Map into the space, desaturate, map back out and return
    inter_rep = list(map_in(*rgb))
    inter_rep[sat_chan] = 1
    new_color = map_out(*inter_rep)
    return new_color


def _hue_space_params(space):
    """Get parameters to go in and out of hue-based color space."""
    try:
        sat_chan = dict(hsv=1, hls=2)[space]
    except KeyError:
        raise ValueError(space + " is not a valid space value")

    # Get the right function to map into a space with a
    # saturation channel
    map_in = getattr(colorsys, "rgb_to_" + space)
    map_out = getattr(colorsys, space + "_to_rgb")

    return sat_chan, map_in, map_out
