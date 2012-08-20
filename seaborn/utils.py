"""Small plotting-related utility functions."""
from __future__ import division
import colorsys
import numpy as np
import matplotlib.colors as mplcol


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
