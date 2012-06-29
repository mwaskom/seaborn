"""Tests for utils.py."""
from __future__ import division
import numpy as np


def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.

    Parameters
    ----------
    cis: n x 2 sequence
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
