"""Tests for utils.py."""
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
    for i, (low, high) in enumerate(cis):
        h = heights[i]
        elow = h - low
        ehigh = high - h
        errsize.append([elow, ehigh])

    errsize = np.transpose(errsize)
    return errsize
