from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from seaborn._stats.base import Stat


@dataclass
class PolyFit(Stat):
    """
    Fit a polynomial of the given order and resample data onto predicted curve.
    """

    # This is a provisional class that is useful for building out functionality.
    # It may or may not change substantially in form or dissappear as we think
    # through the organization of the stats subpackage.

    order: int = 2
    gridsize: int = 100

    def _fit_predict(self, data):
        x = data["x"]
        y = data["y"]
        if x.nunique() <= self.order:
            # TODO warn?
            xx = yy = []
        else:
            p = np.polyfit(x, y, self.order)
            xx = np.linspace(x.min(), x.max(), self.gridsize)
            yy = np.polyval(p, xx)

        return pd.DataFrame(dict(x=xx, y=yy))

    # TODO we should have a way of identifying the method that will be applied
    # and then only define __call__ on a base-class of stats with this pattern

    def __call__(self, data, groupby, orient, scales):
        return groupby.apply(data.dropna(subset=["x", "y"]), self._fit_predict)


@dataclass
class OLSFit(Stat):
    ...


import statsmodels.api as sm

@dataclass
class Lowess(Stat):
    """
    Perform locally-weighted regression (LOWESS) to smooth data.

    This statistical method allows fitting a smooth curve to your data
    using a local regression. It can be useful to visualize the trend of the data.

    Parameters
    ----------
    frac : float
        The fraction of data used when estimating each y-value.
    gridsize : int
        The number of points in the grid to which the LOWESS is applied.
        Higher values result in a smoother curve.
    it : int
        The number of robustifying iterations which should be performed.
    delta : float
        Distance within which to use linear-interpolation instead of weighted regression.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the smoothed curve's 'x' and 'y' coordinates.
    """

    frac: float = 0.2
    gridsize: int = 100
    it: int = 3
    delta: float = 0.0
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the grid for x values once after the object is created."""
        self._x = np.linspace(self._x.min(), self._x.max(), self.gridsize)

    def _fit_predict(self):
        """Apply LOWESS and return the estimated y values."""
        # Perform LOWESS smoothing
        smoothed = sm.nonparametric.lowess(
            endog=self._y, exog=self._x, frac=self.frac, it=self.it, delta=self.delta
        )
        # Return a DataFrame with the smoothed curve's coordinates
        return pd.DataFrame(smoothed, columns=["x", "y"])

    def __call__(self, data, groupby, orient, scales):
        # Ensure data is valid and contains no NaNs in 'x' and 'y'
        valid_data = data.dropna(subset=["x", "y"])

        # Extract x and y values for LOWESS
        self._x = valid_data["x"].to_numpy()
        self._y = valid_data["y"].to_numpy()

        # Apply the LOWESS to each group
        return groupby.apply(valid_data, self._fit_predict)
