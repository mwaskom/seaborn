from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
try:
    from scipy.stats import gaussian_kde
    _no_scipy = False
except ImportError:
    from .external.kde import gaussian_kde
    _no_scipy = True

from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat


@dataclass
class KDE(Stat):
    """
    Compute a univariate kernel density estimate.

    Parameters
    ----------
    bw_adjust : float
        Factor that multiplicatively scales the value chosen using
        ``bw_method``. Increasing will make the curve smoother. See Notes.
    bw_method : string, scalar, or callable
        Method for determining the smoothing bandwidth to use; passed to
        :class:`scipy.stats.gaussian_kde`.
    common_norm : bool or list of variables
        If `True`, normalize so that the sum of all curves sums to 1.
        If `False`, normalize each curve independently. If a list, defines
        variable(s) to group by and normalize within.
    common_grid : bool or list of variables
        If `True`, all curves will share the same evaluation grid.
        If `False`, each evaluation grid is independent. If a list, defines
        variable(s) to group by and share a grid within.
    gridsize : int or None
        Number of points in the evaluation grid. If None, the
        density is evaluated at the original datapoints.
    cut : float
        Factor, multiplied by the smoothing bandwidth, that determines how far
        the evaluation grid extends past the extreme datapoints. When set to 0,
        the curve is trunacted at the data limits.
    cumulative : bool
        If True, estimate a cumulative distribution function. Requires scipy.

    Notes
    -----
    The *bandwidth*, or standard deviation of the smoothing kernel, is an
    important parameter. Misspecification of the bandwidth can produce a
    distorted representation of the data. Much like the choice of bin width in a
    histogram, an over-smoothed curve can erase true features of a distribution,
    while an under-smoothed curve can create false features out of random
    variability. The rule-of-thumb that sets the default bandwidth works best
    when the true distribution is smooth, unimodal, and roughly bell-shaped.  It
    is always a good idea to check the default behavior by using `bw_adjust` to
    increase or decrease the amount of smoothing.

    Because the smoothing is performed with a Gaussian kernel, the estimated
    density curve can extend to values that do not make sense for a particular
    dataset. For example, the curve may be drawn over negative values when
    smoothing data that are naturally positive. The `cut` parameter can be used
    to control the extent of the curve, but datasets that have many observations
    close to a natural boundary may be better served by a different transform.

    Similar considerations apply when a dataset is naturally discrete or "spiky"
    (containing many repeated observations of the same value). KDEs will always
    produce a smooth curve, which would be misleading in these situations.

    The units on the density axis are a common source of confusion. While kernel
    density estimation produces a probability distribution, the height of the curve
    at each point gives a density, not a probability. A probability can be obtained
    only by integrating the density across a range. The curve is normalized so
    that the integral over all possible values is 1, meaning that the scale of
    the density axis depends on the data values.

    Examples
    --------
    .. include:: ../docstrings/objects.KDE.rst

    """
    bw_adjust: float = 1
    bw_method: str | float | Callable[[gaussian_kde], float] = "scott"
    common_norm: bool | list[str] = True
    common_grid: bool | list[str] = True
    gridsize: int | None = 200
    cut: float = 3
    cumulative: bool = False

    def __post_init__(self):

        if self.cumulative and _no_scipy:
            raise RuntimeError("Cumulative KDE evaluation requires scipy")

    def _check_var_list_or_boolean(self, param: str, grouping_vars: Any) -> None:
        """Do input checks on grouping parameters."""
        value = getattr(self, param)
        if not (
            isinstance(value, bool)
            or (isinstance(value, list) and all(isinstance(v, str) for v in value))
        ):
            param_name = f"{self.__class__.__name__}.{param}"
            raise TypeError(f"{param_name} must be a boolean or list of strings.")
        self._check_grouping_vars(param, grouping_vars)

    def _fit(self, data: DataFrame, orient: str) -> gaussian_kde:
        """Fit and return a KDE object."""
        # TODO need to handle singular data

        fit_kws: dict[str, Any] = {"bw_method": self.bw_method}
        if "weight" in data:
            fit_kws["weights"] = data["weight"]
        kde = gaussian_kde(data[orient], **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _get_support(self, data: DataFrame, orient: str) -> ndarray:
        """Define the grid that the KDE will be evaluated on."""
        if self.gridsize is None:
            return data[orient].to_numpy()

        kde = self._fit(data, orient)
        bw = np.sqrt(kde.covariance.squeeze())
        gridmin = data[orient].min() - bw * self.cut
        gridmax = data[orient].max() + bw * self.cut
        return np.linspace(gridmin, gridmax, self.gridsize)

    def _fit_and_evaluate(
        self, data: DataFrame, orient: str, support: ndarray
    ) -> DataFrame:
        """Transform single group by fitting a KDE and evaluating on a support grid."""
        empty = pd.DataFrame(columns=[orient, "weight", "density"], dtype=float)
        if len(data) < 2:
            return empty
        try:
            kde = self._fit(data, orient)
        except np.linalg.LinAlgError:
            return empty

        if self.cumulative:
            s_0 = support[0]
            density = np.array([kde.integrate_box_1d(s_0, s_i) for s_i in support])
        else:
            density = kde(support)

        weight = data["weight"].sum()
        return pd.DataFrame({orient: support, "weight": weight, "density": density})

    def _transform(
        self, data: DataFrame, orient: str, grouping_vars: list[str]
    ) -> DataFrame:
        """Transform multiple groups by fitting KDEs and evaluating."""
        empty = pd.DataFrame(columns=[*data.columns, "density"], dtype=float)
        if len(data) < 2:
            return empty
        try:
            support = self._get_support(data, orient)
        except np.linalg.LinAlgError:
            return empty

        grouping_vars = [x for x in grouping_vars if data[x].nunique() > 1]
        if not grouping_vars:
            return self._fit_and_evaluate(data, orient, support)
        groupby = GroupBy(grouping_vars)
        return groupby.apply(data, self._fit_and_evaluate, orient, support)

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        if "weight" not in data:
            data = data.assign(weight=1)

        # Transform each group separately
        grouping_vars = [str(v) for v in data if v in groupby.order]
        if not grouping_vars or self.common_grid is True:
            res = self._transform(data, orient, grouping_vars)
        else:
            if self.common_grid is False:
                grid_vars = grouping_vars
            else:
                self._check_var_list_or_boolean("common_grid", grouping_vars)
                grid_vars = [v for v in self.common_grid if v in grouping_vars]

            res = (
                GroupBy(grid_vars)
                .apply(data, self._transform, orient, grouping_vars)
            )

        # Normalize, potentially within groups
        if not grouping_vars or self.common_norm is True:
            res = res.assign(group_weight=data["weight"].sum())
        else:
            if self.common_norm is False:
                norm_vars = grouping_vars
            else:
                self._check_var_list_or_boolean("common_norm", grouping_vars)
                norm_vars = [v for v in self.common_norm if v in grouping_vars]

            res = res.join(
                data.groupby(norm_vars)["weight"].sum().rename("group_weight"),
                on=norm_vars,
            )

        value_var = {"x": "y", "y": "x"}[orient]
        res["density"] *= res.eval("weight / group_weight")
        res[value_var] = res["density"]

        return res
