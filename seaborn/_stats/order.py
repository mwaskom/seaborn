
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Literal, cast

import numpy as np
from pandas import DataFrame

from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat


# From https://github.com/numpy/numpy/blob/main/numpy/lib/function_base.pyi
_MethodKind = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


@dataclass
class Perc(Stat):
    """
    Replace observations along the value axis with percentile statistics.

    Parameters
    ----------
    k : list of numbers or int
        If a list of numbers, these give the percentiles (in [0, 100]) to compute.
        If an integer, compute that many evenly-spaced percentiles between 0 and 100.
        For example, `k=5` computes the 0th, 25th, 50th, 75th, and 100th percentiles.
    method : str
        Interpolation method for estimating percentiles between observed datapoints.
        See :func:`numpy.percentile` for valid options and more details.

    Examples
    --------
    .. include:: ../docstrings/objects.Perc.rst

    """
    k: int | list[float] = 5
    method: str = "linear"

    group_by_orient: ClassVar[bool] = True

    def _percentile(self, data: DataFrame, var: str) -> DataFrame:

        k = list(np.linspace(0, 100, self.k)) if isinstance(self.k, int) else self.k
        method = cast(_MethodKind, self.method)
        return DataFrame(
            {var: np.percentile(data[var], k, method=method), "percentile": k}
        )

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        var = {"x": "y", "y": "x"}[orient]
        res = (
            groupby
            .apply(data, self._percentile, var)
            .dropna()
            .reset_index(drop=True)
        )
        return res
