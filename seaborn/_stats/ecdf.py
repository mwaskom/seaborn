from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from seaborn._statistics import ECDF as _ECDF


@dataclass
class ECDF(Stat):
    """
    Compute a empirical distribution function


    Parameters
    ----------
    stat : {{“proportion”, “percent”, “count”}}
        Distribution statistic to compute.
    complementary : bool
        If True, show the complementary CDF (1 - CDF)

    Examples
    --------
    .. include:: ../docstrings/objects.ECDF.rst
    """
    stat: str = "proportion"
    complementary: bool = False

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale]) -> DataFrame:
        self._check_param_one_of("stat", ("proportion", "percent", "count"))

        if "weight" not in data:
            data = data.assign(weight=1)
        data = data.dropna(subset=[orient, "weight"])
        def ecdf_transform(data):
            estimator = _ECDF(stat=self.stat, complementary=self.complementary)
            y, x = estimator(data[orient],weights=data["weight"])
            return pd.DataFrame({orient: x, "y": y})
        
        return groupby.apply(data, ecdf_transform)

