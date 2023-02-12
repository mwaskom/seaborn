from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, TypedDict

import pandas as pd
from pandas import DataFrame

from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn._statistics import EstimateAggregator
from seaborn._core.typing import Vector


@dataclass
class Agg(Stat):
    """
    Aggregate data along the value axis using given method.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`pandas.Series` method or a vector -> scalar function.

    See Also
    --------
    objects.Est : Aggregation with error bars.

    Examples
    --------
    .. include:: ../docstrings/objects.Agg.rst

    """

    def __init__(
        self,
        func: str | Callable[[Vector], float] = "mean",
        group_by_orient = True,
    ):
        self.func = func
        self.group_by_orient = group_by_orient
    

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        var = {"x": "y", "y": "x"}.get(orient)
        res = (
            groupby
            .agg(data, {var: self.func})
            .dropna(subset=[var])
            .reset_index(drop=True)
        )
        return res


class AggCustom(Stat):
    """
    Aggregate data using provided aggregation function(s).

    """
    # TODO: Maybe enable tweak grouping vars as well.
    def __init__(
        self,
        func: str | Callable[[Vector], float] | TypedDict[str | Callable[[Vector], float]] = "mean",
        group_by_orient=True,
    ):
        self.group_by_orient = group_by_orient
        self.func = func
    
    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        if isinstance(self.func, dict):
            agg_kws = self.func
            if self.group_by_orient and orient in agg_kws:
                raise ValueError("Aggregating 'orient' values which are used for grouping, too.")
        else:
            # Note: alternatively, we could 'groupby._get_groups(data)'
            #       and apply func on remining cols
            agg_kws = {"x": self.func, "y": self.func}
        
        if self.group_by_orient:
            agg_kws.pop(orient, None)

        res = (
            groupby
            .agg(data, agg_kws)
            .dropna(subset=agg_kws.keys())
            .reset_index(drop=True)
        )
        return res


class Agg2d(Stat):
    """
    Aggregate both axes data using given method.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`pandas.Series` method or a vector -> scalar function.
    
    See Also
    --------
    objects.Agg : Aggregation along the value axis.

    Examples
    --------
    TODO
    """
    func: str | Callable[[Vector], float] = "mean"

    group_by_orient: ClassVar[bool] = False

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        res = (
            groupby
            .agg(data, {"x": self.func, "y": self.func})
            .dropna(subset=["x","y"])
            .reset_index(drop=True)
        )
        return res



@dataclass
class Est(Stat):
    """
    Calculate a point estimate and error bar interval.

    For additional information about the various `errorbar` choices, see
    the :doc:`errorbar tutorial </tutorial/error_bars>`.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`numpy.ndarray` method or a vector -> scalar function.
    errorbar : str, (str, float) tuple, or callable
        Name of errorbar method (one of "ci", "pi", "se" or "sd"), or a tuple
        with a method name ane a level parameter, or a function that maps from a
        vector to a (min, max) interval.
    n_boot : int
       Number of bootstrap samples to draw for "ci" errorbars.
    seed : int
        Seed for the PRNG used to draw bootstrap samples.

    Examples
    --------
    .. include:: ../docstrings/objects.Est.rst

    """
    func: str | Callable[[Vector], float] = "mean"
    errorbar: str | tuple[str, float] = ("ci", 95)
    n_boot: int = 1000
    seed: int | None = None

    group_by_orient: ClassVar[bool] = True

    def _process(
        self, data: DataFrame, var: str, estimator: EstimateAggregator
    ) -> DataFrame:
        # Needed because GroupBy.apply assumes func is DataFrame -> DataFrame
        # which we could probably make more general to allow Series return
        res = estimator(data, var)
        return pd.DataFrame([res])

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        boot_kws = {"n_boot": self.n_boot, "seed": self.seed}
        engine = EstimateAggregator(self.func, self.errorbar, **boot_kws)

        var = {"x": "y", "y": "x"}[orient]
        res = (
            groupby
            .apply(data, self._process, var, engine)
            .dropna(subset=[var])
            .reset_index(drop=True)
        )

        res = res.fillna({f"{var}min": res[var], f"{var}max": res[var]})

        return res


@dataclass
class Rolling(Stat):
    ...

    def __call__(self, data, groupby, orient, scales):
        ...
