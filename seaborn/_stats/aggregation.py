from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable
from numbers import Number

import numpy as np
import pandas as pd

from seaborn._stats.base import Stat
from seaborn.algorithms import bootstrap
from seaborn.utils import _check_argument

from seaborn._core.typing import Vector


@dataclass
class Agg(Stat):
    """
    Aggregate data along the value axis using given method.

    Parameters
    ----------
    func
        Name of a :class:`pandas.Series` method or a vector -> scalar function.

    """
    func: str | Callable[[Vector], float] = "mean"

    group_by_orient: ClassVar[bool] = True

    def __call__(self, data, groupby, orient, scales):

        var = {"x": "y", "y": "x"}.get(orient)
        res = (
            groupby
            .agg(data, {var: self.func})
            # TODO Could be an option not to drop NA?
            .dropna()
            .reset_index(drop=True)
        )
        return res


@dataclass
class Est(Stat):

    func: str | Callable[[Vector], float] = "mean"
    errorbar: str | tuple[str, float] = ("ci", 95)
    n_boot: int = 1000
    seed: int | None = None

    group_by_orient: ClassVar[bool] = True

    def _process(self, data, var):

        vals = data[var]

        estimate = vals.agg(self.func)

        # Options that produce no error bars
        if self.error_method is None:
            err_min = err_max = np.nan
        elif len(data) <= 1:
            err_min = err_max = np.nan

        # Generic errorbars from user-supplied function
        elif callable(self.error_method):
            err_min, err_max = self.error_method(vals)

        # Parametric options
        elif self.error_method == "sd":
            half_interval = vals.std() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval
        elif self.error_method == "se":
            half_interval = vals.sem() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval

        # Nonparametric options
        elif self.error_method == "pi":
            err_min, err_max = _percentile_interval(vals, self.error_level)
        elif self.error_method == "ci":
            boot_kws = {"n_boot": self.n_boot, "seed": self.seed}
            # units = data.get("units", None)  # TODO change to unit
            units = None
            boots = bootstrap(vals, units=units, func=self.func, **boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        res = {var: estimate, f"{var}min": err_min, f"{var}max": err_max}
        return pd.DataFrame([res])

    def __call__(self, data, groupby, orient, scales):

        method, level = _validate_errorbar_arg(self.errorbar)
        self.error_method = method
        self.error_level = level

        var = {"x": "y", "y": "x"}.get(orient)
        res = (
            groupby
            .apply(data, self._process, var)
            .dropna()
            .reset_index(drop=True)
        )
        return res


@dataclass
class Rolling(Stat):
    ...

    def __call__(self, data, groupby, orient, scales):
        ...


def _percentile_interval(data, width):
    """Return a percentile interval from data of a given width."""
    edge = (100 - width) / 2
    percentiles = edge, 100 - edge
    return np.nanpercentile(data, percentiles)


def _validate_errorbar_arg(arg):
    """Check type and value of errorbar argument and assign default level."""
    DEFAULT_LEVELS = {
        "ci": 95,
        "pi": 95,
        "se": 1,
        "sd": 1,
    }

    usage = "`errorbar` must be a callable, string, or (string, number) tuple"

    if arg is None:
        return None, None
    elif callable(arg):
        return arg, None
    elif isinstance(arg, str):
        method = arg
        level = DEFAULT_LEVELS.get(method, None)
    else:
        try:
            method, level = arg
        except (ValueError, TypeError) as err:
            raise err.__class__(usage) from err

    _check_argument("errorbar", list(DEFAULT_LEVELS), method)
    if level is not None and not isinstance(level, Number):
        raise TypeError(usage)

    return method, level
