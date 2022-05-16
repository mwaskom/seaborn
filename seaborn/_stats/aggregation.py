from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from seaborn._stats.base import Stat

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from numbers import Number
    from seaborn._core.typing import Vector


@dataclass
class Agg(Stat):
    """
    Aggregate data along the value axis using given method.

    Parameters
    ----------
    func
        Name of a method understood by Pandas or an arbitrary vector -> scalar function.

    """
    # TODO In current practice we will always have a numeric x/y variable,
    # but they may represent non-numeric values. Needs clear documentation.
    func: str | Callable[[Vector], Number] = "mean"

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

    # TODO a string here must be a numpy ufunc?
    func: str | Callable[[Vector], Number] = "mean"

    # TODO type errorbar options with literal?
    errorbar: str | tuple[str, float] = ("ci", 95)

    group_by_orient: ClassVar[bool] = True

    def __call__(self, data, groupby, orient, scales):

        # TODO port code over from _statistics
        ...


@dataclass
class Rolling(Stat):
    ...

    def __call__(self, data, groupby, orient, scales):
        ...
