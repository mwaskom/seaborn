"""Base module for statistical transformations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn._core.groupby import GroupBy
    from seaborn._core.scales import Scale


@dataclass
class Stat:
    """Base class for objects that apply statistical transformations."""

    # The class supports a partial-function application pattern. The object is
    # initialized with desired parameters and the result is a callable that
    # accepts and returns dataframes.

    # The statistical transformation logic should not add any state to the instance
    # beyond what is defined with the initialization parameters.

    # Subclasses can declare whether the orient dimension should be used in grouping
    # TODO consider whether this should be a parameter. Motivating example:
    # use the same KDE class violin plots and univariate density estimation.
    # In the former case, we would expect separate densities for each unique
    # value on the orient axis, but we would not in the latter case.
    group_by_orient: ClassVar[bool] = False

    def __call__(
        self,
        data: DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, Scale],
    ) -> DataFrame:
        """Apply statistical transform to data subgroups and return combined result."""
        return data
