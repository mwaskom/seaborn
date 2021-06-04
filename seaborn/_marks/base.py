from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Literal, Any
    from collections.abc import Generator
    from pandas import DataFrame
    from matplotlib.axes import Axes
    from .._core.mappings import SemanticMapping
    from .._stats.base import Stat

    MappingDict = dict[str, SemanticMapping]


class Mark:
    """Base class for objects that control the actual plotting."""
    # TODO where to define vars we always group by (col, row, group)
    default_stat: Optional[Stat] = None
    grouping_vars: list[str]
    orient: Literal["x", "y"]
    requires: list[str]  # List of variabes that must be defined
    supports: list[str]  # List of variables that will be used

    def __init__(self, **kwargs: Any):

        self._kwargs = kwargs

    def _plot(
        self, generate_splits: Generator, mappings: MappingDict,
    ) -> None:
        """Main interface for creating a plot."""
        for keys, data, ax in generate_splits():
            kws = self._kwargs.copy()
            self._plot_split(keys, data, ax, mappings, kws)

        self._finish_plot()

    def _plot_split(
        self,
        keys: dict[str: Any],
        data: DataFrame,
        ax: Axes,
        mappings: MappingDict,
        kws: dict,
    ) -> None:
        """Method that plots specific subsets of data. Must be defined by subclass."""
        raise NotImplementedError()

    def _finish_plot(self) -> None:
        """Method that is called after each data subset has been plotted."""
        pass
