from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
    from pandas import DataFrame


class Stat:

    grouping_vars: list[str] = []

    def setup(self, data: DataFrame, orient: Literal["x", "y"]) -> Stat:
        """The default setup operation is to store a reference to the full data."""
        # TODO make this non-mutating
        self._full_data = data
        self.orient = orient
        return self

    def __call__(self, data: DataFrame):
        """Sub-classes must define the call method to implement the transform."""
        raise NotImplementedError
