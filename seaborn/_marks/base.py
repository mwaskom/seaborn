from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Literal, Any
    from .._stats.base import Stat


class Mark:

    # TODO where to define vars we always group by (col, row, group)
    grouping_vars: list[str]
    default_stat: Optional[Stat] = None  # TODO or identity?
    orient: Literal["x", "y"]

    def __init__(self, **kwargs: Any):

        self._kwargs = kwargs
