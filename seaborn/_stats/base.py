from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal


class Stat:

    orient: Literal["x", "y"]
    grouping_vars: list[str]
