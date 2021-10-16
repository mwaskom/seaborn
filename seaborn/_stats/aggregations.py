from __future__ import annotations
from .base import Stat


class Mean(Stat):

    # TODO use some special code here to group by the orient variable?
    # TODO get automatically
    grouping_vars = ["color", "edgecolor", "marker", "linestyle", "linewidth"]

    def __call__(self, data):
        return data.filter(regex="x|y").mean()
