from __future__ import annotations
from .base import Stat
from seaborn._core.plot import SEMANTICS


class Mean(Stat):

    # TODO use some special code here to group by the orient variable?
    # TODO get automatically
    grouping_vars = [v for v in SEMANTICS if v != "width"]  # TODO fix

    def __call__(self, data):
        return data.filter(regex="x|y").mean()
