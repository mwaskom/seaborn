from __future__ import annotations
from .base import Stat


class Mean(Stat):

    # TODO use some special code here to group by the orient variable?
    grouping_vars = ["hue", "size", "style"]

    def __call__(self, data):
        return data.mean()
