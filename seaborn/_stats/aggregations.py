from __future__ import annotations
from .base import Stat


class Mean(Stat):

    grouping_vars = ["hue", "size", "style"]

    def __call__(self, data):
        return data.mean()
