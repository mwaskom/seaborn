from __future__ import annotations
from .base import Mark


class Point(Mark):

    grouping_vars = []

    def _plot(self, splitgen, mappings):

        for keys, data, ax in splitgen():

            kws = self._kwargs.copy()

            # TODO since names match, can probably be automated!
            if "hue" in data:
                c = mappings["hue"](data["hue"])
            else:
                c = None

            # TODO Not backcompat with allowed (but nonfunctional) univariate plots
            ax.scatter(x=data["x"], y=data["y"], c=c, **kws)


class Line(Mark):

    # TODO how to handle distinction between stat groupers and plot groupers?
    # i.e. Line needs to aggregate by x, but not plot by it
    # also how will this get parametrized to support orient=?
    grouping_vars = ["hue", "size", "style"]

    def _plot(self, splitgen, mappings):

        for keys, data, ax in splitgen():

            kws = self._kwargs.copy()

            # TODO pack sem_kws or similar
            if "hue" in keys:
                kws["color"] = mappings["hue"](keys["hue"])

            ax.plot(data["x"], data["y"], **kws)


class Ribbon(Mark):

    grouping_vars = ["hue"]

    def _plot(self, splitgen, mappings):

        # TODO how will orient work here?

        for keys, data, ax in splitgen():

            kws = self._kwargs.copy()

            if "hue" in keys:
                kws["facecolor"] = mappings["hue"](keys["hue"])

            kws.setdefault("alpha", .2)  # TODO are we assuming this is for errorbars?
            kws.setdefault("linewidth", 0)

            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
