from __future__ import annotations
from .base import Mark


class Point(Mark):

    grouping_vars = []
    requires = []
    supports = ["hue"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        # TODO since names match, can probably be automated!
        # TODO note that newer style is to modify the artists
        if "hue" in data:
            c = mappings["hue"](data["hue"])
        else:
            # TODO prevents passing in c. But do we want to permit that?
            # I think if we implement map_hue("identity"), then no
            c = None

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        ax.scatter(x=data["x"], y=data["y"], c=c, **kws)


class Line(Mark):

    # TODO how to handle distinction between stat groupers and plot groupers?
    # i.e. Line needs to aggregate by x, but not plot by it
    # also how will this get parametrized to support orient=?
    # TODO will this sort by the orient dimension like lineplot currently does?
    grouping_vars = ["hue", "size", "style"]
    requires = []
    supports = ["hue"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "hue" in keys:
            kws["color"] = mappings["hue"](keys["hue"])

        ax.plot(data["x"], data["y"], **kws)


class Area(Mark):

    grouping_vars = ["hue"]
    requires = []
    supports = ["hue"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "hue" in keys:
            kws["facecolor"] = mappings["hue"](keys["hue"])

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if self.orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
