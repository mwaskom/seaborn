from __future__ import annotations
import numpy as np
from .base import Mark


class Point(Mark):

    grouping_vars = []
    requires = []
    supports = ["hue"]

    def __init__(self, jitter=None, **kwargs):

        super().__init__(**kwargs)
        self.jitter = jitter  # TODO decide on form of jitter and add type hinting

    def _adjust(self, df):

        if self.jitter is None:
            return df

        x, y = self.jitter  # TODO maybe not format, and do better error handling

        # TODO maybe accept a Jitter class so we can control things like distribution?
        # If we do that, should we allow convenient flexibility (i.e. (x, y) tuple)
        # in the object interface, or be simpler but more verbose?

        # TODO note that some marks will have multiple adjustments
        # (e.g. strip plot has both dodging and jittering)

        # TODO native scale of jitter? maybe just for a Strip subclass?

        rng = np.random.default_rng()  # TODO seed?

        n = len(df)
        x_jitter = 0 if not x else rng.uniform(-x, +x, n)
        y_jitter = 0 if not y else rng.uniform(-y, +y, n)

        # TODO: this fails if x or y are paired. Apply to all columns that start with y?
        return df.assign(x=df["x"] + x_jitter, y=df["y"] + y_jitter)

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
