from __future__ import annotations
import numpy as np
from seaborn._compat import MarkerStyle
from seaborn._marks.base import Mark


class Point(Mark):

    supports = ["color"]

    def __init__(self, marker="o", fill=True, jitter=None, **kwargs):

        # TODO need general policy on mappable defaults
        # I think a good idea would be to use some kind of singleton, so it's
        # clear what mappable attributes can be directly set, but so that
        # we can also read from rcParams at plot time.
        # Will need to decide which of mapping / fixing supercedes if both set,
        # or if that should raise an error.
        kwargs.update(
            marker=marker,
            fill=fill,
        )
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

        # TODO can we simplify this by modifying data with mappings before sending in?
        # Likewise, will we need to know `keys` here? Elsewhere we do `if key in keys`,
        # but I think we can (or can make it so we can) just do `if key in data`.

        # Then the signature could be _plot_split(ax, data, kws):  ... much simpler!

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots

        kws = kws.copy()

        # TODO need better solution here
        default_marker = kws.pop("marker")
        default_fill = kws.pop("fill")

        points = ax.scatter(x=data["x"], y=data["y"], **kws)

        if "color" in data:
            points.set_facecolors(mappings["color"](data["color"]))

        if "edgecolor" in data:
            points.set_edgecolors(mappings["edgecolor"](data["edgecolor"]))

        # TODO facecolor?

        n = data.shape[0]

        # TODO this doesn't work. Apparently scatter is reading
        # the marker.is_filled attribute and directing colors towards
        # the edge/face and then setting the face to uncolored as needed.
        # We are getting to the point where just creating the PathCollection
        # ourselves is probably easier, but not breaking existing scatterplot
        # calls that leverage ax.scatter features like cmap might be tricky.
        # Another option could be to have some internal-only Marks that support
        # the existing functional interface where doing so through the new
        # interface would be overly cumbersome.
        # Either way, it would be best to have a common function like
        # apply_fill(facecolor, edgecolor, filled)
        # We may want to think about how to work with MarkerStyle objects
        # in the absence of a `fill` semantic so that we can relax the
        # constraint on mixing filled and unfilled markers...

        if "marker" in data:
            markers = mappings["marker"](data["marker"])
        else:
            m = MarkerStyle(default_marker)
            markers = (m for _ in range(n))

        if "fill" in data:
            fills = mappings["fill"](data["fill"])
        else:
            fills = (default_fill for _ in range(n))

        paths = []
        for marker, filled in zip(markers, fills):
            fillstyle = "full" if filled else "none"
            m = MarkerStyle(marker, fillstyle)
            paths.append(m.get_path().transformed(m.get_transform()))
        points.set_paths(paths)


class Line(Mark):

    # TODO how to handle distinction between stat groupers and plot groupers?
    # i.e. Line needs to aggregate by x, but not plot by it
    # also how will this get parametrized to support orient=?
    # TODO will this sort by the orient dimension like lineplot currently does?
    grouping_vars = ["color", "marker", "linestyle", "linewidth"]
    supports = ["color", "marker", "linestyle", "linewidth"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "color" in keys:
            kws["color"] = mappings["color"](keys["color"])
        if "linestyle" in keys:
            kws["linestyle"] = mappings["linestyle"](keys["linestyle"])
        if "linewidth" in keys:
            kws["linewidth"] = mappings["linewidth"](keys["linewidth"])

        ax.plot(data["x"], data["y"], **kws)


class Area(Mark):

    grouping_vars = ["color"]
    supports = ["color"]

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "color" in keys:
            # TODO as we need the kwarg to be facecolor, that should be the mappable?
            kws["facecolor"] = mappings["color"](keys["color"])

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if self.orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
