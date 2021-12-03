from __future__ import annotations
import numpy as np
import matplotlib as mpl

from seaborn._marks.base import Mark, Feature


class Point(Mark):  # TODO types

    supports = ["color"]

    def __init__(
        self,
        *,
        color=Feature("C0"),
        alpha=Feature(1),  # TODO auto alpha?
        fill=Feature(True),
        fillcolor=Feature(depend="color"),
        fillalpha=Feature(.2),
        marker=Feature(rc="scatter.marker"),
        pointsize=Feature(5),  # TODO rcParam?
        linewidth=Feature(.75),  # TODO rcParam?
        jitter=None,  # TODO Does Feature always mean mappable?
        **kwargs,  # TODO needed?
    ):

        super().__init__(**kwargs)

        # TODO should this use SEMANTICS as the source of possible features?
        self.features = dict(
            color=color,
            alpha=alpha,
            fill=fill,
            fillcolor=fillcolor,
            fillalpha=fillalpha,
            marker=marker,
            pointsize=pointsize,
            linewidth=linewidth,
        )

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

    def _plot_split(self, keys, data, ax, kws):

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        # (That should be solved upstream by defaulting to "" for unset x/y?)
        # (Be mindful of xmin/xmax, etc!)

        kws = kws.copy()

        markers = self._resolve(data, "marker")
        fill = self._resolve(data, "fill")
        fill & np.array([m.is_filled() for m in markers])

        edgecolors = self._resolve_color(data)
        facecolors = self._resolve_color(data, "fill")
        facecolors[~fill, 3] = 0

        linewidths = self._resolve(data, "linewidth")
        pointsize = self._resolve(data, "pointsize")

        paths = []
        path_cache = {}
        for m in markers:
            if m not in path_cache:
                path_cache[m] = m.get_path().transformed(m.get_transform())
            paths.append(path_cache[m])

        sizes = pointsize ** 2
        offsets = data[["x", "y"]].to_numpy()

        points = mpl.collections.PathCollection(
            paths=paths,
            sizes=sizes,
            offsets=offsets,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            transOffset=ax.transData,
            transform=mpl.transforms.IdentityTransform(),
        )
        ax.add_collection(points)


class Line(Mark):

    grouping_vars = ["color", "marker", "linestyle", "linewidth"]
    supports = ["color", "marker", "linestyle", "linewidth"]

    def __init__(
        self,
        *,
        color=Feature("C0"),
        alpha=Feature(1),
        linestyle=Feature(rc="lines.linestyle"),
        linewidth=Feature(rc="lines.linewidth"),
        marker=Feature(rc="lines.marker"),
        # ... other features
        sort=True,
        **kwargs,  # TODO needed? Probably, but rather have artist_kws dict?
    ):

        super().__init__(**kwargs)

        # TODO should this use SEMANTICS as the source of possible features?
        self.features = dict(
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
        )

        self.sort = sort

    def _plot_split(self, keys, data, ax, kws):

        if self.sort:
            data = data.sort_values(self.orient)

        line = mpl.lines.Line2D(
            data["x"].to_numpy(),
            data["y"].to_numpy(),
            color=self._resolve_color(keys),
            linewidth=self._resolve(keys, "linewidth"),
            linestyle=self._resolve(keys, "linestyle"),
            marker=self._resolve(keys, "marker"),
            **kws
        )
        ax.add_line(line)


class Area(Mark):

    grouping_vars = ["color"]
    supports = ["color"]

    def _plot_split(self, keys, data, ax, kws):

        if "color" in keys:
            # TODO as we need the kwarg to be facecolor, that should be the mappable?
            kws["facecolor"] = self.mappings["color"](keys["color"])

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if self.orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
