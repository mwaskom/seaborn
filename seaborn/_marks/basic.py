from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import matplotlib as mpl

from seaborn._marks.base import Mark, Feature
from seaborn._stats.regression import PolyFit

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Any

    MappableStr = Union[str, Feature]
    MappableFloat = Union[float, Feature]
    MappableColor = Union[str, tuple, Feature]

    StatParam = Union[Any, Feature]


@dataclass
class Line(Mark):

    # TODO other semantics (marker?)

    color: MappableColor = Feature("C0", groups=True)
    alpha: MappableFloat = Feature(1, groups=True)
    linewidth: MappableFloat = Feature(rc="lines.linewidth", groups=True)
    linestyle: MappableStr = Feature(rc="lines.linestyle", groups=True)

    sort: bool = True

    def plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen():

            keys = self.resolve_features(keys, scales)

            if self.sort:
                # TODO where to dropna?
                data = data.dropna().sort_values(orient)

            line = mpl.lines.Line2D(
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                color=keys["color"],
                alpha=keys["alpha"],
                linewidth=keys["linewidth"],
                linestyle=keys["linestyle"],
                **self.artist_kws,  # TODO keep? remove? be consistent across marks
            )
            ax.add_line(line)

    def _legend_artist(self, variables, value, scales):

        key = self.resolve_features({v: value for v in variables}, scales)

        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            alpha=key["alpha"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
        )


@dataclass
class Area(Mark):

    color: MappableColor = Feature("C0", groups=True)
    alpha: MappableFloat = Feature(1, groups=True)

    def plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen():

            kws = self.artist_kws.copy()

            keys = self.resolve_features(keys, scales)
            kws["facecolor"] = self._resolve_color(keys, scales=scales)
            kws["edgecolor"] = self._resolve_color(keys, scales=scales)

            # TODO how will orient work here?
            # Currently this requires you to specify both orient and use y, xmin, xmin
            # to get a fill along the x axis. Seems like we should need only one?
            # Alternatively, should we just make the PolyCollection manually?
            if orient == "x":
                ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
            else:
                ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)


@dataclass
class PolyLine(Line):

    order: "StatParam" = Feature(stat="order")  # TODO the annotation

    default_stat: ClassVar = PolyFit  # TODO why is this showing up as a field?
