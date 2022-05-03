from __future__ import annotations
from dataclasses import dataclass

import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableColor,
    MappableFloat,
    MappableStyle,
    resolve_properties,
    resolve_color,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from matplotlib.artist import Artist
    from seaborn._core.scales import Scale


@dataclass
class Bar(Mark):
    """
    An interval mark drawn between baseline and data values with a width.
    """
    color: MappableColor = Mappable("C0", )
    alpha: MappableFloat = Mappable(.7, )
    fill: MappableBool = Mappable(True, )
    edgecolor: MappableColor = Mappable(depend="color", )
    edgealpha: MappableFloat = Mappable(1, )
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth")
    edgestyle: MappableStyle = Mappable("-", )
    # pattern: MappableString = Mappable(None, )  # TODO no Property yet

    width: MappableFloat = Mappable(.8, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)  # TODO *is* this mappable?

    def _resolve_properties(self, data, scales):

        resolved = resolve_properties(self, data, scales)

        resolved["facecolor"] = resolve_color(self, data, "", scales)
        resolved["edgecolor"] = resolve_color(self, data, "edge", scales)

        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved

    def _plot(self, split_gen, scales, orient):

        def coords_to_geometry(x, y, w, b):
            # TODO possible too slow with lots of bars (e.g. dense hist)
            # Why not just use BarCollection?
            if orient == "x":
                w, h = w, y - b
                xy = x - w / 2, b
            else:
                w, h = x - b, w
                xy = b, y - h / 2
            return xy, w, h

        for _, data, ax in split_gen():

            xys = data[["x", "y"]].to_numpy()
            data = self._resolve_properties(data, scales)

            bars = []
            for i, (x, y) in enumerate(xys):

                baseline = data["baseline"][i]
                width = data["width"][i]
                xy, w, h = coords_to_geometry(x, y, width, baseline)

                bar = mpl.patches.Rectangle(
                    xy=xy,
                    width=w,
                    height=h,
                    facecolor=data["facecolor"][i],
                    edgecolor=data["edgecolor"][i],
                    linewidth=data["edgewidth"][i],
                    linestyle=data["edgestyle"][i],
                )
                ax.add_patch(bar)
                bars.append(bar)

            # TODO add container object to ax, line ax.bar does

    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    ) -> Artist:
        # TODO return some sensible default?
        key = {v: value for v in variables}
        key = self._resolve_properties(key, scales)
        artist = mpl.patches.Patch(
            facecolor=key["facecolor"],
            edgecolor=key["edgecolor"],
            linewidth=key["edgewidth"],
            linestyle=key["edgestyle"],
        )
        return artist
