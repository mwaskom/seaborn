from __future__ import annotations
from dataclasses import dataclass

import numpy as np
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
from seaborn.external.version import Version

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

        val_idx = ["y", "x"].index(orient)

        for _, data, ax in split_gen():

            xys = data[["x", "y"]].to_numpy()
            data = self._resolve_properties(data, scales)

            bars, vals = [], []
            for i, (x, y) in enumerate(xys):

                baseline = data["baseline"][i]
                width = data["width"][i]
                xy, w, h = coords_to_geometry(x, y, width, baseline)

                # Skip bars with no value. It's possible we'll want to make this
                # an option (i.e so you have an artist for animating or annotating),
                # but let's keep things simple for now.
                if not np.nan_to_num(h):
                    continue

                # TODO Because we are clipping the artist (see below), the edges end up
                # looking half as wide as they actually are. I don't love this clumsy
                # workaround, which is going to cause surprises if you work with the
                # artists directly. We may need to revisit after feedback.
                linewidth = data["edgewidth"][i] * 2
                linestyle = data["edgestyle"][i]
                if linestyle[1]:
                    linestyle = (linestyle[0], tuple(x / 2 for x in linestyle[1]))

                bar = mpl.patches.Rectangle(
                    xy=xy,
                    width=w,
                    height=h,
                    facecolor=data["facecolor"][i],
                    edgecolor=data["edgecolor"][i],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    **self.artist_kws,
                )

                # This is a bit of a hack to handle the fact that the edge lines are
                # centered on the actual extents of the bar, and overlap when bars are
                # stacked or dodged. We may discover that this causes problems and needs
                # to be revisited at some point. Also it should be faster to clip with
                # a bbox than a path, but I cant't work out how to get the intersection
                # with the axes bbox.
                bar.set_clip_path(bar.get_path(), bar.get_transform() + ax.transData)
                if self.artist_kws.get("clip_on", True):
                    # It seems the above hack undoes the default axes clipping
                    bar.set_clip_box(ax.bbox)
                bar.sticky_edges[val_idx][:] = (0, np.inf)
                ax.add_patch(bar)
                bars.append(bar)
                vals.append(h)

            # Add a container which is useful for, e.g. Axes.bar_label
            if Version(mpl.__version__) >= Version("3.4.0"):
                orientation = {"x": "vertical", "y": "horizontal"}[orient]
                container_kws = dict(datavalues=vals, orientation=orientation)
            else:
                container_kws = {}
            container = mpl.container.BarContainer(bars, **container_kws)
            ax.add_container(container)

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
