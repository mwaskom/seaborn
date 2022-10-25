from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableFloat,
    MappableColor,
    MappableStyle,
    resolve_properties,
    resolve_color,
    document_properties,
)


@document_properties
@dataclass
class Rect(Mark):
    """
    """
    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(.7)
    fill: MappableBool = Mappable(True)
    edgecolor: MappableColor = Mappable(depend="color")
    edgealpha: MappableFloat = Mappable(1)
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth")
    edgestyle: MappableStyle = Mappable("-")
    xmin: MappableFloat = Mappable(grouping=False)
    xmax: MappableFloat = Mappable(grouping=False)
    ymin: MappableFloat = Mappable(grouping=False)
    ymax: MappableFloat = Mappable(grouping=False)

    def _plot(self, split_gen, scales, orient):

        patches = defaultdict(list)

        for keys, data, ax in split_gen(allow_empty=True):

            kws = {}
            resolved = resolve_properties(self, keys, scales)
            fc = resolve_color(self, keys, "", scales)
            if not resolved["fill"]:
                fc = mpl.colors.to_rgba(fc, 0)

            kws["facecolor"] = fc
            kws["edgecolor"] = resolve_color(self, keys, "edge", scales)
            kws["linewidth"] = resolved["edgewidth"]
            kws["linestyle"] = resolved["edgestyle"]

            for row in data.to_dict("records"):

                xmin = row.get("xmin", resolved["xmin"])
                xmax = row.get("xmax", resolved["xmax"])
                ymin = row.get("ymin", resolved["ymin"])
                ymax = row.get("ymax", resolved["ymax"])
                verts = [
                    (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                ]
                patches[ax].append(mpl.patches.Polygon(verts, **kws))

            if data.empty:

                verts = [
                    (self.xmin, self.ymin), (self.xmax, self.ymin),
                    (self.xmax, self.ymax), (self.xmin, self.ymax),
                ]
                patches[ax].append(mpl.patches.Polygon(verts, **kws))

        for ax, ax_patches in patches.items():
            for patch in ax_patches:
                ax.add_patch(patch)
