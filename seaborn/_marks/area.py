from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgba

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableFloat,
    MappableColor,
    MappableStyle,
)


class AreaBase:

    def plot(self, split_gen, scales, orient):

        kws = {}

        for keys, data, ax in split_gen():

            data = self._standardize_coordinate_parameters(data, orient)
            keys = self.resolve_properties(keys, scales)
            verts = self._get_verts(data, orient)

            ax.update_datalim(verts)

            kws.setdefault(ax, defaultdict(list))
            kws[ax]["verts"].append(verts)

            alpha = keys["alpha"] if keys["fill"] else 0
            kws[ax]["facecolors"].append(to_rgba(keys["color"], alpha))
            kws[ax]["edgecolors"].append(to_rgba(keys["edgecolor"], keys["edgealpha"]))

            kws[ax]["linewidth"].append(keys["edgewidth"])
            kws[ax]["linestyle"].append(keys["edgestyle"])

        for ax, ax_kws in kws.items():
            ax.add_collection(mpl.collections.PolyCollection(**ax_kws))

    def _standardize_coordinate_parameters(self, data, orient):
        return data

    def _get_verts(self, data, orient):

        dv = {"x": "y", "y": "x"}[orient]
        data = data.sort_values(orient)
        verts = np.concatenate([
            data[[orient, f"{dv}min"]].to_numpy(),
            data[[orient, f"{dv}max"]].to_numpy()[::-1],
        ])
        if orient == "y":
            verts = verts[:, ::-1]
        return verts

    def _legend_artist(self, variables, value, scales):

        key = self.resolve_properties({v: value for v in variables}, scales)

        return mpl.patches.Patch(
            facecolor=to_rgba(key["color"], key["alpha"] if key["fill"] else 0),
            edgecolor=to_rgba(key["edgecolor"], key["edgealpha"]),
            linewidth=key["edgewidth"],
            linestyle=key["edgestyle"],
            **self.artist_kws,
        )


@dataclass
class Area(AreaBase, Mark):
    """
    An interval mark that fills between baseline and data values.
    """
    color: MappableColor = Mappable("C0", groups=True)
    alpha: MappableFloat = Mappable(.2, groups=True)
    fill: MappableBool = Mappable(True, groups=True)
    edgecolor: MappableColor = Mappable(depend="color", groups=True)
    edgealpha: MappableFloat = Mappable(1, groups=True)
    edgewidth: MappableFloat = Mappable(rc="patch.linewidth", groups=True)
    edgestyle: MappableStyle = Mappable("-", groups=True)

    # TODO should this be settable / mappable?
    baseline: MappableFloat = Mappable(0)

    def _standardize_coordinate_parameters(self, data, orient):
        dv = {"x": "y", "y": "x"}[orient]
        return data.rename(columns={"baseline": f"{dv}min", dv: f"{dv}max"})


@dataclass
class Ribbon(AreaBase, Mark):
    """
    An interval mark that fills between minimum and maximum values.
    """
    color: MappableColor = Mappable("C0", groups=True)
    alpha: MappableFloat = Mappable(.2, groups=True)
    fill: MappableBool = Mappable(True, groups=True)
    edgecolor: MappableColor = Mappable(depend="color", groups=True)
    edgealpha: MappableFloat = Mappable(1, groups=True)
    edgewidth: MappableFloat = Mappable(0, groups=True)
    edgestyle: MappableFloat = Mappable("-", groups=True)

    def _standardize_coordinate_parameters(self, data, orient):
        # dv = {"x": "y", "y": "x"}[orient]
        # TODO assert that all(ymax >= ymin)?
        return data
