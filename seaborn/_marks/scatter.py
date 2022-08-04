from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableBool,
    MappableFloat,
    MappableString,
    MappableColor,
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
class Scatter(Mark):
    """
    A point mark defined by strokes with optional fills.
    """
    # TODO retype marker as MappableMarker
    marker: MappableString = Mappable(rc="scatter.marker", grouping=False)
    stroke: MappableFloat = Mappable(.75, grouping=False)  # TODO rcParam?
    pointsize: MappableFloat = Mappable(3, grouping=False)  # TODO rcParam?
    color: MappableColor = Mappable("C0", grouping=False)
    alpha: MappableFloat = Mappable(1, grouping=False)  # TODO auto alpha?
    fill: MappableBool = Mappable(True, grouping=False)
    fillcolor: MappableColor = Mappable(depend="color", grouping=False)
    fillalpha: MappableFloat = Mappable(.2, grouping=False)

    def _resolve_paths(self, data):

        paths = []
        path_cache = {}
        marker = data["marker"]

        def get_transformed_path(m):
            return m.get_path().transformed(m.get_transform())

        if isinstance(marker, mpl.markers.MarkerStyle):
            return get_transformed_path(marker)

        for m in marker:
            if m not in path_cache:
                path_cache[m] = get_transformed_path(m)
            paths.append(path_cache[m])
        return paths

    def _resolve_properties(self, data, scales):

        resolved = resolve_properties(self, data, scales)
        resolved["path"] = self._resolve_paths(resolved)

        if isinstance(data, dict):  # TODO need a better way to check
            filled_marker = resolved["marker"].is_filled()
        else:
            filled_marker = [m.is_filled() for m in resolved["marker"]]

        resolved["linewidth"] = resolved["stroke"]
        resolved["fill"] = resolved["fill"] * filled_marker
        resolved["size"] = resolved["pointsize"] ** 2

        resolved["edgecolor"] = resolve_color(self, data, "", scales)
        resolved["facecolor"] = resolve_color(self, data, "fill", scales)

        # Because only Dot, and not Scatter, has an edgestyle
        resolved.setdefault("edgestyle", (0, None))

        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved

    def _plot(self, split_gen, scales, orient):

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        # (That should be solved upstream by defaulting to "" for unset x/y?)
        # (Be mindful of xmin/xmax, etc!)

        for _, data, ax in split_gen():

            offsets = np.column_stack([data["x"], data["y"]])
            data = self._resolve_properties(data, scales)

            points = mpl.collections.PathCollection(
                offsets=offsets,
                paths=data["path"],
                sizes=data["size"],
                facecolors=data["facecolor"],
                edgecolors=data["edgecolor"],
                linewidths=data["linewidth"],
                linestyles=data["edgestyle"],
                transOffset=ax.transData,
                transform=mpl.transforms.IdentityTransform(),
                **self.artist_kws,
            )
            ax.add_collection(points)

    def _legend_artist(
        self, variables: list[str], value: Any, scales: dict[str, Scale],
    ) -> Artist:

        key = {v: value for v in variables}
        res = self._resolve_properties(key, scales)

        return mpl.collections.PathCollection(
            paths=[res["path"]],
            sizes=[res["size"]],
            facecolors=[res["facecolor"]],
            edgecolors=[res["edgecolor"]],
            linewidths=[res["linewidth"]],
            linestyles=[res["edgestyle"]],
            transform=mpl.transforms.IdentityTransform(),
            **self.artist_kws,
        )


# TODO change this to depend on ScatterBase?
@dataclass
class Dot(Scatter):
    """
    A point mark defined by shape with optional edges.
    """
    marker: MappableString = Mappable("o", grouping=False)
    color: MappableColor = Mappable("C0", grouping=False)
    alpha: MappableFloat = Mappable(1, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(depend="color", grouping=False)
    edgealpha: MappableFloat = Mappable(depend="alpha", grouping=False)
    pointsize: MappableFloat = Mappable(6, grouping=False)  # TODO rcParam?
    edgewidth: MappableFloat = Mappable(.5, grouping=False)  # TODO rcParam?
    edgestyle: MappableStyle = Mappable("-", grouping=False)

    def _resolve_properties(self, data, scales):
        # TODO this is maybe a little hacky, is there a better abstraction?
        resolved = super()._resolve_properties(data, scales)

        filled = resolved["fill"]

        main_stroke = resolved["stroke"]
        edge_stroke = resolved["edgewidth"]
        resolved["linewidth"] = np.where(filled, edge_stroke, main_stroke)

        # Overwrite the colors that the super class set
        main_color = resolve_color(self, data, "", scales)
        edge_color = resolve_color(self, data, "edge", scales)

        if not np.isscalar(filled):
            # Expand dims to use in np.where with rgba arrays
            filled = filled[:, None]
        resolved["edgecolor"] = np.where(filled, edge_color, main_color)

        filled = np.squeeze(filled)
        if isinstance(main_color, tuple):
            main_color = tuple([*main_color[:3], main_color[3] * filled])
        else:
            main_color = np.c_[main_color[:, :3], main_color[:, 3] * filled]
        resolved["facecolor"] = main_color

        return resolved
