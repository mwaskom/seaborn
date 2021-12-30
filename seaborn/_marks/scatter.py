from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import Mark, Feature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Union
    from matplotlib.artist import Artist

    MappableBool = Union[bool, Feature]
    MappableFloat = Union[float, Feature]
    MappableString = Union[str, Feature]
    MappableColor = Union[str, tuple, Feature]  # TODO


@dataclass
class Scatter(Mark):

    color: MappableColor = Feature("C0")
    alpha: MappableFloat = Feature(1)  # TODO auto alpha?
    fill: MappableBool = Feature(True)
    fillcolor: MappableColor = Feature(depend="color")
    fillalpha: MappableFloat = Feature(.2)
    marker: MappableString = Feature(rc="scatter.marker")
    pointsize: MappableFloat = Feature(3)  # TODO rcParam?
    linewidth: MappableFloat = Feature(.75)  # TODO rcParam?

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

    def resolve_features(self, data):

        resolved = super().resolve_features(data)
        resolved["path"] = self._resolve_paths(resolved)

        if isinstance(data, dict):  # TODO need a better way to check
            filled_marker = resolved["marker"].is_filled()
        else:
            filled_marker = [m.is_filled() for m in resolved["marker"]]

        resolved["fill"] = resolved["fill"] & filled_marker
        resolved["size"] = resolved["pointsize"] ** 2

        resolved["edgecolor"] = self._resolve_color(data)
        resolved["facecolor"] = self._resolve_color(data, "fill")

        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved

    def _plot_split(self, keys, data, ax, kws):

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots
        # (That should be solved upstream by defaulting to "" for unset x/y?)
        # (Be mindful of xmin/xmax, etc!)

        kws = kws.copy()

        offsets = np.column_stack([data["x"], data["y"]])

        # Maybe this can be out in plot()? How do we get coordinates?
        data = self.resolve_features(data)

        points = mpl.collections.PathCollection(
            offsets=offsets,
            paths=data["path"],
            sizes=data["size"],
            facecolors=data["facecolor"],
            edgecolors=data["edgecolor"],
            linewidths=data["linewidth"],
            transOffset=ax.transData,
            transform=mpl.transforms.IdentityTransform(),
        )
        ax.add_collection(points)

    def _legend_artist(self, variables: list[str], value: Any) -> Artist:

        key = {v: value for v in variables}
        key = self.resolve_features(key)

        return mpl.collections.PathCollection(
            paths=[key["path"]],
            sizes=[key["size"]],
            facecolors=[key["facecolor"]],
            edgecolors=[key["edgecolor"]],
            linewidths=[key["linewidth"]],
            transform=mpl.transforms.IdentityTransform(),
        )


@dataclass
class Dot(Scatter):  # TODO depend on ScatterBase or similar?

    color: MappableColor = Feature("C0")
    alpha: MappableFloat = Feature(1)
    edgecolor: MappableColor = Feature(depend="color")
    edgealpha: MappableFloat = Feature(depend="alpha")
    fill: MappableBool = Feature(True)
    marker: MappableString = Feature("o")
    pointsize: MappableFloat = Feature(6)  # TODO rcParam?
    linewidth: MappableFloat = Feature(.5)  # TODO rcParam?

    def resolve_features(self, data):
        # TODO this is maybe a little hacky, is there a better abstraction?
        resolved = super().resolve_features(data)
        resolved["edgecolor"] = self._resolve_color(data, "edge")
        resolved["facecolor"] = self._resolve_color(data)

        # TODO Could move this into a method but solving it at the root feels ideal
        fc = resolved["facecolor"]
        if isinstance(fc, tuple):
            resolved["facecolor"] = fc[0], fc[1], fc[2], fc[3] * resolved["fill"]
        else:
            fc[:, 3] = fc[:, 3] * resolved["fill"]  # TODO Is inplace mod a problem?
            resolved["facecolor"] = fc

        return resolved
