from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableFloat,
    MappableString,
    MappableColor,
    resolve_properties,
)


@dataclass
class Path(Mark):
    """
    A mark connecting data points in the order they appear.
    """
    # TODO other semantics (marker?)

    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(1)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    linestyle: MappableString = Mappable(rc="lines.linestyle")
    marker: MappableString = Mappable(rc="lines.marker")
    pointsize: MappableFloat = Mappable(rc="lines.markersize")
    fillcolor: MappableColor = Mappable(depend="color")
    edgecolor: MappableColor = Mappable(depend="color")
    edgewidth: MappableFloat = Mappable(rc="lines.markeredgewidth")

    _sort: ClassVar[bool] = False

    def _plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen(dropna=False):

            keys = resolve_properties(self, keys, scales)

            if self._sort:
                data = data.sort_values(orient)
            else:
                data.loc[data.isna().any(axis=1), ["x", "y"]] = np.nan

            line = mpl.lines.Line2D(
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                color=keys["color"],
                alpha=keys["alpha"],
                linewidth=keys["linewidth"],
                linestyle=keys["linestyle"],
                marker=keys["marker"],
                markersize=keys["pointsize"],
                markerfacecolor=keys["fillcolor"],
                markeredgecolor=keys["edgecolor"],
                markeredgewidth=keys["edgewidth"],
                **self.artist_kws,
            )
            ax.add_line(line)

    def _legend_artist(self, variables, value, scales):

        key = resolve_properties(self, {v: value for v in variables}, scales)

        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            alpha=key["alpha"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
            marker=key["marker"],
            markersize=key["pointsize"],
            markerfacecolor=key["fillcolor"],
            markeredgecolor=key["edgecolor"],
            markeredgewidth=key["edgewidth"],
            **self.artist_kws,
        )


@dataclass
class Line(Path):
    """
    A mark connecting data points with sorting along the orientation axis.
    """
    _sort: ClassVar[bool] = True
