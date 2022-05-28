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
    resolve_color,
)
from seaborn.external.version import Version


@dataclass
class Path(Mark):
    """
    A mark connecting data points in the order they appear.
    """
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

        for keys, data, ax in split_gen(keep_na=not self._sort):

            vals = resolve_properties(self, keys, scales)
            vals["color"] = resolve_color(self, keys, scales=scales)
            vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
            vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

            # https://github.com/matplotlib/matplotlib/pull/16692
            if Version(mpl.__version__) < Version("3.3.0"):
                vals["marker"] = vals["marker"]._marker

            if self._sort:
                data = data.sort_values(orient)

            line = mpl.lines.Line2D(
                data["x"].to_numpy(),
                data["y"].to_numpy(),
                color=vals["color"],
                linewidth=vals["linewidth"],
                linestyle=vals["linestyle"],
                marker=vals["marker"],
                markersize=vals["pointsize"],
                markerfacecolor=vals["fillcolor"],
                markeredgecolor=vals["edgecolor"],
                markeredgewidth=vals["edgewidth"],
                **self.artist_kws,
            )
            ax.add_line(line)

    def _legend_artist(self, variables, value, scales):

        keys = {v: value for v in variables}
        vals = resolve_properties(self, keys, scales)
        vals["color"] = resolve_color(self, keys, scales=scales)
        vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
        vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

        # https://github.com/matplotlib/matplotlib/pull/16692
        if Version(mpl.__version__) < Version("3.3.0"):
            vals["marker"] = vals["marker"]._marker

        return mpl.lines.Line2D(
            [], [],
            color=vals["color"],
            linewidth=vals["linewidth"],
            linestyle=vals["linestyle"],
            marker=vals["marker"],
            markersize=vals["pointsize"],
            markerfacecolor=vals["fillcolor"],
            markeredgecolor=vals["edgecolor"],
            markeredgewidth=vals["edgewidth"],
            **self.artist_kws,
        )


@dataclass
class Line(Path):
    """
    A mark connecting data points with sorting along the orientation axis.
    """
    _sort: ClassVar[bool] = True


@dataclass
class Paths(Mark):
    """
    A faster but less-flexible mark for drawing many paths.
    """
    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(1)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    linestyle: MappableString = Mappable(rc="lines.linestyle")

    _sort: ClassVar[bool] = False

    def _plot(self, split_gen, scales, orient):

        line_data = {}

        for keys, data, ax in split_gen(keep_na=not self._sort):

            if ax not in line_data:
                line_data[ax] = {
                    "segments": [],
                    "colors": [],
                    "linewidths": [],
                    "linestyles": [],
                }

            vals = resolve_properties(self, keys, scales)
            vals["color"] = resolve_color(self, keys, scales=scales)

            if self._sort:
                data = data.sort_values(orient)

            # TODO comment about block consolidation
            xy = np.column_stack([data["x"], data["y"]])
            line_data[ax]["segments"].append(xy)
            line_data[ax]["colors"].append(vals["color"])
            line_data[ax]["linewidths"].append(vals["linewidth"])
            line_data[ax]["linestyles"].append(vals["linestyle"])

        for ax, ax_data in line_data.items():
            lines = mpl.collections.LineCollection(
                **ax_data,
                **self.artist_kws,
            )
            ax.add_collection(lines, autolim=False)
            # https://github.com/matplotlib/matplotlib/issues/23129
            # TODO get paths from lines object?
            xy = np.concatenate(ax_data["segments"])
            ax.dataLim.update_from_data_xy(
                xy, ax.ignore_existing_data_limits, updatex=True, updatey=True
            )

    def _legend_artist(self, variables, value, scales):

        key = resolve_properties(self, {v: value for v in variables}, scales)

        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
            **self.artist_kws,
        )


@dataclass
class Lines(Paths):
    """
    A faster but less-flexible mark for drawing many lines.
    """
    _sort: ClassVar[bool] = True
