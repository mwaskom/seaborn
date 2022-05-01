from __future__ import annotations
from dataclasses import dataclass

import matplotlib as mpl

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableFloat,
    MappableString,
    MappableColor,
)


# TODO the collection of marks defined here is a holdover from very early
# "let's just got some plots on the screen" phase. They should maybe go elsewhere.


@dataclass
class Line(Mark):
    """
    A mark connecting data points with sorting along the orientation axis.
    """

    # TODO other semantics (marker?)

    color: MappableColor = Mappable("C0", groups=True)
    alpha: MappableFloat = Mappable(1, groups=True)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth", groups=True)
    linestyle: MappableString = Mappable(rc="lines.linestyle", groups=True)

    # TODO alternately, have Path mark that doesn't sort
    sort: bool = True

    def plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen():

            keys = self.resolve_properties(keys, scales)

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

        key = self.resolve_properties({v: value for v in variables}, scales)

        return mpl.lines.Line2D(
            [], [],
            color=key["color"],
            alpha=key["alpha"],
            linewidth=key["linewidth"],
            linestyle=key["linestyle"],
        )
