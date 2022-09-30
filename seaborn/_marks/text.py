from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib as mpl
from matplotlib.artist import Artist

from seaborn._marks.base import (
    Mark,
    Mappable,
    MappableFloat,
    MappableString,
    MappableColor,
    resolve_properties,
    document_properties,
)
from seaborn._core.scales import Scale


@document_properties
@dataclass
class Text(Mark):
    """
    TODO
    """

    text: MappableString = Mappable("")
    color: MappableColor = Mappable("k")
    fontsize: MappableFloat = Mappable(rc="font.size")
    halign: MappableString = Mappable("center")
    valign: MappableString = Mappable("center_baseline")

    def _plot(self, split_gen, scales, orient):

        ax_data = defaultdict(list)

        for keys, data, ax in split_gen():
            vals = resolve_properties(self, keys, scales)

            for row in data.to_dict("records"):
                artist = mpl.text.Text(
                    x=row["x"],
                    y=row["y"],
                    text=str(row["text"]),  # TODO format
                    color=vals["color"],
                    fontsize=vals["fontsize"],
                    horizontalalignment=vals["halign"],
                    verticalalignment=vals["valign"],
                    **self.artist_kws,
                )
                ax.add_artist(artist)
                ax_data[ax].append([row["x"], row["y"]])

        for ax, ax_vals in ax_data.items():
            ax.update_datalim(np.array(ax_vals))

    def _legend_artist(
        self, variables: list[str], value: Any, scales=dict[str, Scale],
    ) -> Artist:

        # TODO
        return mpl.line.Line2D()
