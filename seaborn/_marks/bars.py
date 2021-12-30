from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import Mark, Feature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional

    MappableBool = Union[bool, Feature]
    MappableFloat = Union[float, Feature]
    MappableString = Union[str, Feature]
    MappableColor = Union[str, tuple, Feature]  # TODO


@dataclass
class Bar(Mark):

    color: MappableColor = Feature("C0")
    alpha: MappableFloat = Feature(1)
    fill: MappableBool = Feature(True)
    pattern: MappableString = Feature(None)
    width: MappableFloat = Feature(.8)

    baseline: float = 0
    multiple: Optional[str] = None

    def _adjust(self, df):

        # Abstract out the pos/val axes based on orientation
        if self.orient == "y":
            pos, val = "yx"
        else:
            pos, val = "xy"

        # Initialize vales for bar shape/location parameterization
        df = df.assign(
            width=self._resolve(df, "width"),
            baseline=self.baseline,
        )

        if self.multiple is None:
            return df

        # Now we need to know the levels of the grouping variables, hmmm.
        # Should `_plot_layer` pass that in here?
        # TODO maybe instead of that we have the dataframe sorted by categorical order?

        # Adjust as appropriate
        # TODO currently this does not check that it is necessary to adjust!
        if self.multiple.startswith("dodge"):

            # TODO this is pretty general so probably doesn't need to be in Bar.
            # but it will require a lot of work to fix up, especially related to
            # ordering of groups (including representing groups that are specified
            # in the variable levels but are not in the dataframe

            # TODO this implements "flexible" dodge, i.e. fill the original space
            # even with missing levels, which is nice and worth adding, but:
            # 1) we also need to implement "fixed" dodge
            # 2) we need to think of the right API for allowing that
            # The dodge/dodgefill thing is a provisional idea

            width_by_pos = df.groupby(pos, sort=False)["width"]
            if self.multiple == "dodgefill":  # Not great name given other "fill"
                # TODO e.g. what should we do here with empty categories?
                # is it too confusing if we appear to ignore "dodgefill",
                # or is it inconsistent with behavior elsewhere?
                max_by_pos = width_by_pos.max()
                sum_by_pos = width_by_pos.sum()
            else:
                # TODO meanwhile here, we do get empty space, but
                # it is always to the right of the bars that are there
                max_width = df["width"].max()
                max_by_pos = {p: max_width for p, _ in width_by_pos}
                max_sum = width_by_pos.sum().max()
                sum_by_pos = {p: max_sum for p, _ in width_by_pos}

            df.loc[:, "width"] = width_by_pos.transform(
                lambda x: (x / sum_by_pos[x.name]) * max_by_pos[x.name]
            )

            # TODO maybe this should be building a mapping dict for pos?
            # (It is probably less relevent for bars, but what about e.g.
            # a dense stripplot, where we'd be doing a lot more operations
            # than we need to be doing this way.
            df.loc[:, pos] = (
                df[pos]
                - df[pos].map(max_by_pos) / 2
                + width_by_pos.transform(
                    lambda x: x.shift(1).fillna(0).cumsum()
                )
                + df["width"] / 2
            )

        return df

    def _plot_split(self, keys, data, ax, kws):

        x, y = data[["x", "y"]].to_numpy().T
        b = data["baseline"]
        w = data["width"]

        if self.orient == "x":
            w, h = w, y - b
            xy = np.column_stack([x - w / 2, b])
        else:
            w, h = w, x - b
            xy = np.column_stack([b, y - h / 2])

        geometry = xy, w, h
        features = [
            self._resolve_color(data),  # facecolor
        ]

        bars = []
        for xy, w, h, fc in zip(*geometry, *features):
            bar = mpl.patches.Rectangle(
                xy=xy,
                width=w,
                height=h,
                facecolor=fc,
                # TODO leaving this incomplete for now
                # Need decision about the best way to parametrize color
            )
            ax.add_patch(bar)
            bars.append(bar)

        # TODO add container object to ax, line ax.bar does
