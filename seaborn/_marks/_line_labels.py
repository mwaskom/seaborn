import dataclasses
import operator
from typing import ClassVar, Any, Mapping
import logging
import matplotlib as mpl
from matplotlib import collections
import pandas as pd
import numpy as np
import scipy.optimize
import seaborn.objects as so
from seaborn._core.groupby import GroupBy
from seaborn._marks.base import (
    Mappable,
    MappableColor,
    MappableFloat,
    MappableString,
    resolve_color,
    resolve_properties,
)


@dataclasses.dataclass
class LineLabel(so.Mark):
    text: MappableString = Mappable("")
    color: MappableColor = Mappable("k")
    alpha: MappableFloat = Mappable(1)
    fontsize: MappableFloat = Mappable(rc="font.size")
    offset: float = 4
    additional_distance_offset: float = 0

    def _compute_target_positions(
        self,
        data: dict[mpl.axes.Axes, list[Mapping[str, Any]]],  # pyright: ignore
        scales,
        other: str,
        *,
        offset: float = 0,
    ) -> dict[mpl.axes.Axes, np.ndarray]:  # pyright: ignore
        """Solves a constrained optimization problem to determine the optimal target point positions."""
        # https://github.com/nschloe/matplotx/blob/main/src/matplotx/_labels.py
        target_positions: dict[mpl.axes.Axes, np.ndarray] = {}  # pyright: ignore
        point_dtype = np.dtype([("x", "f8"), ("y", "f8")])

        def _resolve_fontsize(keys) -> float:
            return resolve_properties(self, keys, scales)["fontsize"]

        for ax, rows in data.items():
            # Calculate offsets for each target based on the fontsize and additional offset.
            offsets = np.array(
                [[_resolve_fontsize(row["_keys"]) * (5 / 3) + offset] for row in rows]
            )
            # Transform points to screen coordinates so min_distance_apart is scale-agnostic.
            points = ax.transData.transform([(row["x"], row["y"]) for row in rows])
            points = points.view(point_dtype)
            # Record the sorting indices so we can recover the original order.
            sorted_indexes = np.argsort(points, axis=0, order=other)
            original_indexes = np.argsort(sorted_indexes, axis=0)
            sorted_offsets = np.take_along_axis(offsets, sorted_indexes, axis=0)
            sorted_points = np.take_along_axis(points, sorted_indexes, axis=0)

            # Calculate min y0 position to bootstrap the first index.
            num_points = points.size
            min_point = sorted_points[other][0] - num_points * sorted_offsets[0]

            # Solve non-negative least squares problem
            A = np.tril(np.ones((num_points, num_points)))
            b = sorted_points[other].squeeze(1) - (
                min_point + np.arange(num_points) * sorted_offsets.squeeze(1)
            )
            sol, objective_value = scipy.optimize.nnls(A, b)
            # Recover points
            sol = (
                np.cumsum(sol)
                + min_point
                + np.arange(num_points) * sorted_offsets.squeeze(1)
            )
            sol = np.take_along_axis(sol[:, np.newaxis], original_indexes, axis=0)
            logging.info(
                "Found line label positions with final objective value: %f",
                objective_value,
            )

            # Update original points
            points[other] = sol

            # Transform back to data coordinates
            screen_to_data = ax.transData.inverted()
            target_positions[ax] = screen_to_data.transform(points.view("f8")).view(
                point_dtype
            )

        return target_positions

    def _plot(self, split_gen, scales, orient):
        data_by_axes: dict[
            mpl.axes.Axes, list[Mapping[str, Any]]
        ] = collections.defaultdict(
            list
        )  # pyright: ignore

        other = {"x": "y", "y": "x"}[orient]
        for keys, data, ax in split_gen():
            records = data.query(f"`{orient}` == {orient}.max()").to_dict("records")
            records = collections.ChainMap(*records, {"_keys": keys})
            data_by_axes[ax].append(records)

        target_positions = self._compute_target_positions(
            data_by_axes,
            scales,
            other,
            offset=self.additional_distance_offset,
        )
        for ax, data in data_by_axes.items():
            for idx, row in enumerate(data):
                vals = resolve_properties(self, row["_keys"], scales)
                color = resolve_color(self, row["_keys"], "", scales)
                fontsize = vals["fontsize"]

                transform = mpl.transforms.offset_copy(  # pyright: ignore
                    ax.transData,
                    fig=ax.figure,
                    x=self.offset if orient == "x" else 0,
                    y=self.offset if orient == "y" else 0,
                    units="points",
                )

                target_position = target_positions[ax][idx]
                ax.add_artist(
                    mpl.text.Text(  # pyright: ignore
                        x=target_position["x"],
                        y=target_position["y"],
                        text=str(row.get("text", vals["text"])),
                        color=color,
                        fontsize=fontsize,
                        horizontalalignment="left" if orient == "x" else "center",
                        verticalalignment="center" if orient == "x" else "bottom",
                        transform=transform,
                        rotation=90 if orient == "y" else 0,
                        zorder=2,
                        clip_on=False,
                        in_layout=True,
                        **self.artist_kws,
                    )
                )
