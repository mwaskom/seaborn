## Author: [Dominik Matula](https://github.com/matulad)

import dataclasses
from typing import ClassVar

import seaborn.objects as so
from seaborn._marks.base import (
    Mappable,
    MappableFloat,
    resolve_color,
    resolve_properties,
)

class AxlineBase(so.Path):
    """
    Abstract ancestor for the Axline marks.

    See also:
    ---------
    Axline : Arbitrary line mark.
    Axhline : Horizontal line mark.
    Axvline : Vertical line mark.
    """
    _sort: ClassVar[bool] = False

    def _get_passthrough_points(self, data: dict):
        raise NotImplementedError()
    
    def _plot(self, split_gen, scales, orient):
        
        for keys, data, ax in split_gen():
            vals = resolve_properties(self, keys, scales)
            
            # Use input data, because custom aesthetics are not supported, yet, 
            # and x/y values are not provided in `vals`. Later on, I would like
            # to utilize `intercept` and `slope` instead.
            if not "x" in vals and "x" in data.columns:
                vals["x"] = data["x"]
            if not "y" in vals and "y" in data.columns:
                vals["y"] = data["y"]

            vals["color"] = resolve_color(self, keys, scales=scales)
            vals["fillcolor"] = resolve_color(self, keys, prefix="fill", scales=scales)
            vals["edgecolor"] = resolve_color(self, keys, prefix="edge", scales=scales)

            artist_kws = self.artist_kws.copy()
            xy1, xy2 = self._get_passthrough_points(vals)

            for point1, point2 in zip(xy1, xy2):
                ax.axline(
                    point1,
                    point2,
                    color=vals["color"],
                    linewidth=vals["linewidth"],
                    linestyle=vals["linestyle"],
                    marker=vals["marker"],
                    markersize=vals["pointsize"],
                    markerfacecolor=vals["fillcolor"],
                    markeredgecolor=vals["edgecolor"],
                    markeredgewidth=vals["edgewidth"],
                    **artist_kws,
                )

@dataclasses.dataclass
class Axline(AxlineBase):
    """
    A mark adding *arbitrary* line to your plot.
    
    TODO: MAPPING NOT SUPPORTED YET.
    At this phase, we're able to use it with scalars only. 
    The structure is prepared, but there is a bug (or a feature)
    that limit's used scales to predefined aesthetics only. And 
    neither `intercept` nor `slope` is among them.

    Hotfix would utilize e.g. `x` instead of `intercept` and `y`
    instead of `slope`. BUT I won't to that here because it would
    be too confusing later on. Instead I'll postpone resolving 
    this issue until there is possibility to use own aesthetics.

    See also
    --------
    Axhline : A mark adding *horizontal* line to your plot.
    Axvline : A mark adding *vertical* line to your plot.

    Examples
    --------
    .. include:: ../docstrings/objects.Axline.rst    # TODO: Add
    """
    intercept: MappableFloat = Mappable(0)
    slope: MappableFloat =Mappable(1)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["intercept"], "__iter__"):
            vals["intercept"] = [vals["intercept"]]
        if not hasattr(vals["slope"], "__iter__"):
            vals["slope"] = [vals["slope"]]
            
        xy1 = [(0, intercept) for intercept in vals["intercept"]]
        xy2 = [(1, intercept + slope) for intercept, slope in zip(vals["intercept"], vals["slope"])]
        return xy1, xy2


@dataclasses.dataclass
class Axhline(AxlineBase):
    """
    A mark adding *horizontal* line to the plot.

    See also
    --------
    Axline : A mark adding *arbitrary* line to the plot.
    Axvline : A mark adding *vertical* line to the plot.

    Examples
    --------
    .. include:: ../docstrings/objects.Axhline.rst    # TODO: Add
    """

    y: MappableFloat = Mappable(0)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["y"], "__iter__"):
            vals["y"] = [vals["y"]]
        xy1 = ((0, y) for y in  vals["y"])
        xy2 = ((1, y) for y in  vals["y"])
        return xy1, xy2


@dataclasses.dataclass
class Axvline(AxlineBase):
    """
    A mark adding *vertical* line to the plot.

    See also
    --------
    Axline : A mark adding arbitrary line to the plot.
    Axhline : A mark adding horizontal line to the plot.

    Examples
    --------
    .. include:: ../docstrings/objects.Axvline.rst    # TODO: Add
    """
    x: MappableFloat = Mappable(0)

    def _get_passthrough_points(self, vals: dict):
        if not hasattr(vals["x"], '__iter__'):
            vals["x"] = [vals["x"]]
        xy1 = ((x, 0) for x in vals["x"])
        xy2 = ((x, 1) for x in vals["x"])
        return xy1, xy2