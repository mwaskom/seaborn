from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl

from seaborn._core.scales import ScaleSpec, Nominal, Continuous
from seaborn._core.rules import categorical_order, variable_type
from seaborn._compat import MarkerStyle
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Tuple, List, Union, Optional
    from pandas import Series
    from numpy.typing import ArrayLike
    from matplotlib.path import Path

    DashPattern = Tuple[float, ...]
    DashPatternWithOffset = Tuple[float, Optional[DashPattern]]
    MarkerPattern = Union[
        float,
        str,
        Tuple[int, int, float],
        List[Tuple[float, float]],
        Path,
        MarkerStyle,
    ]


class Property:

    legend = False
    normed = True

    _default_range: tuple[float, float]

    @property
    def default_range(self) -> tuple[float, float]:
        return self._default_range

    def default_scale(self, data: Series) -> ScaleSpec:
        # TODO use Boolean if we add that as a scale
        # TODO how will this handle data with units that can be treated as numeric
        # if passed through a registered matplotlib converter?
        var_type = variable_type(data, boolean_type="categorical")
        if var_type == "numeric":
            return Continuous()
        # TODO others ...
        else:
            return Nominal()

    def infer_scale(self, arg: Any, data: Series) -> ScaleSpec:
        # TODO what is best base-level default?
        var_type = variable_type(data)

        # TODO put these somewhere external for validation
        # TODO putting this here won't pick it up if subclasses define infer_scale
        # (e.g. color). How best to handle that? One option is to call super after
        # handling property-specific possibilities (e.g. for color check that the
        # arg is not a valid palette name) but that could get tricky.
        trans_args = ["log", "symlog", "logit", "pow", "sqrt"]
        if isinstance(arg, str) and any(arg.startswith(k) for k in trans_args):
            return Continuous(transform=arg)

        # TODO should Property have a default transform, i.e. "sqrt" for PointSize?

        if var_type == "categorical":
            return Nominal(arg)
        else:
            return Continuous(arg)

    def get_mapping(
        self, scale: ScaleSpec, data: Series
    ) -> Callable[[ArrayLike], ArrayLike] | None:

        return None


class Coordinate(Property):

    legend = False
    normed = False


class SemanticProperty(Property):
    legend = True


class SizedProperty(SemanticProperty):

    # TODO pass default range to constructor and avoid defining a bunch of subclasses?
    _default_range: tuple[float, float] = (0, 1)

    def _get_categorical_mapping(self, scale, data):

        levels = categorical_order(data, scale.order)

        if scale.values is None:
            vmin, vmax = self.default_range
            values = np.linspace(vmax, vmin, len(levels))
        elif isinstance(scale.values, tuple):
            vmin, vmax = scale.values
            values = np.linspace(vmax, vmin, len(levels))
        elif isinstance(scale.values, dict):
            # TODO check dict not missing levels
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            # TODO check list length
            values = scale.values
        else:
            # TODO nice error message
            assert False

        def mapping(x):
            ixs = x.astype(np.intp)
            out = np.full(x.shape, np.nan)
            use = np.isfinite(x)
            out[use] = np.take(values, ixs[use])
            return out

        return mapping

    def get_mapping(self, scale, data):

        if isinstance(scale, Nominal):
            return self._get_categorical_mapping(scale, data)

        if scale.values is None:
            vmin, vmax = self.default_range
        else:
            vmin, vmax = scale.values

        def f(x):
            return x * (vmax - vmin) + vmin

        return f


class PointSize(SizedProperty):
    _default_range = 2, 8


class LineWidth(SizedProperty):
    @property
    def default_range(self) -> tuple[float, float]:
        base = mpl.rcParams["lines.linewidth"]
        return base * .5, base * 2


class EdgeWidth(SizedProperty):
    @property
    def default_range(self) -> tuple[float, float]:
        base = mpl.rcParams["patch.linewidth"]
        return base * .5, base * 2


class ObjectProperty(SemanticProperty):
    # TODO better name; this is unclear?

    null_value: Any = None

    # TODO add abstraction for logic-free default scale type?
    def default_scale(self, data):
        return Nominal()

    def infer_scale(self, arg, data):
        return Nominal(arg)

    def get_mapping(self, scale, data):

        levels = categorical_order(data, scale.order)
        n = len(levels)

        if isinstance(scale.values, dict):
            # self._check_dict_not_missing_levels(levels, values)
            # TODO where to ensure that dict values have consistent representation?
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            # colors = self._ensure_list_not_too_short(levels, values)
            # TODO check not too long also?
            values = scale.values
        elif scale.values is None:
            values = self._default_values(n)
        else:
            # TODO add nice error message
            assert False, values

        values = self._standardize_values(values)

        def mapping(x):
            ixs = x.astype(np.intp)
            return [
                values[ix] if np.isfinite(x_i) else self.null_value
                for x_i, ix in zip(x, ixs)
            ]

        return mapping

    def _default_values(self, n):
        raise NotImplementedError()

    def _standardize_values(self, values):

        return values


class Marker(ObjectProperty):

    normed = False

    null_value = MarkerStyle("")

    # TODO should we have named marker "palettes"? (e.g. see d3 options)

    # TODO will need abstraction to share with LineStyle, etc.

    # TODO need some sort of "require_scale" functionality
    # to raise when we get the wrong kind explicitly specified

    def _standardize_values(self, values):

        return [MarkerStyle(x) for x in values]

    def _default_values(self, n: int) -> list[MarkerStyle]:
        """Build an arbitrarily long list of unique marker styles for points.

        Parameters
        ----------
        n : int
            Number of unique marker specs to generate.

        Returns
        -------
        markers : list of string or tuples
            Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
            All markers will be filled.

        """
        # Start with marker specs that are well distinguishable
        markers = [
            "o",
            "X",
            (4, 0, 45),
            "P",
            (4, 0, 0),
            (4, 1, 0),
            "^",
            (4, 1, 45),
            "v",
        ]

        # Now generate more from regular polygons of increasing order
        s = 5
        while len(markers) < n:
            a = 360 / (s + 1) / 2
            markers.extend([
                (s + 1, 1, a),
                (s + 1, 0, a),
                (s, 1, 0),
                (s, 0, 0),
            ])
            s += 1

        markers = [MarkerStyle(m) for m in markers[:n]]

        return markers


class LineStyle(ObjectProperty):

    null_value = ""

    def _default_values(self, n: int):  # -> list[DashPatternWithOffset]:
        """Build an arbitrarily long list of unique dash styles for lines.

        Parameters
        ----------
        n : int
            Number of unique dash specs to generate.

        Returns
        -------
        dashes : list of strings or tuples
            Valid arguments for the ``dashes`` parameter on
            :class:`matplotlib.lines.Line2D`. The first spec is a solid
            line (``""``), the remainder are sequences of long and short
            dashes.

        """
        # Start with dash specs that are well distinguishable
        dashes = [  # TODO : list[str | DashPattern] = [
            "-",  # TODO do we need to handle this elsewhere for backcompat?
            (4, 1.5),
            (1, 1),
            (3, 1.25, 1.5, 1.25),
            (5, 1, 1, 1),
        ]

        # Now programmatically build as many as we need
        p = 3
        while len(dashes) < n:

            # Take combinations of long and short dashes
            a = itertools.combinations_with_replacement([3, 1.25], p)
            b = itertools.combinations_with_replacement([4, 1], p)

            # Interleave the combinations, reversing one of the streams
            segment_list = itertools.chain(*zip(
                list(a)[1:-1][::-1],
                list(b)[1:-1]
            ))

            # Now insert the gaps
            for segments in segment_list:
                gap = min(segments)
                spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
                dashes.append(spec)

            p += 1

        return self._standardize_values(dashes)

    def _standardize_values(self, values):
        """Standardize values as dash pattern (with offset)."""
        return [self._get_dash_pattern(x) for x in values]

    @staticmethod
    def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
        """Convert linestyle arguments to dash pattern with offset."""
        # Copied and modified from Matplotlib 3.4
        # go from short hand -> full strings
        ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
        if isinstance(style, str):
            style = ls_mapper.get(style, style)
            # un-dashed styles
            if style in ['solid', 'none', 'None']:
                offset = 0
                dashes = None
            # dashed styles
            elif style in ['dashed', 'dashdot', 'dotted']:
                offset = 0
                dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])

        elif isinstance(style, tuple):
            if len(style) > 1 and isinstance(style[1], tuple):
                offset, dashes = style
            elif len(style) > 1 and style[1] is None:
                offset, dashes = style
            else:
                offset = 0
                dashes = style
        else:
            raise ValueError(f'Unrecognized linestyle: {style}')

        # Normalize offset to be positive and shorter than the dash cycle
        if dashes is not None:
            dsum = sum(dashes)
            if dsum:
                offset %= dsum

        return offset, dashes


class Color(SemanticProperty):

    def infer_scale(self, arg, data) -> ScaleSpec:

        # TODO do color standardization on dict / list values?
        if isinstance(arg, (dict, list)):
            return Nominal(arg)

        if isinstance(arg, tuple):
            return Continuous(arg)

        if callable(arg):
            return Continuous(arg)

        # TODO Do we accept str like "log", "pow", etc. for semantics?

        # TODO what about
        # - Temporal? (i.e. datetime)
        # - Boolean?

        assert isinstance(arg, str)  # TODO sanity check

        var_type = (
            "categorical" if arg in QUAL_PALETTES
            else variable_type(data, boolean_type="categorical")
        )

        if var_type == "categorical":
            return Nominal(arg)

        if var_type == "numeric":
            return Continuous(arg)

        # TODO just to see when we get here
        assert False

    def _get_categorical_mapping(self, scale, data):

        levels = categorical_order(data, scale.order)
        n = len(levels)
        values = scale.values

        if isinstance(values, dict):
            # self._check_dict_not_missing_levels(levels, values)
            # TODO where to ensure that dict values have consistent representation?
            colors = [values[x] for x in levels]
        else:
            if values is None:
                if n <= len(get_color_cycle()):
                    # Use current (global) default palette
                    colors = color_palette(n_colors=n)
                else:
                    colors = color_palette("husl", n)
            elif isinstance(values, list):
                # colors = self._ensure_list_not_too_short(levels, values)
                # TODO check not too long also?
                colors = color_palette(values)
            else:
                colors = color_palette(values, n)

        def mapping(x):
            ixs = x.astype(np.intp)
            use = np.isfinite(x)
            out = np.full((len(x), 3), np.nan)  # TODO rgba?
            out[use] = np.take(colors, ixs[use], axis=0)
            return out

        return mapping

    def get_mapping(self, scale, data):

        # TODO what is best way to do this conditional?
        if isinstance(scale, Nominal):
            return self._get_categorical_mapping(scale, data)

        elif scale.values is None:
            # TODO data-dependent default type
            # (Or should caller dispatch to function / dictionary mapping?)
            mapping = color_palette("ch:", as_cmap=True)
        elif isinstance(scale.values, tuple):
            mapping = blend_palette(scale.values, as_cmap=True)
        elif isinstance(scale.values, str):
            # TODO data-dependent return type?
            # TODO for matplotlib colormaps this will clip, which is different behavior
            mapping = color_palette(scale.values, as_cmap=True)

        # TODO just during dev
        else:
            assert False

        # TODO figure out better way to do this, maybe in color_palette?
        # Also note that this does not preserve alpha channels when given
        # as part of the range values, which we want.
        def _mapping(x):
            return mapping(x)[:, :3]

        return _mapping


class Alpha(SizedProperty):
    # TODO Calling Alpha "Sized" seems wrong, but they share the basic mechanics
    # aside from Alpha having an upper bound.
    _default_range = .15, .95
    # TODO validate that output is in [0, 1]


class Fill(SemanticProperty):

    normed = False

    # TODO default to Nominal scale always?

    def default_scale(self, data):
        return Nominal()

    def infer_scale(self, arg, data):
        return Nominal(arg)

    def _default_values(self, n: int) -> list:
        """Return a list of n values, alternating True and False."""
        if n > 2:
            msg = " ".join([
                "There are only two possible `fill` values,",
                # TODO allowing each Property instance to have a variable name
                # is useful for good error message, but disabling for now
                # f"There are only two possible {self.variable} values,",
                "so they will cycle and may produce an uninterpretable plot",
            ])
            warnings.warn(msg, UserWarning)
        return [x for x, _ in zip(itertools.cycle([True, False]), range(n))]

    def get_mapping(self, scale, data):

        order = categorical_order(data, scale.order)

        if isinstance(scale.values, pd.Series):
            # What's best here? If we simply cast to bool, np.nan -> False, bad!
            # "boolean"/BooleanDType, is described as experimental/subject to change
            # But if we don't require any particular behavior, is that ok?
            # See https://github.com/pandas-dev/pandas/issues/44293
            values = scale.values.astype("boolean").to_list()
        elif isinstance(scale.values, list):
            values = [bool(x) for x in scale.values]
        elif isinstance(scale.values, dict):
            values = [bool(scale.values[x]) for x in order]
        elif scale.values is None:
            values = self._default_values(len(order))
        else:
            raise TypeError(f"Type of `values` ({type(scale.values)}) not understood.")

        def mapping(x):
            return np.take(values, x.astype(np.intp))

        return mapping


# TODO should these be instances or classes?
PROPERTIES = {
    "x": Coordinate(),
    "y": Coordinate(),
    "color": Color(),
    "fillcolor": Color(),
    "edgecolor": Color(),
    "alpha": Alpha(),
    "fillalpha": Alpha(),
    "edgealpha": Alpha(),
    "fill": Fill(),
    "marker": Marker(),
    "linestyle": LineStyle(),
    "pointsize": PointSize(),
    "linewidth": LineWidth(),
    "edgewidth": EdgeWidth(),
}
