"""
Classes that, together with the scales module, implement semantic mapping logic.

Semantic mappings in seaborn transform data values into visual features.
The implementations in this module output values that are suitable arguments for
matplotlib artists or plotting functions.

There are two main class hierarchies here: Semantic classes and SemanticMapping
classes. One way to think of the relationship is that a Semantic is a partial
initialization of a SemanticMapping. Semantics hold the parameters specified by
the user through the Plot interface and contain methods relevant to defining
default values for specific visual features (e.g. generating arbitrarily-large
sets of distinct marker shapes) or standardizing user-provided values. The
user-specified (or default) parameters are then used in combination with the
data values to setup the SemanticMapping objects that are used to actually
create the plot. SemanticMappings are more general, and they operate using just
a few different patterns.

Unlike the original articulation of the grammar of graphics, or other
implementations, seaborn makes some distinctions between the concepts of
"scaling" and "mapping", both in the internal code and in the external
interfaces. Semantic mapping uses scales when there are numeric or ordinal
relationships between inputs, but the scale abstraction is not used for
transforming inputs into discrete output values. This is partly for historical
reasons (some concepts were introduced in ways that are difficult to re-express
only using scales), and also because it feels more natural to use a dictionary
lookup as the core operation for mapping discrete properties, such as marker shape
or dash pattern.

"""
from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl

from seaborn._compat import MarkerStyle
from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn.utils import get_color_cycle
from seaborn.palettes import QUAL_PALETTES, color_palette

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Tuple, List, Optional, Union
    from numbers import Number
    from numpy.typing import ArrayLike
    from pandas import Series
    from matplotlib.colors import Colormap
    from matplotlib.scale import Scale
    from matplotlib.path import Path
    from seaborn._core.typing import PaletteSpec, DiscreteValueSpec, ContinuousValueSpec

    RGBTuple = Tuple[float, float, float]
    RGBATuple = Tuple[float, float, float, float]
    ColorSpec = Union[RGBTuple, RGBATuple, str]

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


class Semantic:
    """Holds semantic mapping parameters and creates mapping based on data."""
    variable: str

    def setup(self, data: Series, scale: Scale) -> SemanticMapping:
        """Define the semantic mapping using data values."""
        raise NotImplementedError

    def _standardize_value(self, value: Any) -> Any:
        """Convert value to a standardized representation."""
        return value

    def _standardize_values(
        self, values: DiscreteValueSpec | Series
    ) -> DiscreteValueSpec | Series:
        """Convert collection of values to standardized representations."""
        if values is None:
            return None
        elif isinstance(values, dict):
            return {k: self._standardize_value(v) for k, v in values.items()}
        elif isinstance(values, pd.Series):
            return values.map(self._standardize_value)
        else:
            return [self._standardize_value(x) for x in values]

    def _check_dict_not_missing_levels(self, levels: list, values: dict) -> None:
        """Input check when values are provided as a dictionary."""
        missing = set(levels) - set(values)
        if missing:
            formatted = ", ".join(map(repr, sorted(missing, key=str)))
            err = f"Missing {self.variable} for following value(s): {formatted}"
            raise ValueError(err)

    def _ensure_list_not_too_short(self, levels: list, values: list) -> list:
        """Input check when values are provided as a list."""
        if len(levels) > len(values):
            msg = " ".join([
                f"The {self.variable} list has fewer values ({len(values)})",
                f"than needed ({len(levels)}) and will cycle, which may",
                "produce an uninterpretable plot."
            ])
            warnings.warn(msg, UserWarning)

            values = [x for _, x in zip(levels, itertools.cycle(values))]

        return values


class DiscreteSemantic(Semantic):
    """Define semantic mapping where output values have no numeric relationship."""
    def __init__(self, values: DiscreteValueSpec, variable: str):
        self.values = self._standardize_values(values)
        self.variable = variable

    def _standardize_values(
        self, values: DiscreteValueSpec | Series
    ) -> DiscreteValueSpec | Series:
        """Convert collection of values to standardized representations."""
        if values is None:
            return values
        elif isinstance(values, pd.Series):
            return values.map(self._standardize_value)
        else:
            return super()._standardize_values(values)

    def _default_values(self, n: int) -> list:
        """Return n unique values. Must be defined by subclass if used."""
        raise NotImplementedError

    def setup(self, data: Series, scale: Scale) -> LookupMapping:
        """Define the mapping using data values."""
        scale = scale.setup(data)
        levels = categorical_order(data, scale.order)

        if self.values is None:
            mapping = dict(zip(levels, self._default_values(len(levels))))
        elif isinstance(self.values, dict):
            self._check_dict_not_missing_levels(levels, self.values)
            mapping = self.values
        elif isinstance(self.values, list):
            values = self._ensure_list_not_too_short(levels, self.values)
            mapping = dict(zip(levels, values))

        return LookupMapping(mapping)


class BooleanSemantic(DiscreteSemantic):
    """Semantic mapping where only possible output values are True or False."""
    def _standardize_value(self, value: Any) -> bool:
        return bool(value)

    def _standardize_values(
        self, values: DiscreteValueSpec | Series
    ) -> DiscreteValueSpec | Series:
        """Convert values into booleans using Python's truthy rules."""
        if isinstance(values, pd.Series):
            # What's best here? If we simply cast to bool, np.nan -> False, bad!
            # "boolean"/BooleanDType, is described as experimental/subject to change
            # But if we don't require any particular behavior, is that ok?
            # See https://github.com/pandas-dev/pandas/issues/44293
            return values.astype("boolean")
        elif isinstance(values, list):
            return [bool(x) for x in values]
        elif isinstance(values, dict):
            return {k: bool(v) for k, v in values.items()}
        elif values is None:
            return None
        else:
            raise TypeError(f"Type of `values` ({type(values)}) not understood.")

    def _default_values(self, n: int) -> list:
        """Return a list of n values, alternating True and False."""
        if n > 2:
            msg = " ".join([
                f"There are only two possible {self.variable} values,",
                "so they will cycle and may produce an uninterpretable plot",
            ])
            warnings.warn(msg, UserWarning)
        return [x for x, _ in zip(itertools.cycle([True, False]), range(n))]


class ContinuousSemantic(Semantic):
    """Semantic mapping where output values have numeric relationships."""
    _default_range: tuple[float, float] = (0, 1)

    def __init__(self, values: ContinuousValueSpec = None, variable: str = ""):

        if values is None:
            values = self.default_range
        self.values = self._standardize_values(values)
        self.variable = variable

    @property
    def default_range(self) -> tuple[float, float]:
        """Default output range; implemented as a property so rcParams can be used."""
        return self._default_range

    def _standardize_value(self, value: Any) -> float:
        """Convert value to float for numeric operations."""
        return float(value)

    def _standardize_values(self, values: ContinuousValueSpec) -> ContinuousValueSpec:

        if isinstance(values, tuple):
            lo, hi = values
            return self._standardize_value(lo), self._standardize_value(hi)
        return super()._standardize_values(values)

    def _infer_map_type(
        self,
        scale: Scale,
        values: ContinuousValueSpec,
        data: Series,
    ) -> VarType:
        """Determine how to implement the mapping based on parameters or data."""
        if scale.type_declared:
            return scale.scale_type
        elif isinstance(values, (list, dict)):
            return VarType("categorical")
        else:
            return variable_type(data, boolean_type="categorical")

    def setup(self, data: Series, scale: Scale) -> SemanticMapping:
        """Define the mapping using data values."""
        scale = scale.setup(data)
        map_type = self._infer_map_type(scale, self.values, data)

        if map_type == "categorical":

            levels = categorical_order(data, scale.order)
            if isinstance(self.values, tuple):
                numbers = np.linspace(1, 0, len(levels))
                transform = RangeTransform(self.values)
                mapping_dict = dict(zip(levels, transform(numbers)))
            elif isinstance(self.values, dict):
                self._check_dict_not_missing_levels(levels, self.values)
                mapping_dict = self.values
            elif isinstance(self.values, list):
                values = self._ensure_list_not_too_short(levels, self.values)
                # TODO check list not too long as well?
                mapping_dict = dict(zip(levels, values))

            return LookupMapping(mapping_dict)

        if not isinstance(self.values, tuple):
            # We shouldn't actually get here through the Plot interface (there is a
            # guard upstream), but this check prevents mypy from complaining.
            t = type(self.values).__name__
            raise TypeError(
                f"Using continuous {self.variable} mapping, but values provided as {t}."
            )
        transform = RangeTransform(self.values)
        return NormedMapping(scale, transform)


# ==================================================================================== #


class ColorSemantic(Semantic):
    """Semantic mapping that produces RGB colors."""
    def __init__(self, palette: PaletteSpec = None, variable: str = "color"):
        self.palette = palette
        self.variable = variable

    def _standardize_value(
        self, value: str | RGBTuple | RGBATuple
    ) -> RGBTuple | RGBATuple:

        has_alpha = (
            (isinstance(value, str) and value.startswith("#") and len(value) in [5, 9])
            or (isinstance(value, tuple) and len(value) == 4)
        )
        rgb_func = mpl.colors.to_rgba if has_alpha else mpl.colors.to_rgb

        return rgb_func(value)

    def _standardize_values(
        self, values: DiscreteValueSpec | Series
    ) -> list[RGBTuple | RGBATuple] | dict[Any, RGBTuple | RGBATuple] | None:
        """Standardize colors as an RGB tuple or n x 3 RGB array."""
        if values is None:
            return None
        elif isinstance(values, dict):
            return {k: self._standardize_value(v) for k, v in values.items()}
        else:
            return list(map(self._standardize_value, values))

    def setup(self, data: Series, scale: Scale) -> SemanticMapping:
        """Define the mapping using data values."""
        # TODO We also need to add some input checks ...
        # e.g. specifying a numeric scale and a qualitative colormap should fail nicely.

        # TODO FIXME:mappings
        # In current function interface, we can assign a numeric variable to hue and set
        # either a named qualitative palette or a list/dict of colors.
        # In current implementation here, that raises with an unpleasant error.
        # The problem is that the scale.type currently dominates.
        # How to distinguish between "user set numeric scale and qualitative palette,
        # this is an error" from "user passed numeric values but did not set explicit
        # scale, then asked for a qualitative mapping by the form of the palette?

        scale = scale.setup(data)
        map_type = self._infer_map_type(scale, self.palette, data)

        if map_type == "categorical":
            return LookupMapping(
                self._setup_categorical(data, self.palette, scale.order)
            )

        lookup, transform = self._setup_numeric(data, self.palette)
        if lookup:
            # TODO See comments in _setup_numeric about deprecation of this
            return LookupMapping(lookup)
        else:
            return NormedMapping(scale, transform)

    def _setup_categorical(
        self,
        data: Series,
        palette: PaletteSpec,
        order: list | None,
    ) -> dict[Any, RGBTuple | RGBATuple]:
        """Determine colors when the mapping is categorical."""
        levels = categorical_order(data, order)
        n_colors = len(levels)

        if isinstance(palette, dict):
            self._check_dict_not_missing_levels(levels, palette)
            mapping = palette
        else:
            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    # None uses current (global) default palette
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                colors = self._ensure_list_not_too_short(levels, palette)
                # TODO check not too long also?
            else:
                colors = color_palette(palette, n_colors)
            mapping = dict(zip(levels, colors))

        # It would be cleaner to have this check in standardize_values, but that
        # makes the typing a little tricky. The right solution is to properly type
        # the function so that we know the return type matches the input type.
        mapping = {k: self._standardize_value(v) for k, v in mapping.items()}
        if len(set(len(v) for v in mapping.values())) > 1:
            err = "Palette cannot mix colors defined with and without alpha channel."
            raise ValueError(err)

        return mapping

    def _setup_numeric(
        self,
        data: Series,
        palette: PaletteSpec,
    ) -> tuple[dict[Any, tuple[float, float, float, float]], Callable[[Series], Any]]:
        """Determine colors when the variable is quantitative."""
        cmap: Colormap
        if isinstance(palette, dict):

            # In the function interface, the presence of a norm object overrides
            # a dictionary of colors to specify a numeric mapping, so we need
            # to process it here.
            # TODO this functionality only exists to support the old relplot
            # hack for linking hue orders across facets.  We don't need that any
            # more and should probably remove this, but needs deprecation.
            # (Also what should new behavior be? I think an error probably).
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            mapping = palette.copy()

        else:

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # This is something we may revisit and change; it has drawbacks
            palette = "ch:" if palette is None else palette

            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)

            mapping = {}

        transform = RGBTransform(cmap)

        return mapping, transform

    def _infer_map_type(
        self,
        scale: Scale,
        palette: PaletteSpec,
        data: Series,
    ) -> VarType:
        """Infer type of color mapping based on relevant parameters."""
        map_type: VarType
        if scale is not None and scale.type_declared:
            return scale.scale_type
        elif palette in QUAL_PALETTES:
            map_type = VarType("categorical")
        elif isinstance(palette, (dict, list)):
            map_type = VarType("categorical")
        else:
            map_type = variable_type(data, boolean_type="categorical")
        return map_type


class MarkerSemantic(DiscreteSemantic):
    """Mapping that produces values for matplotlib's marker parameter."""
    def __init__(self, shapes: DiscreteValueSpec = None, variable: str = "marker"):

        self.values = self._standardize_values(shapes)
        self.variable = variable

    def _standardize_value(self, value: MarkerPattern) -> MarkerStyle:
        """Standardize values as MarkerStyle objects."""
        return MarkerStyle(value)

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


class LineStyleSemantic(DiscreteSemantic):
    """Mapping that produces values for matplotlib's linestyle parameter."""
    def __init__(
        self,
        styles: list | dict | None = None,
        variable: str = "linestyle"
    ):
        # TODO full types
        self.values = self._standardize_values(styles)
        self.variable = variable

    def _standardize_value(self, value: str | DashPattern) -> DashPatternWithOffset:
        """Standardize values as dash pattern (with offset)."""
        return self._get_dash_pattern(value)

    def _default_values(self, n: int) -> list[DashPatternWithOffset]:
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
        dashes: list[str | DashPattern] = [
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

        return [self._get_dash_pattern(d) for d in dashes[:n]]

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

        # normalize offset to be positive and shorter than the dash cycle
        if dashes is not None:
            dsum = sum(dashes)
            if dsum:
                offset %= dsum

        return offset, dashes


# TODO or pattern?
class HatchSemantic(DiscreteSemantic):
    ...


class PointSizeSemantic(ContinuousSemantic):
    _default_range = 2, 8


class WidthSemantic(ContinuousSemantic):
    _default_range = .2, .8


# TODO or opacity?
class AlphaSemantic(ContinuousSemantic):
    _default_range = .2, 1


class LineWidthSemantic(ContinuousSemantic):
    @property
    def default_range(self) -> tuple[float, float]:
        base = mpl.rcParams["lines.linewidth"]
        return base * .5, base * 2


class EdgeWidthSemantic(ContinuousSemantic):
    @property
    def default_range(self) -> tuple[float, float]:
        # TODO use patch.linewidth or lines.markeredgewidth here?
        base = mpl.rcParams["patch.linewidth"]
        return base * .5, base * 2


# ==================================================================================== #


class SemanticMapping:
    """Stateful and callable object that maps data values to matplotlib arguments."""
    def __call__(self, x: Any) -> Any:
        raise NotImplementedError


class IdentityMapping(SemanticMapping):
    """Return input value, possibly after converting to standardized representation."""
    def __init__(self, func: Callable[[Any], Any]):
        self._standardization_func = func

    def __call__(self, x: Any) -> Any:
        return self._standardization_func(x)


class LookupMapping(SemanticMapping):
    """Discrete mapping defined by dictionary lookup."""
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, x: Any) -> Any:
        if isinstance(x, pd.Series):
            return [self.mapping.get(x_i) for x_i in x]
        else:
            return self.mapping[x]


class NormedMapping(SemanticMapping):
    """Continuous mapping defined by domain normalization and range transform."""
    def __init__(self, scale: Scale, transform: Callable[[Series], Any]):

        self.scale = scale
        self.transform = transform

    def __call__(self, x: Series | Number) -> Series | Number:

        if isinstance(x, pd.Series):
            normed = self.scale.normalize(x)
        else:
            normed = self.scale.normalize(pd.Series(x)).item()
        return self.transform(normed)


class RangeTransform:
    """Transform normed data values into float array after linear range scaling."""
    def __init__(self, out_range: tuple[float, float]):
        self.out_range = out_range

    def __call__(self, x: ArrayLike) -> ArrayLike:
        lo, hi = self.out_range
        return lo + x * (hi - lo)


class RGBTransform:
    """Transform data values into n x 3 rgb array using colormap."""
    def __init__(self, cmap: Colormap):
        self.cmap = cmap

    def __call__(self, x: ArrayLike) -> ArrayLike:
        rgba = mpl.colors.to_rgba_array(self.cmap(x))
        return rgba.squeeze()
