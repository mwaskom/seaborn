from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb

from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn.utils import get_color_cycle, remove_na
from seaborn.palettes import QUAL_PALETTES, color_palette

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import Series
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.scale import Scale
    from seaborn._core.typing import PaletteSpec


class SemanticMapping:
    """Base class for mappings between data and visual attributes."""

    levels: list  # TODO Alternately, use keys of lookup_table?

    def setup(self, data: Series, scale: Scale | None) -> SemanticMapping:
        # TODO why not just implement the GroupMapping setup() here?
        raise NotImplementedError()

    def __call__(self, x):  # TODO types; will need to overload (wheee)
        # TODO this is a hack to get things working
        # We are missing numeric maps and lots of other things
        if isinstance(x, pd.Series):
            if x.dtype.name == "category":  # TODO! possible pandas bug
                x = x.astype(object)
            # TODO where is best place to ensure that LUT values are rgba tuples?
            return np.stack(x.map(self.lookup_table).map(to_rgb))
        else:
            return to_rgb(self.lookup_table[x])


# TODO Currently, the SemanticMapping objects are also the source of the information
# about the levels/order of the semantic variables. Do we want to decouple that?

# In favor:
# Sometimes (i.e. categorical plots) we need to know x/y order, and we also need
# to know facet variable orders, so having a consistent way of defining order
# across all of the variables would be nice.

# Against:
# Our current external interface consumes both mapping parameterization like the
# color palette to use and the order information. I think this makes a fair amount
# of sense. But we could also break those, e.g. have `scale_fixed("hue", order=...)`
# similar to what we are currently developing for the x/y. Is is another method call
# which may be annoying. But then alternately it is maybe more consistent (and would
# consistently hook into whatever internal representation we'll use for variable order).
# Also, the parameters of the semantic mapping often implies a particular scale
# (i.e., providing a palette list forces categorical treatment) so it's not clear
# that it makes sense to determine that information at different points in time.


class GroupMapping(SemanticMapping):
    """Mapping that does not alter any visual properties of the artists."""
    def setup(self, data: Series, scale: Scale | None = None) -> GroupMapping:
        self.levels = categorical_order(data)
        return self


class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""

    # TODO type the important class attributes here

    def __init__(self, palette: PaletteSpec = None):

        self._input_palette = palette

    def setup(
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
        scale: Scale | None = None,  # TODO or always have a Scale?
    ) -> HueMapping:
        """Infer the type of mapping to use and define it using this vector of data."""
        palette: PaletteSpec = self._input_palette
        cmap: Colormap | None = None

        norm = None if scale is None else scale.norm
        order = None if scale is None else scale.order

        # TODO We need to add some input checks ...
        # e.g. specifying a numeric scale and a qualitative colormap should fail nicely.

        map_type = self._infer_map_type(scale, palette, data)
        assert map_type in ["numeric", "categorical", "datetime"]

        # Our goal is to end up with a dictionary mapping every unique
        # value in `data` to a color. We will also keep track of the
        # metadata about this mapping we will need for, e.g., a legend

        # --- Option 1: numeric mapping with a matplotlib colormap

        if map_type == "numeric":

            data = pd.to_numeric(data)
            levels, lookup_table, norm, cmap = self._setup_numeric(
                data, palette, norm,
            )

        # --- Option 2: categorical mapping using seaborn palette

        elif map_type == "categorical":

            levels, lookup_table = self._setup_categorical(
                data, palette, order,
            )

        # --- Option 3: datetime mapping

        elif map_type == "datetime":
            # TODO this needs actual implementation
            cmap = norm = None
            levels, lookup_table = self._setup_categorical(
                # Casting data to list to handle differences in the way
                # pandas and numpy represent datetime64 data
                list(data), palette, order,
            )

        # TODO do we need to return and assign out here or can the
        # type-specific methods do the assignment internally

        # TODO I don't love how this is kind of a mish-mash of attributes
        # Can we be more consistent across SemanticMapping subclasses?
        self.lookup_table = lookup_table
        self.palette = palette
        self.levels = levels
        self.norm = norm
        self.cmap = cmap

        return self

    def _infer_map_type(
        self,
        scale: Scale,
        palette: PaletteSpec,
        data: Series,
    ) -> VarType:
        """Determine how to implement the mapping."""
        map_type: VarType
        if scale is not None:
            return scale.type
        elif palette in QUAL_PALETTES:
            map_type = VarType("categorical")
        elif isinstance(palette, (dict, list)):
            map_type = VarType("categorical")
        else:
            map_type = variable_type(data, boolean_type="categorical")
        return map_type

    def _setup_categorical(
        self,
        data: Series,
        palette: PaletteSpec,
        order: list | None,
    ) -> tuple[list, dict]:
        """Determine colors when the hue mapping is categorical."""
        # -- Identify the order and name of the levels

        levels = categorical_order(data, order)
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            lookup_table = palette

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)  # TODO downgrade this to a warning?
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table

    def _setup_numeric(
        self,
        data: Series,
        palette: PaletteSpec,
        norm: Normalize | None,
    ) -> tuple[list, dict, Normalize | None, Colormap]:
        """Determine colors when the hue variable is quantitative."""
        cmap: Colormap
        if isinstance(palette, dict):

            # The presence of a norm object overrides a dictionary of hues
            # in specifying a numeric mapping, so we need to process it here.
            # TODO this functionality only exists to support the old relplot
            # hack for linking hue orders across facets. We don't need that any
            # more and should probably remove this, but needs deprecation.
            # (Also what should new behavior be? I think an error probably).
            levels = list(sorted(palette))
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            lookup_table = palette.copy()

        else:

            # The levels are the sorted unique values in the data
            levels = list(np.sort(remove_na(data.unique())))

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # TODO do we want to do something complicated to ensure contrast?
            palette = "ch:" if palette is None else palette

            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)

            # Now sort out the data normalization
            # TODO consolidate in ScaleWrapper so we always have a norm here?
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "`hue_norm` must be None, tuple, or Normalize object."
                raise ValueError(err)
            norm.autoscale_None(data.dropna())

            lookup_table = dict(zip(levels, cmap(norm(levels))))

        return levels, lookup_table, norm, cmap
