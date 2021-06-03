from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl

from .rules import categorical_order, variable_type
from ..utils import get_color_cycle, remove_na
from ..palettes import QUAL_PALETTES, color_palette

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Literal
    from pandas import Series
    from matplotlib.colors import Colormap, Normalize
    from .typing import PaletteSpec


class SemanticMapping:

    def __call__(self, x):  # TODO types; will need to overload (wheee)
        # TODO this is a hack to get things working
        # We are missing numeric maps and lots of other things
        if isinstance(x, pd.Series):
            if x.dtype.name == "category":  # TODO! possible pandas bug
                x = x.astype(object)
            return x.map(self.lookup_table)
        else:
            return self.lookup_table[x]


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


class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""

    # TODO type the important class attributes here

    def __init__(
        self,
        palette: Optional[PaletteSpec] = None,
        order: Optional[list] = None,
        norm: Optional[Normalize] = None,
    ):

        self._input_palette = palette
        self._input_order = order
        self._input_norm = norm

    def train(  # TODO ggplot name; let's come up with something better
        self,
        data: Series,  # TODO generally rename Series arguments to distinguish from DF?
    ) -> None:

        palette: Optional[PaletteSpec] = self._input_palette
        order: Optional[list] = self._input_order
        norm: Optional[Normalize] = self._input_norm
        cmap: Optional[Colormap] = None

        # TODO these are currently extracted from a passed in plotter instance
        # TODO can just remove if we excise wide-data handling from core
        input_format: Literal["long", "wide"] = "long"

        map_type = self._infer_map_type(data, palette, norm, input_format)

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

        else:
            raise RuntimeError()  # TODO should never get here ...

        # TODO do we need to return and assign out here or can the
        # type-specific methods do the assignment internally

        # TODO I don't love how this is kind of a mish-mash of attributes
        # Can we be more consistent across SemanticMapping subclasses?
        self.map_type = map_type
        self.lookup_table = lookup_table
        self.palette = palette
        self.levels = levels
        self.norm = norm
        self.cmap = cmap

    def _infer_map_type(
        self,
        data: Series,
        palette: Optional[PaletteSpec],
        norm: Optional[Normalize],
        input_format: Literal["long", "wide"],
    ) -> Optional[Literal["numeric", "categorical", "datetime"]]:
        """Determine how to implement the mapping."""
        map_type: Optional[Literal["numeric", "categorical", "datetime"]]
        if palette in QUAL_PALETTES:
            map_type = "categorical"
        elif norm is not None:
            map_type = "numeric"
        elif isinstance(palette, (dict, list)):  # TODO mapping/sequence?
            map_type = "categorical"
        elif input_format == "wide":
            map_type = "categorical"
        else:
            map_type = variable_type(data)

        return map_type

    def _setup_categorical(
        self,
        data: Series,
        palette: Optional[PaletteSpec],
        order: Optional[list],
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
        palette: Optional[PaletteSpec],
        norm: Optional[Normalize],
    ) -> tuple[list, dict, Optional[Normalize], Colormap]:
        """Determine colors when the hue variable is quantitative."""
        cmap: Colormap
        if isinstance(palette, dict):

            # The presence of a norm object overrides a dictionary of hues
            # in specifying a numeric mapping, so we need to process it here.
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
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = "``hue_norm`` must be None, tuple, or Normalize object."
                raise ValueError(err)

            if not norm.scaled():
                norm(np.asarray(data.dropna()))

            lookup_table = dict(zip(levels, cmap(norm(levels))))

        return levels, lookup_table, norm, cmap
