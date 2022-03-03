"""
Classes that implements transforms for coordinate and semantic variables.

Seaborn uses a coarse typology for scales. There are four classes: numeric,
categorical, datetime, and identity. The first three correspond to the coarse
typology for variable types. Just like how numeric variables may have differnet
underlying dtypes, numeric scales may have different underlying scaling
transformations (e.g. log, sqrt). Categorical scaling handles the logic of
assigning integer indexes for (possibly) non-numeric data values. DateTime
scales handle the logic of transforming between datetime and numeric
representations, so that statistical operations can be performed on datetime
data. The identity scale shares the basic interface of the other scales, but
applies no transformations. It is useful for supporting identity mappings of
the semantic variables, where users supply literal values to be passed through
to matplotlib.

The implementation of the scaling in these classes aims to leverage matplotlib
as much as possible. That is to reduce the amount of logic that needs to be
implemented in seaborn and to keep seaborn operations in sync with what
matplotlib does where that makes sense. Therefore, in most cases seaborn
dispatches the transformations directly to a matplotlib object. This does
lead to some slightly awkward and brittle logic, especially for categorical
scales, because matplotlib does not expose much control or introspection of
the way it handles categorical (really, string-typed) variables.

Matplotlib draws a distinction between "scales" and "units", and the categorical
and datetime operations performed by the seaborn Scale objects mostly fall in
the latter category from matplotlib's perspective. Seaborn does not make this
distinction, as we think that handling categorical data falls better under the
scaling abstraction than the unit abstraction. The datetime scale feels a bit
more awkward and under-utilized, but we will perhaps further improve it in the
future, or folded into the numeric scale (the main reason to have an interface
method dealing with datetimes is to expose explicit control over tick
formatting).

The classes here, like the rest of next-gen seaborn, use a
partial-initialization pattern, where the class is initialized with
user-provided (or default) parameters, and then "setup" with data and
(optionally) a matplotlib Axis object. The setup process should not mutate
the original scale object; unlike with the Semantic classes (which produce
a different type of object when setup) scales return the type of self, but
with attributes copied to the new object.

"""
from __future__ import annotations
from copy import copy

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize
from matplotlib.axis import Axis

from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn._compat import norm_from_scale

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from pandas import Series
    from matplotlib.scale import ScaleBase


class Scale:
    """Base class for seaborn scales, implementing common transform operations."""
    axis: DummyAxis
    scale_obj: ScaleBase
    scale_type: VarType

    def __init__(
        self,
        scale_obj: ScaleBase | None,
        norm: Normalize | tuple[Any, Any] | None,
    ):

        if norm is not None and not isinstance(norm, (Normalize, tuple)):
            err = f"`norm` must be a Normalize object or tuple, not {type(norm)}"
            raise TypeError(err)

        self.scale_obj = scale_obj
        self.norm = norm_from_scale(scale_obj, norm)

        # Initialize attributes that might not be set by subclasses
        self.order: list[Any] | None = None
        self.formatter: Callable[[Any], str] | None = None
        self.type_declared: bool | None = None

    def _units_seed(self, data: Series) -> Series:
        """Representative values passed to matplotlib's update_units method."""
        return self.cast(data).dropna()

    def setup(self, data: Series, axis: Axis | None = None) -> Scale:
        """Copy self, attach to the axis, and determine data-dependent parameters."""
        out = copy(self)
        out.norm = copy(self.norm)
        if axis is None:
            axis = DummyAxis(self)
        axis.update_units(self._units_seed(data).to_numpy())
        out.axis = axis
        # Autoscale norm if unset, nulling out values that will be nulled by transform
        # (e.g., if log scale, set negative values to na so vmin is always positive)
        out.normalize(data.where(out.forward(data).notna()))
        if isinstance(axis, DummyAxis):
            # TODO This is a little awkward but I think we want to avoid doing this
            # to an actual Axis (unclear whether using Axis machinery in bits and
            # pieces is a good design, though)
            num_data = out.convert(data)
            vmin, vmax = num_data.min(), num_data.max()
            axis.set_data_interval(vmin, vmax)
            margin = .05 * (vmax - vmin)  # TODO configure?
            axis.set_view_interval(vmin - margin, vmax + margin)
        return out

    def cast(self, data: Series) -> Series:
        """Convert data type to canonical type for the scale."""
        raise NotImplementedError()

    def convert(self, data: Series, axis: Axis | None = None) -> Series:
        """Convert data type to numeric (plottable) representation, using axis."""
        if axis is None:
            axis = self.axis
        orig_array = self.cast(data).to_numpy()
        axis.update_units(orig_array)
        array = axis.convert_units(orig_array)
        return pd.Series(array, data.index, name=data.name)

    def normalize(self, data: Series) -> Series:
        """Return numeric data normalized (but not clipped) to unit scaling."""
        array = self.convert(data).to_numpy()
        normed_array = self.norm(np.ma.masked_invalid(array))
        return pd.Series(normed_array, data.index, name=data.name)

    def forward(self, data: Series, axis: Axis | None = None) -> Series:
        """Apply the transformation from the axis scale."""
        transform = self.scale_obj.get_transform().transform
        array = transform(self.convert(data, axis).to_numpy())
        return pd.Series(array, data.index, name=data.name)

    def reverse(self, data: Series) -> Series:
        """Invert and apply the transformation from the axis scale."""
        transform = self.scale_obj.get_transform().inverted().transform
        array = transform(data.to_numpy())
        return pd.Series(array, data.index, name=data.name)

    def legend(self, values: list | None = None) -> tuple[list[Any], list[str]]:

        # TODO decide how we want to allow more control over the legend
        # (e.g., how we could accept a Locator object, or specified number of ticks)
        # If we move towards a gradient legend for continuous mappings (as I'd like),
        # it will complicate the value -> label mapping that this assumes.

        # TODO also, decide whether it would be cleaner to define a more structured
        # class for the return value; the type signatures for the components of the
        # legend pipeline end up extremely complicated.

        vmin, vmax = self.axis.get_view_interval()
        if values is None:
            locs = np.array(self.axis.major.locator())
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            values = list(locs)
        else:
            locs = self.convert(pd.Series(values)).to_numpy()
        labels = list(self.axis.major.formatter.format_ticks(locs))
        return values, labels


class NumericScale(Scale):
    """Scale appropriate for numeric data; can apply mathematical transformations."""
    scale_type = VarType("numeric")

    def __init__(
        self,
        scale_obj: ScaleBase,
        norm: Normalize | tuple[float | None, float | None] | None,
    ):

        super().__init__(scale_obj, norm)
        self.dtype = float  # Any reason to make this a parameter?

    def cast(self, data: Series) -> Series:
        """Convert data type to a numeric dtype."""
        return data.astype(self.dtype)


class CategoricalScale(Scale):
    """Scale appropriate for categorical data; order and format can be controlled."""
    scale_type = VarType("categorical")

    def __init__(
        self,
        scale_obj: ScaleBase,
        order: list | None,
        formatter: Callable[[Any], str]
    ):

        super().__init__(scale_obj, None)
        self.order = order
        self.formatter = formatter
        # TODO use axis Formatter for nice batched formatting? Requires reorg

    def _units_seed(self, data: Series) -> Series:
        """Representative values passed to matplotlib's update_units method."""
        return pd.Series(categorical_order(data, self.order)).map(self.formatter)

    def cast(self, data: Series) -> Series:
        """Convert data type to canonical type for the scale."""
        # Would maybe be nice to use string type here, but conflicts with use of
        # categoricals. To avoid having multiple dtypes, stick with object for now.
        strings = pd.Series(index=data.index, dtype=object)
        strings.update(data.dropna().map(self.formatter))
        if self.order is not None:
            strings[~data.isin(self.order)] = None
        return strings

    def convert(self, data: Series, axis: Axis | None = None) -> Series:
        """
        Convert data type to numeric (plottable) representation, using axis.

        Converting categorical data to a plottable representation is tricky,
        for several reasons. Seaborn's categorical plotting functionality predates
        matplotlib's, and while they are mostly compatible, they differ in key ways.
        For instance, matplotlib's "categorical" scaling is implemented in terms of
        "string units" transformations. Additionally, matplotlib does not expose much
        control, or even introspection over the mapping from category values to
        index integers. The hardest design objective is that seaborn should be able
        to accept a matplotlib Axis that already has some categorical data plotted
        onto it and integrate the new data appropriately. Additionally, seaborn
        has independent control over category ordering, while matplotlib always
        assigns an index to a category in the order that category was encountered.

        """
        if axis is None:
            axis = self.axis

        # Matplotlib "string" unit handling can't handle missing data
        strings = self.cast(data)
        mask = strings.notna().to_numpy()
        array = np.full_like(strings, np.nan, float)
        array[mask] = axis.convert_units(strings[mask].to_numpy())
        return pd.Series(array, data.index, name=data.name)


class DateTimeScale(Scale):
    """Scale appropriate for datetimes; can be normed but not otherwise transformed."""
    scale_type = VarType("datetime")

    def __init__(
        self,
        scale_obj: ScaleBase,
        norm: Normalize | tuple[Any, Any] | None = None
    ):

        # A potential issue with this class is that we are using pd.to_datetime as the
        # canonical way of casting to date objects, but pandas uses ns resolution.
        # Matplotlib uses day resolution for dates. Thus there are cases where we could
        # fail to plot dates that matplotlib can handle.
        # Another option would be to use numpy datetime64 functionality, but pandas
        # solves a *lot* of problems with pd.to_datetime. Let's leave this as TODO.

        if isinstance(norm, tuple):
            norm = tuple(mpl.dates.date2num(self.cast(pd.Series(norm)).to_numpy()))

        # TODO should expose other kwargs for pd.to_datetime and pass through in cast()

        super().__init__(scale_obj, norm)

    def cast(self, data: Series) -> Series:
        """Convert data to a numeric representation."""
        if variable_type(data) == "datetime":
            return data
        elif variable_type(data) == "numeric":
            return pd.to_datetime(data, unit="D")
        else:
            return pd.to_datetime(data)


class IdentityScale(Scale):
    """Scale where all transformations are defined as identity mappings."""
    def __init__(self):
        super().__init__(None, None)

    def setup(self, data: Series, axis: Axis | None = None) -> Scale:
        return self

    def cast(self, data: Series) -> Series:
        """Return input data."""
        return data

    def normalize(self, data: Series) -> Series:
        """Return input data."""
        return data

    def convert(self, data: Series, axis: Axis | None = None) -> Series:
        """Return input data."""
        return data

    def forward(self, data: Series, axis: Axis | None = None) -> Series:
        """Return input data."""
        return data

    def reverse(self, data: Series) -> Series:
        """Return input data."""
        return data


class DummyAxis:
    """
    Internal class implementing minimal interface equivalent to matplotlib Axis.

    Coordinate variables are typically scaled by attaching the Axis object from
    the figure where the plot will end up. Matplotlib has no similar concept of
    and axis for the other mappable variables (color, etc.), but to simplify the
    code, this object acts like an Axis and can be used to scale other variables.

    """
    axis_name = ""  # TODO Needs real value? Just used for x/y logic in matplotlib

    def __init__(self, scale):

        self.converter = None
        self.units = None
        self.major = mpl.axis.Ticker()
        self.scale = scale

        scale.scale_obj.set_default_locators_and_formatters(self)
        # self.set_default_intervals()  TODO mock?

    def set_view_interval(self, vmin, vmax):
        # TODO this gets called when setting DateTime units,
        # but we may not need it to do anything
        self._view_interval = vmin, vmax

    def get_view_interval(self):
        return self._view_interval

    # TODO do we want to distinguish view/data intervals? e.g. for a legend
    # we probably want to represent the full range of the data values, but
    # still norm the colormap. If so, we'll need to track data range separately
    # from the norm, which we currently don't do.

    def set_data_interval(self, vmin, vmax):
        self._data_interval = vmin, vmax

    def get_data_interval(self):
        return self._data_interval

    def get_tick_space(self):
        # TODO how to do this in a configurable / auto way?
        # Would be cool to have legend density adapt to figure size, etc.
        return 5

    def set_major_locator(self, locator):
        self.major.locator = locator
        locator.set_axis(self)

    def set_major_formatter(self, formatter):
        # TODO matplotlib method does more handling (e.g. to set w/format str)
        self.major.formatter = formatter
        formatter.set_axis(self)

    def set_minor_locator(self, locator):
        pass

    def set_minor_formatter(self, formatter):
        pass

    def set_units(self, units):
        self.units = units

    def update_units(self, x):
        """Pass units to the internal converter, potentially updating its mapping."""
        self.converter = mpl.units.registry.get_converter(x)
        if self.converter is not None:
            self.converter.default_units(x, self)

            info = self.converter.axisinfo(self.units, self)

            if info is None:
                return
            if info.majloc is not None:
                # TODO matplotlib method has more conditions here; are they needed?
                self.set_major_locator(info.majloc)
            if info.majfmt is not None:
                self.set_major_formatter(info.majfmt)

            # TODO this is in matplotlib method; do we need this?
            # self.set_default_intervals()

    def convert_units(self, x):
        """Return a numeric representation of the input data."""
        if self.converter is None:
            return x
        return self.converter.convert(x, self.units, self)


def get_default_scale(data: Series) -> Scale:
    """Return an initialized scale of appropriate type for data."""
    axis = data.name
    scale_obj = LinearScale(axis)

    var_type = variable_type(data)
    if var_type == "numeric":
        return NumericScale(scale_obj, norm=mpl.colors.Normalize())
    elif var_type == "categorical":
        return CategoricalScale(scale_obj, order=None, formatter=format)
    elif var_type == "datetime":
        return DateTimeScale(scale_obj)
    else:
        # Can't really get here given seaborn logic, but avoid mypy complaints
        raise ValueError("Unknown variable type")
