from __future__ import annotations
from copy import copy

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize

from seaborn._core.rules import VarType, variable_type, categorical_order
from seaborn._compat import norm_from_scale

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from pandas import Series
    from matplotlib.axis import Axis
    from matplotlib.scale import ScaleBase


class Scale:

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
        ...

    def _units_seed(self, data: Series) -> Series:

        return self.cast(data).dropna()

    def setup(self, data: Series, axis: Axis | None = None) -> Scale:

        out = copy(self)
        out.norm = copy(self.norm)
        if axis is None:
            axis = DummyAxis()
        axis.update_units(self._units_seed(data).to_numpy())
        out.axis = axis
        out.normalize(data)  # Autoscale norm if unset
        return out

    def cast(self, data: Series) -> Series:
        raise NotImplementedError()

    def convert(self, data: Series, axis: Axis | None = None) -> Series:

        if axis is None:
            axis = self.axis
        orig_array = self.cast(data).to_numpy()
        axis.update_units(orig_array)
        array = axis.convert_units(orig_array)
        return pd.Series(array, data.index, name=data.name)

    def normalize(self, data: Series) -> Series:

        array = self.convert(data).to_numpy()
        normed_array = self.norm(np.ma.masked_invalid(array))
        return pd.Series(normed_array, data.index, name=data.name)

    def forward(self, data: Series, axis: Axis | None = None) -> Series:

        transform = self.scale_obj.get_transform().transform
        array = transform(self.convert(data, axis).to_numpy())
        return pd.Series(array, data.index, name=data.name)

    def reverse(self, data: Series) -> Series:

        transform = self.scale_obj.get_transform().inverted().transform
        array = transform(data.to_numpy())
        return pd.Series(array, data.index, name=data.name)


class NumericScale(Scale):

    scale_type = VarType("numeric")

    def __init__(
        self,
        scale_obj: ScaleBase,
        norm: Normalize | tuple[float | None, float | None] | None,
    ):

        super().__init__(scale_obj, norm)
        self.dtype = float  # Any reason to make this a parameter?

    def cast(self, data: Series) -> Series:

        return data.astype(self.dtype)


class CategoricalScale(Scale):

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

    def _units_seed(self, data: Series) -> Series:

        return pd.Series(categorical_order(data, self.order)).map(self.formatter)

    def cast(self, data: Series) -> Series:

        # TODO explicit cast to string, or at least verify strings?
        # TODO string dtype or object?
        # strings = pd.Series(index=data.index, dtype="string")
        strings = pd.Series(index=data.index, dtype=object)
        strings.update(data.dropna().map(self.formatter))
        if self.order is not None:
            strings[~data.isin(self.order)] = None
        return strings

    def convert(self, data: Series, axis: Axis | None = None) -> Series:

        if axis is None:
            axis = self.axis

        # axis.update_units(self._units_seed(data).to_numpy())  TODO

        # Matplotlib "string" unit handling can't handle missing data
        strings = self.cast(data)
        mask = strings.notna().to_numpy()
        array = np.full_like(strings, np.nan, float)
        array[mask] = axis.convert_units(strings[mask].to_numpy())
        return pd.Series(array, data.index, name=data.name)


class DateTimeScale(Scale):

    scale_type = VarType("datetime")

    def __init__(
        self,
        scale_obj: ScaleBase,
        norm: Normalize | tuple[Any, Any] | None = None
    ):

        if isinstance(norm, tuple):
            norm_dates = np.array(norm, "datetime64[D]")
            norm = tuple(mpl.dates.date2num(norm_dates))

        super().__init__(scale_obj, norm)

    def cast(self, data: pd.Series) -> Series:

        if variable_type(data) == "datetime":
            return data
        elif variable_type(data) == "numeric":
            return pd.to_datetime(data, unit="D")  # TODO kwargs...
        else:
            return pd.to_datetime(data)  # TODO kwargs...


class DummyAxis:

    def __init__(self):

        self.converter = None
        self.units = None

    def set_units(self, units):

        self.units = units

    def update_units(self, x):  # TODO types

        self.converter = mpl.units.registry.get_converter(x)
        if self.converter is not None:
            self.converter.default_units(x, self)

    def convert_units(self, x):  # TODO types

        if self.converter is None:
            return x
        return self.converter.convert(x, self.units, self)


def get_default_scale(data: Series):

    axis = data.name
    scale_obj = LinearScale(axis)

    var_type = variable_type(data)
    if var_type == "numeric":
        return NumericScale(scale_obj, norm=mpl.colors.Normalize())
    elif var_type == "categorical":
        return CategoricalScale(scale_obj, order=None, formatter=format)
    elif var_type == "datetime":
        return DateTimeScale(scale_obj)
