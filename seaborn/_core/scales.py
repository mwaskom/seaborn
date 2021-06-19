from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize

from seaborn._core.rules import VarType, variable_type, categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from pandas import Series
    from matplotlib.scale import ScaleBase
    from seaborn._core.typing import VariableType


class ScaleWrapper:

    def __init__(
        self,
        scale: ScaleBase,
        type: VariableType,  # TODO don't use builtin name?
        norm: tuple[float | None, float | None] | Normalize | None = None,
    ):

        transform = scale.get_transform()
        self.forward = transform.transform
        self.reverse = transform.inverted().transform

        # TODO can't we get type from the scale object in most cases?
        self.type = VarType(type)

        if norm is None:
            norm = norm_from_scale(scale, norm)
        self.norm = norm

        self._scale = scale

    @property
    def order(self):
        if hasattr(self._scale, "order"):
            return self._scale.order
        return None

    def cast(self, data):
        if hasattr(self._scale, "cast"):
            return self._scale.cast(data)
        return data


class CategoricalScale(LinearScale):

    def __init__(
        self,
        axis: str | None = None,
        order: list | None = None,
        formatter: Any = None
    ):
        # TODO what type is formatter?

        super().__init__(axis)
        self.order = order
        self.formatter = formatter

    def cast(self, data: Series) -> Series:

        order = pd.Index(categorical_order(data, self.order))
        if self.formatter is None:
            order = order.astype(str)
            data = data.astype(str)
        else:
            order = order.map(self.formatter)
            data = data.map(self.formatter)

        data = pd.Series(pd.Categorical(
            data, order.unique(), self.order is not None
        ), index=data.index)

        return data


class DatetimeScale(LinearScale):

    def __init__(self, axis: str):  # TODO norm? formatter?

        super().__init__(axis)

    def cast(self, data):

        if variable_type(data) == "numeric":
            # Use day units for consistency with matplotlib datetime handling
            # Note that pandas ends up converting everything to ns internally afterwards
            return pd.to_datetime(data, unit="D")
        else:
            return pd.to_datetime(data)


def norm_from_scale(
    scale: ScaleBase, norm: tuple[float | None, float | None] | None,
) -> Normalize:

    if isinstance(norm, Normalize):
        return norm

    if norm is None:
        vmin = vmax = None
    else:
        vmin, vmax = norm  # TODO more helpful error if this fails?

    class ScaledNorm(Normalize):

        transform: Callable

        def __call__(self, value, clip=None):
            # From github.com/matplotlib/matplotlib/blob/v3.4.2/lib/matplotlib/colors.py
            # See github.com/matplotlib/matplotlib/tree/v3.4.2/LICENSE
            value, is_scalar = self.process_value(value)
            self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            # Our changes start
            t_value = self.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
            # Our changes end
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

    new_norm = ScaledNorm(vmin, vmax)

    # TODO do this, or build the norm into the ScaleWrapper.foraward interface?
    new_norm.transform = scale.get_transform().transform  # type: ignore  # mypy #2427

    return new_norm
