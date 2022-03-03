from __future__ import annotations
from copy import copy
from dataclasses import dataclass
from functools import partial

import numpy as np
import matplotlib as mpl
from matplotlib.axis import Axis

from seaborn._core.rules import categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Tuple, List, Optional, Union
    from matplotlib.scale import ScaleBase as MatplotlibScale
    from pandas import Series
    from numpy.typing import ArrayLike
    from seaborn._core.properties import Property

    Transforms = Tuple[
        Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
    ]

    # TODO standardize String / ArrayLike interface
    Pipeline = List[Optional[Callable[[Union[Series, ArrayLike]], ArrayLike]]]


class Scale:

    def __init__(
        self,
        forward_pipe: Pipeline,
        inverse_pipe: Pipeline,
        legend: tuple[list[Any], list[str]] | None,
        scale_type: Literal["nominal", "continuous"],
        matplotlib_scale: MatplotlibScale,
    ):

        self.forward_pipe = forward_pipe
        self.inverse_pipe = inverse_pipe
        self.legend = legend
        self.scale_type = scale_type
        self.matplotlib_scale = matplotlib_scale

        # TODO need to make this work
        self.order = None

    def __call__(self, data: Series) -> ArrayLike:

        return self._apply_pipeline(data, self.forward_pipe)

    def _apply_pipeline(
        self, data: ArrayLike, pipeline: Pipeline,
    ) -> ArrayLike:

        # TODO sometimes we need to handle scalars (e.g. for Line)
        # but what is the best way to do that?
        scalar_data = np.isscalar(data)
        if scalar_data:
            data = np.array([data])

        for func in pipeline:
            if func is not None:
                data = func(data)

        if scalar_data:
            data = data[0]

        return data

    def invert_transform(self, data):
        assert self.inverse_pipe is not None  # TODO raise or no-op?
        return self._apply_pipeline(data, self.inverse_pipe)


@dataclass
class ScaleSpec:

    ...
    # TODO have Scale define width (/height?) (using data?), so e.g. nominal scale sets
    # width=1, continuous scale sets width min(diff(unique(data))), etc.

    def setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        ...


@dataclass
class Nominal(ScaleSpec):
    # Categorical (convert to strings), un-sortable

    values: str | list | dict | None = None
    order: list | None = None

    def setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:

        class CatScale(mpl.scale.LinearScale):
            # TODO turn this into a real thing I guess
            name = None  # To work around mpl<3.4 compat issues

            def set_default_locators_and_formatters(self, axis):
                pass

        # TODO flexibility over format() which isn't great for numbers / dates
        stringify = np.vectorize(format)

        units_seed = categorical_order(data, self.order)

        mpl_scale = CatScale(data.name)
        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.set_view_interval(0, len(units_seed) - 1)

        # TODO array cast necessary to handle float/int mixture, which we need
        # to solve in a more systematic way probably
        # (i.e. if we have [1, 2.5], do we want [1.0, 2.5]? Unclear)
        axis.update_units(stringify(np.array(units_seed)))

        # TODO define this more centrally
        def convert_units(x):
            # TODO only do this with explicit order?
            # (But also category dtype?)
            keep = np.isin(x, units_seed)
            out = np.full(len(x), np.nan)
            out[keep] = axis.convert_units(stringify(x[keep]))
            return out

        forward_pipe = [
            convert_units,
            prop.get_mapping(self, data),
            # TODO how to handle color representation consistency?
        ]

        inverse_pipe: Pipeline = []

        if prop.legend:
            legend = units_seed, list(stringify(units_seed))
        else:
            legend = None

        scale = Scale(forward_pipe, inverse_pipe, legend, "nominal", mpl_scale)
        return scale


@dataclass
class Ordinal(ScaleSpec):
    # Categorical (convert to strings), sortable, can skip ticklabels
    ...


@dataclass
class Discrete(ScaleSpec):
    # Numeric, integral, can skip ticks/ticklabels
    ...


@dataclass
class Continuous(ScaleSpec):

    values: tuple | str | None = None  # TODO stricter tuple typing?
    norm: tuple[float | None, float | None] | None = None
    transform: str | Transforms | None = None
    outside: Literal["keep", "drop", "clip"] = "keep"

    def tick(self, count=None, *, every=None, at=None, format=None):

        # Other ideas ... between?
        # How to minor ticks? I am fine with minor ticks never getting labels
        # so it is just a matter or specifing a) you want them and b) how many?
        # Unlike with ticks, knowing how many minor ticks in each interval suffices.
        # So I guess we just need a good parameter name?
        # Do we want to allow tick appearance parameters here?
        # What about direction? Tick on alternate axis?
        # And specific tick label values? Only allow for categorical scales?
        # Should Continuous().tick(None) mean no tick/legend? If so what should
        # default value be for count? (I guess Continuous().tick(False) would work?)
        ...

    # How to *allow* use of more complex third party objects? It seems shortsighted
    # not to maintain capabilities afforded by Scale / Ticker / Locator / UnitData,
    # despite the complexities of that API.
    # def using(self, scale: mpl.scale.ScaleBase) ?

    def setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:

        new = copy(self)
        forward, inverse = self.get_transform()

        # matplotlib_scale = mpl.scale.LinearScale(data.name)
        mpl_scale = mpl.scale.FuncScale(data.name, (forward, inverse))

        normalize: Optional[Callable[[ArrayLike], ArrayLike]]
        if prop.normed:
            if self.norm is None:
                vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = self.norm
            a = forward(vmin)
            b = forward(vmax) - forward(vmin)

            def normalize(x):
                return (x - a) / b

        else:
            normalize = vmin = vmax = None

        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.update_units(data)

        forward_pipe = [
            axis.convert_units,
            forward,
            normalize,
            prop.get_mapping(new, data)
        ]

        inverse_pipe = [inverse]

        # TODO make legend optional on per-plot basis with ScaleSpec parameter?
        if prop.legend:
            axis.set_view_interval(vmin, vmax)
            locs = axis.major.locator()
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            labels = axis.major.formatter.format_ticks(locs)
            legend = list(locs), list(labels)
        else:
            legend = None

        return Scale(forward_pipe, inverse_pipe, legend, "continuous", mpl_scale)

    def get_transform(self):

        arg = self.transform

        def get_param(method, default):
            if arg == method:
                return default
            return float(arg[len(method):])

        if arg is None:
            return _make_identity_transforms()
        elif isinstance(arg, tuple):
            return arg
        elif isinstance(arg, str):
            if arg == "ln":
                return _make_log_transforms()
            elif arg == "logit":
                base = get_param("logit", 10)
                return _make_logit_transforms(base)
            elif arg.startswith("log"):
                base = get_param("log", 10)
                return _make_log_transforms(base)
            elif arg.startswith("symlog"):
                c = get_param("symlog", 1)
                return _make_symlog_transforms(c)
            elif arg.startswith("pow"):
                exp = get_param("pow", 2)
                return _make_power_transforms(exp)
            elif arg == "sqrt":
                return _make_sqrt_transforms()
            else:
                # TODO useful error message
                raise ValueError()


# ----------------------------------------------------------------------------------- #


class Temporal(ScaleSpec):
    ...


class Calendric(ScaleSpec):
    ...


class Binned(ScaleSpec):
    # Needed? Or handle this at layer (in stat or as param, eg binning=)
    ...


# TODO any need for color-specific scales?


class Sequential(Continuous):
    ...


class Diverging(Continuous):
    ...
    # TODO alt approach is to have Continuous.center()


class Qualitative(Nominal):
    ...


# ----------------------------------------------------------------------------------- #


class PseudoAxis:
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
        self.scale = scale
        self.major = mpl.axis.Ticker()

        scale.set_default_locators_and_formatters(self)
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


# ------------------------------------------------------------------------------------


def _make_identity_transforms() -> Transforms:

    def identity(x):
        return x

    return identity, identity


def _make_logit_transforms(base: float = None) -> Transforms:

    log, exp = _make_log_transforms(base)

    def logit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return log(x) - log(1 - x)

    def expit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return exp(x) / (1 + exp(x))

    return logit, expit


def _make_log_transforms(base: float | None = None) -> Transforms:

    if base is None:
        fs = np.log, np.exp
    elif base == 2:
        fs = np.log2, partial(np.power, 2)
    elif base == 10:
        fs = np.log10, partial(np.power, 10)
    else:
        def forward(x):
            return np.log(x) / np.log(base)
        fs = forward, partial(np.power, base)

    def log(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[0](x)

    def exp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[1](x)

    return log, exp


def _make_symlog_transforms(c: float = 1, base: float = 10) -> Transforms:

    # From https://iopscience.iop.org/article/10.1088/0957-0233/24/2/027001

    # Note: currently not using base because we only get
    # one parameter from the string, and are using c (this is consistent with d3)

    log, exp = _make_log_transforms(base)

    def symlog(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

    def symexp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * c * (exp(np.abs(x)) - 1)

    return symlog, symexp


def _make_sqrt_transforms() -> Transforms:

    def sqrt(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    def square(x):
        return np.sign(x) * np.square(x)

    return sqrt, square


def _make_power_transforms(exp: float) -> Transforms:

    def forward(x):
        return np.sign(x) * np.power(np.abs(x), exp)

    def inverse(x):
        return np.sign(x) * np.power(np.abs(x), 1 / exp)

    return forward, inverse
