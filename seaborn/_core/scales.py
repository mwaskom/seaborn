from __future__ import annotations
import re
from copy import copy
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import (
    Locator,
    Formatter,
    AutoLocator,
    AutoMinorLocator,
    FixedLocator,
    LinearLocator,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
    ScalarFormatter,
)
from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter,
    ConciseDateFormatter,
)
from matplotlib.axis import Axis

from seaborn._core.rules import categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Tuple, Optional, Union
    from collections.abc import Sequence
    from matplotlib.scale import ScaleBase as MatplotlibScale
    from pandas import Series
    from numpy.typing import ArrayLike
    from seaborn._core.properties import Property

    Transforms = Tuple[
        Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
    ]

    # TODO standardize String / ArrayLike interface
    Pipeline = Sequence[Optional[Callable[[Union[Series, ArrayLike]], ArrayLike]]]


class Scale:

    def __init__(
        self,
        forward_pipe: Pipeline,
        spacer: Callable[[Series], float],
        legend: tuple[list[Any], list[str]] | None,
        scale_type: str,
        matplotlib_scale: MatplotlibScale,
    ):

        self.forward_pipe = forward_pipe
        self.spacer = spacer
        self.legend = legend
        self.scale_type = scale_type
        self.matplotlib_scale = matplotlib_scale

        # TODO need to make this work
        self.order = None

    def __call__(self, data: Series) -> ArrayLike:

        return self._apply_pipeline(data, self.forward_pipe)

    # TODO def as_identity(cls):  ?

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

    def spacing(self, data: Series) -> float:
        return self.spacer(data)

    def invert_axis_transform(self, x):
        # TODO we may no longer need this method as we use the axis
        # transform directly in Plotter._unscale_coords
        finv = self.matplotlib_scale.get_transform().inverted().transform
        out = finv(x)
        if isinstance(x, pd.Series):
            return pd.Series(out, index=x.index, name=x.name)
        return out


@dataclass
class ScaleSpec:

    values: tuple | str | list | dict | None = None

    ...
    # TODO have Scale define width (/height?) ('space'?) (using data?), so e.g. nominal
    # scale sets width=1, continuous scale sets width min(diff(unique(data))), etc.

    def __post_init__(self):

        # TODO do we need anything else here?
        self.tick()
        self.format()

    def tick(self):
        # TODO what is the right base method?
        self._major_locator: Locator
        self._minor_locator: Locator
        return self

    def format(self):
        self._major_formatter: Formatter
        return self

    def setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        ...

    # TODO typing
    def _get_scale(self, name, forward, inverse):

        major_locator = self._major_locator
        minor_locator = self._minor_locator

        # TODO hack, need to add default to Continuous
        major_formatter = getattr(self, "_major_formatter", ScalarFormatter())
        # major_formatter = self._major_formatter

        class Scale(mpl.scale.FuncScale):
            def set_default_locators_and_formatters(self, axis):
                axis.set_major_locator(major_locator)
                if minor_locator is not None:
                    axis.set_minor_locator(minor_locator)
                axis.set_major_formatter(major_formatter)

        return Scale(name, (forward, inverse))


@dataclass
class Nominal(ScaleSpec):
    """
    A categorical scale without relative importance / magnitude.
    """
    # Categorical (convert to strings), un-sortable

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

            # TODO Currently just used in non-Coordinate contexts, but should
            # we use this to (A) set the padding we want for categorial plots
            # and (B) allow the values parameter for a Coordinate to set xlim/ylim
            axis.set_view_interval(0, len(units_seed) - 1)

        # TODO array cast necessary to handle float/int mixture, which we need
        # to solve in a more systematic way probably
        # (i.e. if we have [1, 2.5], do we want [1.0, 2.5]? Unclear)
        axis.update_units(stringify(np.array(units_seed)))

        # TODO define this more centrally
        def convert_units(x):
            # TODO only do this with explicit order?
            # (But also category dtype?)
            # TODO isin fails when units_seed mixes numbers and strings (numpy error?)
            # but np.isin also does not seem any faster? (Maybe not broadcasting in C)
            # keep = x.isin(units_seed)
            keep = np.array([x_ in units_seed for x_ in x], bool)
            out = np.full(len(x), np.nan)
            out[keep] = axis.convert_units(stringify(x[keep]))
            return out

        forward_pipe = [
            convert_units,
            prop.get_mapping(self, data),
            # TODO how to handle color representation consistency?
        ]

        def spacer(x):
            return 1

        if prop.legend:
            legend = units_seed, list(stringify(units_seed))
        else:
            legend = None

        scale_type = self.__class__.__name__.lower()
        scale = Scale(forward_pipe, spacer, legend, scale_type, mpl_scale)
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
class ContinuousBase(ScaleSpec):

    values: tuple | str | None = None
    norm: tuple | None = None

    def setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:

        new = copy(self)
        forward, inverse = self._get_transform()

        mpl_scale = self._get_scale(data.name, forward, inverse)

        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.update_units(data)

        mpl_scale.set_default_locators_and_formatters(axis)

        normalize: Optional[Callable[[ArrayLike], ArrayLike]]
        if prop.normed:
            if self.norm is None:
                vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = self.norm
            vmin, vmax = axis.convert_units((vmin, vmax))
            a = forward(vmin)
            b = forward(vmax) - forward(vmin)

            def normalize(x):
                return (x - a) / b

        else:
            normalize = vmin = vmax = None

        forward_pipe = [
            axis.convert_units,
            forward,
            normalize,
            prop.get_mapping(new, data)
        ]

        def spacer(x):
            return np.min(np.diff(np.sort(x.dropna().unique())))

        # TODO make legend optional on per-plot basis with ScaleSpec parameter?
        if prop.legend:
            axis.set_view_interval(vmin, vmax)
            locs = axis.major.locator()
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            labels = axis.major.formatter.format_ticks(locs)
            legend = list(locs), list(labels)

        else:
            legend = None

        scale_type = self.__class__.__name__.lower()
        return Scale(forward_pipe, spacer, legend, scale_type, mpl_scale)

    def _get_transform(self):

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


@dataclass
class Continuous(ContinuousBase):
    """
    A numeric scale supporting norms and functional transforms.
    """
    transform: str | Transforms | None = None

    # TODO Add this to deal with outliers?
    # outside: Literal["keep", "drop", "clip"] = "keep"

    # TODO maybe expose matplotlib more directly like this?
    # def using(self, scale: mpl.scale.ScaleBase) ?

    def tick(
        self,
        locator: Locator | None = None, *,
        at: Sequence[float] = None,
        upto: int | None = None,
        count: int | None = None,
        every: float | None = None,
        between: tuple[float, float] | None = None,
        minor: int | None = None,
    ) -> Continuous:  # TODO type return value as Self
        """
        Configure the selection of ticks for the scale's axis or legend.

        Parameters
        ----------
        locator: matplotlib Locator
            Pre-configured matplotlib locator; other parameters will not be used.
        at : sequence of floats
            Place ticks at these specific locations (in data units).
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.
        count : int
            Choose exactly this number of ticks, bounded by `between` or axis limits.
        every : float
            Choose locations at this interval of separation (in data units).
        between : pair of floats
            Bound upper / lower ticks when using `every` or `count`.
        minor : int
            Number of unlabeled ticks to draw between labeled "major" ticks.

        Returns
        -------
        Returns self with new tick configuration.

        """

        # TODO what about symlog?
        if isinstance(self.transform, str):
            m = re.match(r"log(\d*)", self.transform)
            log_transform = m is not None
            log_base = m[1] or 10 if m is not None else None
            forward, inverse = self._get_transform()
        else:
            log_transform = False
            log_base = forward = inverse = None

        if locator is not None:
            # TODO accept tuple for major, minor?
            if not isinstance(locator, Locator):
                err = (
                    f"Tick locator must be an instance of {Locator!r}, "
                    f"not {type(locator)!r}."
                )
                raise TypeError(err)
            major_locator = locator

        # TODO raise if locator is passed with any other parameters

        elif upto is not None:
            if log_transform:
                major_locator = LogLocator(base=log_base, numticks=upto)
            else:
                major_locator = MaxNLocator(upto, steps=[1, 1.5, 2, 2.5, 3, 5, 10])

        elif count is not None:
            if between is None:
                if log_transform:
                    msg = "`count` requires `between` with log transform."
                    raise RuntimeError(msg)
                # This is rarely useful (unless you are setting limits)
                major_locator = LinearLocator(count)
            else:
                if log_transform:
                    lo, hi = forward(between)
                    ticks = inverse(np.linspace(lo, hi, num=count))
                else:
                    ticks = np.linspace(*between, num=count)
                major_locator = FixedLocator(ticks)

        elif every is not None:
            if log_transform:
                msg = "`every` not supported with log transform."
                raise RuntimeError(msg)
            if between is None:
                major_locator = MultipleLocator(every)
            else:
                lo, hi = between
                ticks = np.arange(lo, hi + every, every)
                major_locator = FixedLocator(ticks)

        elif at is not None:
            major_locator = FixedLocator(at)

        else:
            major_locator = LogLocator(log_base) if log_transform else AutoLocator()

        if minor is None:
            minor_locator = LogLocator(log_base, subs=None) if log_transform else None
        else:
            if log_transform:
                subs = np.linspace(0, log_base, minor + 2)[1:-1]
                minor_locator = LogLocator(log_base, subs=subs)
            else:
                minor_locator = AutoMinorLocator(minor + 1)

        self._major_locator = major_locator
        self._minor_locator = minor_locator

        return self

    # TODO need to fill this out
    # def format(self, ...):


@dataclass
class Temporal(ContinuousBase):
    """
    A scale for date/time data.
    """
    # TODO date: bool?
    # For when we only care about the time component, would affect
    # default formatter and norm conversion. Should also happen in
    # Property.default_scale. The alternative was having distinct
    # Calendric / Temporal scales, but that feels a bit fussy, and it
    # would get in the way of using first-letter shorthands because
    # Calendric and Continuous would collide. Still, we haven't implemented
    # those yet, and having a clear distinction betewen date(time) / time
    # may be more useful.

    transform = None

    def tick(
        self, locator: Locator | None = None, *,
        upto: int | None = None,
    ) -> Temporal:

        if locator is not None:
            # TODO accept tuple for major, minor?
            if not isinstance(locator, Locator):
                err = (
                    f"Tick locator must be an instance of {Locator!r}, "
                    f"not {type(locator)!r}."
                )
                raise TypeError(err)
            major_locator = locator

        elif upto is not None:
            # TODO atleast for minticks?
            major_locator = AutoDateLocator(minticks=2, maxticks=upto)

        else:
            major_locator = AutoDateLocator(minticks=2, maxticks=6)

        self._major_locator = major_locator
        self._minor_locator = None

        self.format()

        return self

    def format(
        self, formater: Formatter | None = None, *,
        concise: bool = False,
    ) -> Temporal:

        # TODO ideally we would have concise coordinate ticks,
        # but full semantic ticks. Is that possible?
        if concise:
            major_formatter = ConciseDateFormatter(self._major_locator)
        else:
            major_formatter = AutoDateFormatter(self._major_locator)
        self._major_formatter = major_formatter

        return self


# ----------------------------------------------------------------------------------- #


class Calendric(ScaleSpec):
    # TODO have this separate from Temporal or have Temporal(date=True) or similar?
    ...


class Binned(ScaleSpec):
    # Needed? Or handle this at layer (in stat or as param, eg binning=)
    ...


# TODO any need for color-specific scales?
# class Sequential(Continuous):
# class Diverging(Continuous):
# class Qualitative(Nominal):


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
        self.minor = mpl.axis.Ticker()

        # It appears that this needs to be initialized this way on matplotlib 3.1,
        # but not later versions. It is unclear whether there are any issues with it.
        self._data_interval = None, None

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
        # We will probably handle that in the tick/format interface, though
        self.major.formatter = formatter
        formatter.set_axis(self)

    def set_minor_locator(self, locator):
        self.minor.locator = locator
        locator.set_axis(self)

    def set_minor_formatter(self, formatter):
        self.minor.formatter = formatter
        formatter.set_axis(self)

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
        if np.issubdtype(np.asarray(x).dtype, np.number):
            return x
        elif self.converter is None:
            return x
        return self.converter.convert(x, self.units, self)

    def get_scale(self):
        # TODO matplotlib actually returns a string here!
        # Currently we just hit it with minor ticks where it checks for
        # scale == "log". I'm not sure how you'd actually use log-scale
        # minor "ticks" in a legend context, so this is fine.....
        return self.scale

    def get_majorticklocs(self):
        return self.major.locator()


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
