from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, Union, ClassVar

import numpy as np
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
    EngFormatter,
    FuncFormatter,
    LogFormatterSciNotation,
    ScalarFormatter,
    StrMethodFormatter,
)
from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter,
    ConciseDateFormatter,
)
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series

from seaborn._core.rules import categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from seaborn._core.properties import Property
    from numpy.typing import ArrayLike

    Transforms = Tuple[
        Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
    ]

    # TODO standardize String / ArrayLike interface
    Pipeline = Sequence[Optional[Callable[[Union[Series, ArrayLike]], ArrayLike]]]


class Scale:

    values: tuple | str | list | dict | None

    _priority: ClassVar[int]
    _pipeline: Pipeline
    _matplotlib_scale: ScaleBase
    _spacer: staticmethod
    _legend: tuple[list[str], list[Any]] | None

    def __post_init__(self):

        self._tick_params = None
        self._label_params = None
        self._legend = None

    def tick(self):
        raise NotImplementedError()

    def label(self):
        raise NotImplementedError()

    def _get_locators(self):
        raise NotImplementedError()

    def _get_formatter(self):
        raise NotImplementedError()

    # TODO typing
    def _get_scale(self, name, forward, inverse):

        major_locator, minor_locator = self._get_locators(**self._tick_params)
        major_formatter = self._get_formatter(major_locator, **self._label_params)

        class InternalScale(mpl.scale.FuncScale):
            def set_default_locators_and_formatters(self, axis):
                axis.set_major_locator(major_locator)
                if minor_locator is not None:
                    axis.set_minor_locator(minor_locator)
                axis.set_major_formatter(major_formatter)

        return InternalScale(name, (forward, inverse))

    def _spacing(self, x: Series) -> float:
        return self._spacer(x)

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        raise NotImplementedError()

    def __call__(self, data: Series) -> ArrayLike:

        # TODO sometimes we need to handle scalars (e.g. for Line)
        # but what is the best way to do that?
        scalar_data = np.isscalar(data)
        if scalar_data:
            data = np.array([data])

        for func in self._pipeline:
            if func is not None:
                data = func(data)

        if scalar_data:
            data = data[0]

        return data

    @staticmethod
    def _identity():

        class Identity(Scale):
            _pipeline = []
            _spacer = None
            _legend = None
            _matplotlib_scale = None

        return Identity()


@dataclass
class Nominal(Scale):
    """
    A categorical scale without relative importance / magnitude.
    """
    # Categorical (convert to strings), un-sortable

    values: tuple | str | list | dict | None = None
    order: list | None = None

    _priority: ClassVar[int] = 3

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:

        new = copy(self)
        if new._tick_params is None:
            new = new.tick()
        if new._label_params is None:
            new = new.label()

        # TODO flexibility over format() which isn't great for numbers / dates
        stringify = np.vectorize(format)

        units_seed = categorical_order(data, new.order)

        # TODO move to new._get_scale?
        # TODO this needs some more complicated rethinking about how to pass
        # a unit dictionary down to these methods, along with how much we want
        # to invest in their API. What is it useful for tick() to do here?
        # (Ordinal may be different if we draw that contrast).
        # Any customization we do to allow, e.g., label wrapping will probably
        # require defining our own Formatter subclass.
        # We could also potentially implement auto-wrapping in an Axis subclass
        # (see Axis.draw ... it already is computing the bboxes).
        # major_locator, minor_locator = new._get_locators(**new._tick_params)
        # major_formatter = new._get_formatter(major_locator, **new._label_params)

        class CatScale(mpl.scale.LinearScale):
            # TODO turn this into a real thing I guess
            name = None  # To work around mpl<3.4 compat issues

            def set_default_locators_and_formatters(self, axis):
                ...
                # axis.set_major_locator(major_locator)
                # if minor_locator is not None:
                #     axis.set_minor_locator(minor_locator)
                # axis.set_major_formatter(major_formatter)

        mpl_scale = CatScale(data.name)
        if axis is None:
            axis = PseudoAxis(mpl_scale)

            # TODO Currently just used in non-Coordinate contexts, but should
            # we use this to (A) set the padding we want for categorial plots
            # and (B) allow the values parameter for a Coordinate to set xlim/ylim
            axis.set_view_interval(0, len(units_seed) - 1)

        new._matplotlib_scale = mpl_scale

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

        new._pipeline = [
            convert_units,
            prop.get_mapping(new, data),
            # TODO how to handle color representation consistency?
        ]

        def spacer(x):
            return 1

        new._spacer = spacer

        if prop.legend:
            new._legend = units_seed, list(stringify(units_seed))

        return new

    def tick(self, locator=None):

        new = copy(self)
        new._tick_params = {
            "locator": locator,
        }
        return new

    def label(self, formatter=None):

        new = copy(self)
        new._label_params = {
            "formatter": formatter,
        }
        return new

    def _get_locators(self, locator):

        if locator is not None:
            return locator, None

        locator = mpl.category.StrCategoryLocator({})

        return locator, None

    def _get_formatter(self, locator, formatter):

        if formatter is not None:
            return formatter

        formatter = mpl.category.StrCategoryFormatter({})

        return formatter


@dataclass
class Ordinal(Scale):
    # Categorical (convert to strings), sortable, can skip ticklabels
    ...


@dataclass
class Discrete(Scale):
    # Numeric, integral, can skip ticks/ticklabels
    ...


@dataclass
class ContinuousBase(Scale):

    values: tuple | str | None = None
    norm: tuple | None = None

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:

        new = copy(self)
        if new._tick_params is None:
            new = new.tick()
        if new._label_params is None:
            new = new.label()

        forward, inverse = new._get_transform()

        mpl_scale = new._get_scale(data.name, forward, inverse)

        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.update_units(data)

        mpl_scale.set_default_locators_and_formatters(axis)
        new._matplotlib_scale = mpl_scale

        normalize: Optional[Callable[[ArrayLike], ArrayLike]]
        if prop.normed:
            if new.norm is None:
                vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = new.norm
            vmin, vmax = axis.convert_units((vmin, vmax))
            a = forward(vmin)
            b = forward(vmax) - forward(vmin)

            def normalize(x):
                return (x - a) / b

        else:
            normalize = vmin = vmax = None

        new._pipeline = [
            axis.convert_units,
            forward,
            normalize,
            prop.get_mapping(new, data)
        ]

        def spacer(x):
            return np.min(np.diff(np.sort(x.dropna().unique())))
        new._spacer = spacer

        # TODO make legend optional on per-plot basis with Scale parameter?
        if prop.legend:
            axis.set_view_interval(vmin, vmax)
            locs = axis.major.locator()
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            labels = axis.major.formatter.format_ticks(locs)
            new._legend = list(locs), list(labels)

        return new

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
    values: tuple | str | None = None
    transform: str | Transforms | None = None

    # TODO Add this to deal with outliers?
    # outside: Literal["keep", "drop", "clip"] = "keep"

    _priority: ClassVar[int] = 1

    def tick(
        self,
        locator: Locator | None = None, *,
        at: Sequence[float] = None,
        upto: int | None = None,
        count: int | None = None,
        every: float | None = None,
        between: tuple[float, float] | None = None,
        minor: int | None = None,
    ) -> Continuous:
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
        Copy of self with new tick configuration.

        """
        # Input checks
        if locator is not None and not isinstance(locator, Locator):
            raise TypeError(
                f"Tick locator must be an instance of {Locator!r}, "
                f"not {type(locator)!r}."
            )

        # Note: duplicating some logic from _get_locators so that we can do input checks
        # in the function the user actually calls rather than from within _setup
        log_transform = (
            isinstance(self.transform, str)
            and re.match(r"log(\d*)", self.transform)
        )

        if count is not None and between is None and log_transform:
            msg = "`count` requires `between` with log transform."
            raise RuntimeError(msg)

        if every is not None and log_transform:
            msg = "`every` not supported with log transform."
            raise RuntimeError(msg)

        new = copy(self)
        new._tick_params = {
            "locator": locator,
            "at": at,
            "upto": upto,
            "count": count,
            "every": every,
            "between": between,
            "minor": minor,
        }
        return new

    def label(
        self,
        formatter=None, *,
        like: str | Callable | None = None,
        base: int | None = None,
        unit: str | None = None,
    ) -> Continuous:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        Parameters
        ----------
        formatter : matplotlib Formatter
            Pre-configured formatter to use; other parameters will be ignored.
        like : str or callable
            Either a format pattern (e.g., `".2f"`), a format string with fields named
            `x` and/or `pos` (e.g., `"${x:.2f}"`), or a callable that consumes a number
            and returns a string.
        base : number
            Use log formatter (with scientific notation) having this value as the base.
        unit : str or (str, str) tuple
            Use engineering formatter with SI units (e.g., with `unit="g"`, a tick value
            of 5000 will appear as `5 kg`). When a tuple, the first element gives the
            seperator between the number and unit.

        Returns
        -------
        Copy of self with new label configuration.

        """
        if formatter is not None and not isinstance(formatter, Formatter):
            raise TypeError(
                f"Label formatter must be an instance of {Formatter!r}, "
                f"not {type(formatter)!r}"
            )

        if like is not None and not (isinstance(like, str) or callable(like)):
            msg = f"`like` must be a string or callable, not {type(like).__name__}."
            raise TypeError(msg)

        new = copy(self)
        new._label_params = {
            "formatter": formatter,
            "like": like,
            "base": base,
            "unit": unit,
        }
        return new

    def _get_locators(self, locator, at, upto, count, every, between, minor):

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
            major_locator = locator

        # TODO raise if locator is passed with any other parameters

        elif upto is not None:
            if log_transform:
                major_locator = LogLocator(base=log_base, numticks=upto)
            else:
                major_locator = MaxNLocator(upto, steps=[1, 1.5, 2, 2.5, 3, 5, 10])

        elif count is not None:
            if between is None:
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

        return major_locator, minor_locator

    def _get_formatter(self, locator, formatter, like, base, unit):

        # TODO this has now been copied in a few places
        if base is None and isinstance(self.transform, str):
            # TODO handle symlog too
            m = re.match(r"log(\d*)", self.transform)
            base = m[1] or 10 if m is not None else None

        if formatter is not None:
            return formatter

        if like is not None:
            if isinstance(like, str):
                if "{x" in like or "{pos" in like:
                    fmt = like
                else:
                    fmt = f"{{x:{like}}}"
                formatter = StrMethodFormatter(fmt)
            else:
                formatter = FuncFormatter(like)

        elif base is not None:
            # We could add other log options if necessary
            formatter = LogFormatterSciNotation(base)

        elif unit is not None:
            if isinstance(unit, tuple):
                sep, unit = unit
            else:
                sep = " "
            formatter = EngFormatter(unit, sep=sep)

        else:
            formatter = ScalarFormatter()

        return formatter


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

    _priority: ClassVar[int] = 2

    def tick(
        self, locator: Locator | None = None, *,
        upto: int | None = None,
    ) -> Temporal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        This API is under construction and will be enhanced over time.

        Parameters
        ----------
        locator: matplotlib Locator
            Pre-configured matplotlib locator; other parameters will not be used.
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.

        Returns
        -------
        Copy of self with new tick configuration.

        """
        if locator is not None and not isinstance(locator, Locator):
            err = (
                f"Tick locator must be an instance of {Locator!r}, "
                f"not {type(locator)!r}."
            )
            raise TypeError(err)

        new = copy(self)
        new._tick_params = {"locator": locator, "upto": upto}
        return new

    def label(
        self,
        formatter: Formatter | None = None, *,
        concise: bool = False,
    ) -> Temporal:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        This API is under construction and will be enhanced over time.

        Parameters
        ----------
        formatter : matplotlib Formatter
            Pre-configured formatter to use; other parameters will be ignored.
        concise : bool
            If True, use :class:`matplotlib.dates.ConciseDateFormatter` to make
            the tick labels as compact as possible.

        Returns
        -------
        Copy of self with new label configuration.

        """
        new = copy(self)
        new._label_params = {"formatter": formatter, "concise": concise}
        return new

    def _get_locators(self, locator, upto):

        if locator is not None:
            major_locator = locator
        elif upto is not None:
            major_locator = AutoDateLocator(minticks=2, maxticks=upto)

        else:
            major_locator = AutoDateLocator(minticks=2, maxticks=6)
        minor_locator = None

        return major_locator, minor_locator

    def _get_formatter(self, locator, formatter, concise):

        if formatter is not None:
            return formatter

        if concise:
            # TODO ideally we would have concise coordinate ticks,
            # but full semantic ticks. Is that possible?
            formatter = ConciseDateFormatter(locator)
        else:
            formatter = AutoDateFormatter(locator)

        return formatter


# ----------------------------------------------------------------------------------- #


class Calendric(Scale):
    # TODO have this separate from Temporal or have Temporal(date=True) or similar?
    ...


class Binned(Scale):
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


# ------------------------------------------------------------------------------------ #
# Transform function creation


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
