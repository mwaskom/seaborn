"""The classes for specifying and compiling a declarative visualization."""
from __future__ import annotations

import io
import os
import re
import sys
import inspect
import itertools
import textwrap
from collections import abc
from collections.abc import Callable, Generator, Hashable
from typing import Any

import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property, Coordinate
from seaborn._core.typing import DataSource, VariableSpec, OrderSpec
from seaborn._core.rules import categorical_order
from seaborn._compat import set_scale_obj
from seaborn.external.version import Version

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import SubFigure


if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# ---- Definitions for internal specs --------------------------------- #


class Layer(TypedDict, total=False):

    mark: Mark  # TODO allow list?
    stat: Stat | None  # TODO allow list?
    move: Move | list[Move] | None
    data: PlotData
    source: DataSource
    vars: dict[str, VariableSpec]
    orient: str
    legend: bool


class FacetSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    wrap: int | None


class PairSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    cross: bool
    wrap: int | None


# ---- The main interface for declarative plotting -------------------- #


def build_plot_signature(cls):
    """
    Decorator function for giving Plot a useful signature.

    Currently this mostly saves us some duplicated typing, but we would
    like eventually to have a way of registering new semantic properties,
    at which point dynamic signature generation would become more important.

    """
    sig = inspect.signature(cls)
    params = [
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("data", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    params.extend([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in PROPERTIES
    ])
    new_sig = sig.replace(parameters=params)
    cls.__signature__ = new_sig

    known_properties = textwrap.fill(
        ", ".join(PROPERTIES), 78, subsequent_indent=" " * 8,
    )

    if cls.__doc__ is not None:  # support python -OO mode
        cls.__doc__ = cls.__doc__.format(known_properties=known_properties)

    return cls


@build_plot_signature
class Plot:
    """
    An interface for declaratively specifying statistical graphics.

    Plots are constructed by initializing this class and adding one or more
    layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
    faceting variables or variable pairings may be defined to divide the space
    into multiple subplots. The mappings from data values to visual properties
    can be parametrized using scales, although the plot will try to infer good
    defaults when scales are not explicitly defined.

    The constructor accepts a data source (a :class:`pandas.DataFrame` or
    dictionary with columnar values) and variable assignments. Variables can be
    passed as keys to the data source or directly as data vectors.  If multiple
    data-containing objects are provided, they will be index-aligned.

    The data source and variables defined in the constructor will be used for
    all layers in the plot, unless overridden or disabled when adding a layer.

    The following variables can be defined in the constructor:
        {known_properties}

    The `data`, `x`, and `y` variables can be passed as positional arguments or
    using keywords. Whether the first positional argument is interpreted as a
    data source or `x` variable depends on its type.

    The methods of this class return a copy of the instance; use chaining to
    build up a plot through multiple calls. Methods can be called in any order.

    Most methods only add information to the plot spec; no actual processing
    happens until the plot is shown or saved. It is also possible to compile
    the plot without rendering it to access the lower-level representation.

    """
    # TODO use TypedDict throughout?

    _data: PlotData
    _layers: list[Layer]
    _scales: dict[str, Scale]

    _subplot_spec: dict[str, Any]  # TODO values type
    _facet_spec: FacetSpec
    _pair_spec: PairSpec

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        if args:
            data, variables = self._resolve_positionals(args, data, variables)

        unknown = [x for x in variables if x not in PROPERTIES]
        if unknown:
            err = f"Plot() got unexpected keyword argument(s): {', '.join(unknown)}"
            raise TypeError(err)

        self._data = PlotData(data, variables)
        self._layers = []
        self._scales = {}

        self._subplot_spec = {}
        self._facet_spec = {}
        self._pair_spec = {}

        self._target = None

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataSource, dict[str, VariableSpec]]:
        """Handle positional arguments, which may contain data / x / y."""
        if len(args) > 3:
            err = "Plot() accepts no more than 3 positional arguments (data, x, y)."
            raise TypeError(err)

        # TODO need some clearer way to differentiate data / vector here
        # (There might be an abstract DataFrame class to use here?)
        if isinstance(args[0], (abc.Mapping, pd.DataFrame)):
            if data is not None:
                raise TypeError("`data` given by both name and position.")
            data, args = args[0], args[1:]

        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = *args, None
        else:
            x = y = None

        for name, var in zip("yx", (y, x)):
            if var is not None:
                if name in variables:
                    raise TypeError(f"`{name}` given by both name and position.")
                # Keep coordinates at the front of the variables dict
                variables = {name: var, **variables}

        return data, variables

    def __add__(self, other):

        if isinstance(other, Mark) or isinstance(other, Stat):
            raise TypeError("Sorry, this isn't ggplot! Perhaps try Plot.add?")

        other_type = other.__class__.__name__
        raise TypeError(f"Unsupported operand type(s) for +: 'Plot' and '{other_type}")

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        return self.plot()._repr_png_()

    # TODO _repr_svg_?

    def _clone(self) -> Plot:
        """Generate a new object with the same information as the current spec."""
        new = Plot()

        # TODO any way to enforce that data does not get mutated?
        new._data = self._data

        new._layers.extend(self._layers)
        new._scales.update(self._scales)

        new._subplot_spec.update(self._subplot_spec)
        new._facet_spec.update(self._facet_spec)
        new._pair_spec.update(self._pair_spec)

        new._target = self._target

        return new

    @property
    def _variables(self) -> list[str]:

        variables = (
            list(self._data.frame)
            + list(self._pair_spec.get("variables", []))
            + list(self._facet_spec.get("variables", []))
        )
        for layer in self._layers:
            variables.extend(c for c in layer["vars"] if c not in variables)
        return variables

    def on(self, target: Axes | SubFigure | Figure) -> Plot:
        """
        Draw the plot into an existing Matplotlib object.

        Parameters
        ----------
        target : Axes, SubFigure, or Figure
            Matplotlib object to use. Passing :class:`matplotlib.axes.Axes` will add
            artists without otherwise modifying the figure. Otherwise, subplots will be
            created within the space of the given :class:`matplotlib.figure.Figure` or
            :class:`matplotlib.figure.SubFigure`.

        """
        # TODO alternate name: target?

        accepted_types: tuple  # Allow tuple of various length
        if hasattr(mpl.figure, "SubFigure"):  # Added in mpl 3.4
            accepted_types = (
                mpl.axes.Axes, mpl.figure.SubFigure, mpl.figure.Figure
            )
            accepted_types_str = (
                f"{mpl.axes.Axes}, {mpl.figure.SubFigure}, or {mpl.figure.Figure}"
            )
        else:
            accepted_types = mpl.axes.Axes, mpl.figure.Figure
            accepted_types_str = f"{mpl.axes.Axes} or {mpl.figure.Figure}"

        if not isinstance(target, accepted_types):
            err = (
                f"The `Plot.on` target must be an instance of {accepted_types_str}. "
                f"You passed an instance of {target.__class__} instead."
            )
            raise TypeError(err)

        new = self._clone()
        new._target = target

        return new

    def add(
        self,
        mark: Mark,
        stat: Stat | None = None,
        move: Move | list[Move] | None = None,
        *,
        orient: str | None = None,
        legend: bool = True,
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:
        """
        Define a layer of the visualization.

        This is the main method for specifying how the data should be visualized.
        It can be called multiple times with different arguments to define
        a plot with multiple layers.

        Parameters
        ----------
        mark : :class:`seaborn.objects.Mark`
            The visual representation of the data to use in this layer.
        stat : :class:`seaborn.objects.Stat`
            A transformation applied to the data before plotting.
        move : :class:`seaborn.objects.Move`
            Additional transformation(s) to handle over-plotting.
        legend : bool
            Option to suppress the mark/mappings for this layer from the legend.
        orient : "x", "y", "v", or "h"
            The orientation of the mark, which affects how the stat is computed.
            Typically corresponds to the axis that defines groups for aggregation.
            The "v" (vertical) and "h" (horizontal) options are synonyms for "x" / "y",
            but may be more intuitive with some marks. When not provided, an
            orientation will be inferred from characteristics of the data and scales.
        data : DataFrame or dict
            Data source to override the global source provided in the constructor.
        variables : data vectors or identifiers
            Additional layer-specific variables, including variables that will be
            passed directly to the stat without scaling.

        """
        if not isinstance(mark, Mark):
            msg = f"mark must be a Mark instance, not {type(mark)!r}."
            raise TypeError(msg)

        if stat is not None and not isinstance(stat, Stat):
            msg = f"stat must be a Stat instance, not {type(stat)!r}."
            raise TypeError(msg)

        # TODO decide how to allow Mark to have default Stat/Move
        # if stat is None and hasattr(mark, "default_stat"):
        #     stat = mark.default_stat()

        # TODO it doesn't work to supply scalars to variables, but that would be nice

        # TODO accept arbitrary variables defined by the stat (/move?) here
        # (but not in the Plot constructor)
        # Should stat variables ever go in the constructor, or just in the add call?

        new = self._clone()
        new._layers.append({
            "mark": mark,
            "stat": stat,
            "move": move,
            "vars": variables,
            "source": data,
            "legend": legend,
            "orient": {"v": "x", "h": "y"}.get(orient, orient),  # type: ignore
        })

        return new

    def pair(
        self,
        x: list[Hashable] | Index[Hashable] | None = None,
        y: list[Hashable] | Index[Hashable] | None = None,
        wrap: int | None = None,
        cross: bool = True,
        # TODO other existing PairGrid things like corner?
        # TODO transpose, so that e.g. multiple y axes go across the columns
    ) -> Plot:
        """
        Produce subplots with distinct `x` and/or `y` variables.

        Parameters
        ----------
        x, y : sequence(s) of data identifiers
            Variables that will define the grid of subplots.
        wrap : int
            Maximum height/width of the grid, with additional subplots "wrapped"
            on the other dimension. Requires that only one of `x` or `y` are set here.
        cross : bool
            When True, define a two-dimensional grid using the Cartesian product of `x`
            and `y`.  Otherwise, define a one-dimensional grid by pairing `x` and `y`
            entries in by position.

        """
        # TODO Problems to solve:
        #
        # - Unclear is how to handle the diagonal plots that PairGrid offers
        #
        # - Implementing this will require lots of downscale changes in figure setup,
        #   and especially the axis scaling, which will need to be pair specific

        # TODO lists of vectors currently work, but I'm not sure where best to test
        # Will need to update the signature typing to keep them

        # TODO is it weird to call .pair() to create univariate plots?
        # i.e. Plot(data).pair(x=[...]). The basic logic is fine.
        # But maybe a different verb (e.g. Plot.spread) would be more clear?
        # Then Plot(data).pair(x=[...]) would show the given x vars vs all.

        # TODO would like to add transpose=True, which would then draw
        # Plot(x=...).pair(y=[...]) across the rows
        # This may also be possible by setting `wrap=1`, although currently the axes
        # are shared and the interior labels are disabeled (this is a bug either way)

        pair_spec: PairSpec = {}

        if x is None and y is None:

            # Default to using all columns in the input source data, aside from
            # those that were assigned to a variable in the constructor
            # TODO Do we want to allow additional filtering by variable type?
            # (Possibly even default to using only numeric columns)

            if self._data.source_data is None:
                err = "You must pass `data` in the constructor to use default pairing."
                raise RuntimeError(err)

            all_unused_columns = [
                key for key in self._data.source_data
                if key not in self._data.names.values()
            ]
            if "x" not in self._data:
                x = all_unused_columns
            if "y" not in self._data:
                y = all_unused_columns

        axes = {"x": [] if x is None else x, "y": [] if y is None else y}
        for axis, arg in axes.items():
            if isinstance(arg, (str, int)):
                err = f"You must pass a sequence of variable keys to `{axis}`"
                raise TypeError(err)

        pair_spec["variables"] = {}
        pair_spec["structure"] = {}

        for axis in "xy":
            keys = []
            for i, col in enumerate(axes[axis]):
                key = f"{axis}{i}"
                keys.append(key)
                pair_spec["variables"][key] = col

            if keys:
                pair_spec["structure"][axis] = keys

        # TODO raise here if cross is False and len(x) != len(y)?
        pair_spec["cross"] = cross
        pair_spec["wrap"] = wrap

        new = self._clone()
        new._pair_spec.update(pair_spec)
        return new

    def facet(
        self,
        # TODO require kwargs?
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[str, OrderSpec] = None,
        wrap: int | None = None,
    ) -> Plot:
        """
        Produce subplots with conditional subsets of the data.

        Parameters
        ----------
        col, row : data vectors or identifiers
            Variables used to define subsets along the columns and/or rows of the grid.
            Can be references to the global data source passed in the constructor.
        order : list of strings, or dict with dimensional keys
            Define the order of the faceting variables.
        wrap : int
            Maximum height/width of the grid, with additional subplots "wrapped"
            on the other dimension. Requires that only one of `x` or `y` are set here.

        """
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        structure = {}
        if isinstance(order, dict):
            for dim in ["col", "row"]:
                dim_order = order.get(dim)
                if dim_order is not None:
                    structure[dim] = list(dim_order)
        elif order is not None:
            if col is not None and row is not None:
                err = " ".join([
                    "When faceting on both col= and row=, passing `order` as a list"
                    "is ambiguous. Use a dict with 'col' and/or 'row' keys instead."
                ])
                raise RuntimeError(err)
            elif col is not None:
                structure["col"] = list(order)
            elif row is not None:
                structure["row"] = list(order)

        spec: FacetSpec = {
            "variables": variables,
            "structure": structure,
            "wrap": wrap,
        }

        new = self._clone()
        new._facet_spec.update(spec)

        return new

    # TODO def twin()?

    def scale(self, **scales: Scale) -> Plot:
        """
        Control mappings from data units to visual properties.

        Keywords correspond to variables defined in the plot, including coordinate
        variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

        A number of "magic" arguments are accepted, including:
            - The name of a transform (e.g., `"log"`, `"sqrt"`)
            - The name of a palette (e.g., `"viridis"`, `"muted"`)
            - A tuple of values, defining the output range (e.g. `(1, 5)`)
            - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
            - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

        For more explicit control, pass a scale spec object such as :class:`Continuous`
        or :class:`Nominal`. Or use `None` to use an "identity" scale, which treats data
        values as literally encoding visual properties.

        """
        new = self._clone()
        new._scales.update(**scales)
        return new

    def configure(
        self,
        figsize: tuple[float, float] | None = None,
        sharex: bool | str | None = None,
        sharey: bool | str | None = None,
    ) -> Plot:
        """
        Control the figure size and layout.

        Parameters
        ----------
        figsize: (width, height)
            Size of the resulting figure, in inches.
        sharex, sharey : bool, "row", or "col"
            Whether axis limits should be shared across subplots. Boolean values apply
            across the entire grid, whereas `"row"` or `"col"` have a smaller scope.
            Shared axes will have tick labels disabled.

        """
        # TODO add an "auto" mode for figsize that roughly scales with the rcParams
        # figsize (so that works), but expands to prevent subplots from being squished
        # Also should we have height=, aspect=, exclusive with figsize? Or working
        # with figsize when only one is defined?

        new = self._clone()

        # TODO this is a hack; make a proper figure spec object
        new._figsize = figsize  # type: ignore

        if sharex is not None:
            new._subplot_spec["sharex"] = sharex
        if sharey is not None:
            new._subplot_spec["sharey"] = sharey

        return new

    # TODO def legend (ugh)

    def theme(self) -> Plot:
        """
        Control the default appearance of elements in the plot.

        TODO
        """
        # TODO Plot-specific themes using the seaborn theming system
        raise NotImplementedError()
        new = self._clone()
        return new

    # TODO decorate? (or similar, for various texts) alt names: label?

    def save(self, fname, **kwargs) -> Plot:
        """
        Render the plot and write it to a buffer or file on disk.

        Parameters
        ----------
        fname : str, path, or buffer
            Location on disk to save the figure, or a buffer to write into.
        Other keyword arguments are passed to :meth:`matplotlib.figure.Figure.savefig`.

        """
        # TODO expose important keyword arguments in our signature?
        self.plot().save(fname, **kwargs)
        return self

    def plot(self, pyplot=False) -> Plotter:
        """
        Compile the plot and return the :class:`Plotter` engine.

        """
        # TODO if we have _target object, pyplot should be determined by whether it
        # is hooked into the pyplot state machine (how do we check?)

        plotter = Plotter(pyplot=pyplot)

        common, layers = plotter._extract_data(self)
        plotter._setup_figure(self, common, layers)
        plotter._transform_coords(self, common, layers)

        plotter._compute_stats(self, layers)
        plotter._setup_scales(self, layers)

        # TODO Remove these after updating other methods
        # ---- Maybe have debug= param that attaches these when True?
        plotter._data = common
        plotter._layers = layers

        for layer in layers:
            plotter._plot_layer(self, layer)

        plotter._make_legend()

        # TODO this should be configurable
        if not plotter._figure.get_constrained_layout():
            plotter._figure.set_tight_layout(True)

        return plotter

    def show(self, **kwargs) -> None:
        """
        Render and display the plot.

        """
        # TODO make pyplot configurable at the class level, and when not using,
        # import IPython.display and call on self to populate cell output?

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024

        self.plot(pyplot=True).show(**kwargs)


# ---- The plot compilation engine ---------------------------------------------- #


class Plotter:
    """
    Engine for compiling a :class:`Plot` spec into a Matplotlib figure.

    This class is not intended to be instantiated directly by users.

    """
    # TODO decide if we ever want these (Plot.plot(debug=True))?
    _data: PlotData
    _layers: list[Layer]
    _figure: Figure

    def __init__(self, pyplot=False):

        self.pyplot = pyplot
        self._legend_contents: list[
            tuple[str, str | int], list[Artist], list[str],
        ] = []
        self._scales: dict[str, Scale] = {}

    def save(self, loc, **kwargs) -> Plotter:  # TODO type args
        kwargs.setdefault("dpi", 96)
        try:
            loc = os.path.expanduser(loc)
        except TypeError:
            # loc may be a buffer in which case that would not work
            pass
        self._figure.savefig(loc, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        import matplotlib.pyplot as plt
        plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        # TODO better to do this through a Jupyter hook? e.g.
        # ipy = IPython.core.formatters.get_ipython()
        # fmt = ipy.display_formatter.formatters["text/html"]
        # fmt.for_type(Plot, ...)
        # Would like to have a svg option too, not sure how to make that flexible

        # TODO use matplotlib backend directly instead of going through savefig?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.

        # TODO need to decide what the right default behavior here is:
        # - Use dpi=72 to match default InlineBackend figure size?
        # - Accept a generic "scaling" somewhere and scale DPI from that,
        #   either with 1x -> 72 or 1x -> 96 and the default scaling be .75?
        # - Listen to rcParams? InlineBackend behavior makes that so complicated :(
        # - Do we ever want to *not* use retina mode at this point?

        from PIL import Image

        dpi = 96
        buffer = io.BytesIO()
        self._figure.savefig(buffer, dpi=dpi * 2, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        scaling = .85 / 2
        # w, h = self._figure.get_size_inches()
        w, h = Image.open(buffer).size
        metadata = {"width": w * scaling, "height": h * scaling}
        return data, metadata

    def _extract_data(self, p: Plot) -> tuple[PlotData, list[Layer]]:

        common_data = (
            p._data
            .join(None, p._facet_spec.get("variables"))
            .join(None, p._pair_spec.get("variables"))
        )

        layers: list[Layer] = []
        for layer in p._layers:
            spec = layer.copy()
            spec["data"] = common_data.join(layer.get("source"), layer.get("vars"))
            layers.append(spec)

        return common_data, layers

    def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        # TODO use context manager with theme that has been set
        # TODO (maybe wrap THIS function with context manager; would be cleaner)

        subplot_spec = p._subplot_spec.copy()
        facet_spec = p._facet_spec.copy()
        pair_spec = p._pair_spec.copy()

        for dim in ["col", "row"]:
            if dim in common.frame and dim not in facet_spec["structure"]:
                order = categorical_order(common.frame[dim])
                facet_spec["structure"][dim] = order

        self._subplots = subplots = Subplots(subplot_spec, facet_spec, pair_spec)

        # --- Figure initialization
        figure_kws = {"figsize": getattr(p, "_figsize", None)}  # TODO fix
        self._figure = subplots.init_figure(
            pair_spec, self.pyplot, figure_kws, p._target,
        )

        # --- Figure annotation
        for sub in subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]
                # TODO Should we make it possible to use only one x/y label for
                # all rows/columns in a faceted plot? Maybe using sub{axis}label,
                # although the alignments of the labels from that method leaves
                # something to be desired (in terms of how it defines 'centered').
                names = [
                    common.names.get(axis_key),
                    *(layer["data"].names.get(axis_key) for layer in layers)
                ]
                label = next((name for name in names if name is not None), None)
                ax.set(**{f"{axis}label": label})

                # TODO there should be some override (in Plot.configure?) so that
                # tick labels can be shown on interior shared axes
                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or axis in p._pair_spec and bool(p._pair_spec.get("wrap"))
                    or not p._pair_spec.get("cross", True)
                )
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = (
                    show_axis_label
                    or subplot_spec.get(f"share{axis}") not in (
                        True, "all", {"x": "col", "y": "row"}[axis]
                    )
                )
                for group in ("major", "minor"):
                    for t in getattr(axis_obj, f"get_{group}ticklabels")():
                        t.set_visible(show_tick_labels)

            # TODO title template should be configurable
            # ---- Also we want right-side titles for row facets in most cases?
            # ---- Or wrapped? That can get annoying too.
            # TODO should configure() accept a title= kwarg (for single subplot plots)?
            # Let's have what we currently call "margin titles" but properly using the
            # ax.set_title interface (see my gist)
            title_parts = []
            for dim in ["row", "col"]:
                if sub[dim] is not None:
                    name = common.names.get(dim)  # TODO None = val looks bad
                    title_parts.append(f"{name} = {sub[dim]}")

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and p._facet_spec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)

    def _transform_coords(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:

        for var in p._variables:

            # Parse name to identify variable (x, y, xmin, etc.) and axis (x/y)
            # TODO should we have xmin0/xmin1 or x0min/x1min?
            m = re.match(r"^(?P<prefix>(?P<axis>[x|y])\d*).*", var)

            if m is None:
                continue

            prefix = m["prefix"]
            axis = m["axis"]

            share_state = self._subplots.subplot_spec[f"share{axis}"]

            # Concatenate layers, using only the relevant coordinate and faceting vars,
            # This is unnecessarily wasteful, as layer data will often be redundant.
            # But figuring out the minimal amount we need is more complicated.
            cols = [var, "col", "row"]
            # TODO basically copied from _setup_scales, and very clumsy
            layer_values = [common.frame.filter(cols)]
            for layer in layers:
                if layer["data"].frame is None:
                    for df in layer["data"].frames.values():
                        layer_values.append(df.filter(cols))
                else:
                    layer_values.append(layer["data"].frame.filter(cols))

            if layer_values:
                var_df = pd.concat(layer_values, ignore_index=True)
            else:
                var_df = pd.DataFrame(columns=cols)

            prop = Coordinate(axis)
            scale = self._get_scale(p, prefix, prop, var_df[var])

            # Shared categorical axes are broken on matplotlib<3.4.0.
            # https://github.com/matplotlib/matplotlib/pull/18308
            # This only affects us when sharing *paired* axes. This is a novel/niche
            # behavior, so we will raise rather than hack together a workaround.
            if Version(mpl.__version__) < Version("3.4.0"):
                from seaborn._core.scales import Nominal
                paired_axis = axis in p._pair_spec
                cat_scale = isinstance(scale, Nominal)
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            # Now loop through each subplot, deriving the relevant seed data to setup
            # the scale (so that axis units / categories are initialized properly)
            # And then scale the data in each layer.
            subplots = [view for view in self._subplots if view[axis] == prefix]

            # Setup the scale on all of the data and plug it into self._scales
            # We do this because by the time we do self._setup_scales, coordinate data
            # will have been converted to floats already, so scale inference fails
            self._scales[var] = scale._setup(var_df[var], prop)

            # Set up an empty series to receive the transformed values.
            # We need this to handle piecemeal tranforms of categories -> floats.
            transformed_data = []
            for layer in layers:
                index = layer["data"].frame.index
                transformed_data.append(pd.Series(dtype=float, index=index, name=var))

            for view in subplots:
                axis_obj = getattr(view["ax"], f"{axis}axis")

                if share_state in [True, "all"]:
                    # The all-shared case is easiest, every subplot sees all the data
                    seed_values = var_df[var]
                else:
                    # Otherwise, we need to setup separate scales for different subplots
                    if share_state in [False, "none"]:
                        # Fully independent axes are also easy: use each subplot's data
                        idx = self._get_subplot_index(var_df, view)
                    elif share_state in var_df:
                        # Sharing within row/col is more complicated
                        use_rows = var_df[share_state] == view[share_state]
                        idx = var_df.index[use_rows]
                    else:
                        # This configuration doesn't make much sense, but it's fine
                        idx = var_df.index

                    seed_values = var_df.loc[idx, var]

                scale = scale._setup(seed_values, prop, axis=axis_obj)

                for layer, new_series in zip(layers, transformed_data):
                    layer_df = layer["data"].frame
                    if var in layer_df:
                        idx = self._get_subplot_index(layer_df, view)
                        new_series.loc[idx] = scale(layer_df.loc[idx, var])

                # TODO need decision about whether to do this or modify axis transform
                set_scale_obj(view["ax"], axis, scale._matplotlib_scale)

            # Now the transformed data series are complete, set update the layer data
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer["data"].frame
                if var in layer_df:
                    layer_df[var] = new_series

    def _compute_stats(self, spec: Plot, layers: list[Layer]) -> None:

        grouping_vars = [v for v in PROPERTIES if v not in "xy"]
        grouping_vars += ["col", "row", "group"]

        pair_vars = spec._pair_spec.get("structure", {})

        for layer in layers:

            data = layer["data"]
            mark = layer["mark"]
            stat = layer["stat"]

            if stat is None:
                continue

            iter_axes = itertools.product(*[
                pair_vars.get(axis, [axis]) for axis in "xy"
            ])

            old = data.frame

            if pair_vars:
                data.frames = {}
                data.frame = data.frame.iloc[:0]  # TODO to simplify typing

            for coord_vars in iter_axes:

                pairings = "xy", coord_vars

                df = old.copy()
                scales = self._scales.copy()

                for axis, var in zip(*pairings):
                    if axis != var:
                        df = df.rename(columns={var: axis})
                        drop_cols = [x for x in df if re.match(rf"{axis}\d+", x)]
                        df = df.drop(drop_cols, axis=1)
                        scales[axis] = scales[var]

                orient = layer["orient"] or mark._infer_orient(scales)

                if stat.group_by_orient:
                    grouper = [orient, *grouping_vars]
                else:
                    grouper = grouping_vars
                groupby = GroupBy(grouper)
                res = stat(df, groupby, orient, scales)

                if pair_vars:
                    data.frames[coord_vars] = res
                else:
                    data.frame = res

    def _get_scale(
        self, spec: Plot, var: str, prop: Property, values: Series
    ) -> Scale:

        if var in spec._scales:
            arg = spec._scales[var]
            if arg is None or isinstance(arg, Scale):
                scale = arg
            else:
                scale = prop.infer_scale(arg, values)
        else:
            scale = prop.default_scale(values)

        return scale

    def _setup_scales(self, p: Plot, layers: list[Layer]) -> None:

        # Identify all of the variables that will be used at some point in the plot
        variables = set()
        for layer in layers:
            if layer["data"].frame.empty and layer["data"].frames:
                for df in layer["data"].frames.values():
                    variables.update(df.columns)
            else:
                variables.update(layer["data"].frame.columns)

        for var in variables:

            if var in self._scales:
                # Scales for coordinate variables added in _transform_coords
                continue

            # Get the data all the distinct appearances of this variable.
            parts = []
            for layer in layers:
                if layer["data"].frame.empty and layer["data"].frames:
                    for df in layer["data"].frames.values():
                        parts.append(df.get(var))
                else:
                    parts.append(layer["data"].frame.get(var))
            var_values = pd.concat(
                parts, axis=0, join="inner", ignore_index=True
            ).rename(var)

            # Determine whether this is an coordinate variable
            # (i.e., x/y, paired x/y, or derivative such as xmax)
            m = re.match(r"^(?P<prefix>(?P<axis>x|y)\d*).*", var)
            if m is None:
                axis = None
            else:
                var = m["prefix"]
                axis = m["axis"]

            prop = PROPERTIES.get(var if axis is None else axis, Property())
            scale = self._get_scale(p, var, prop, var_values)

            # Initialize the data-dependent parameters of the scale
            # Note that this returns a copy and does not mutate the original
            # This dictionary is used by the semantic mappings
            if scale is None:
                # TODO what is the cleanest way to implement identity scale?
                # We don't really need a Scale, and Identity() will be
                # overloaded anyway (but maybe a general Identity object
                # that can be used as Scale/Mark/Stat/Move?)
                # Note that this may not be the right spacer to use
                # (but that is only relevant for coordinates, where identity scale
                # doesn't make sense or is poorly defined, since we don't use pixels.)
                self._scales[var] = Scale._identity()
            else:
                scale = scale._setup(var_values, prop)
                if isinstance(prop, Coordinate):
                    # If we have a coordinate here, we didn't assign a scale for it
                    # in _transform_coords, which means it was added during compute_stat
                    # This allows downstream orientation inference to work properly.
                    # But it feels a little hacky, so perhaps revisit.
                    scale._priority = 0  # type: ignore
                self._scales[var] = scale

    def _plot_layer(self, p: Plot, layer: Layer) -> None:

        data = layer["data"]
        mark = layer["mark"]
        move = layer["move"]

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        grouping_properties = [v for v in PROPERTIES if v not in "xy"]

        pair_variables = p._pair_spec.get("structure", {})

        for subplots, df, scales in self._generate_pairings(data, pair_variables):

            orient = layer["orient"] or mark._infer_orient(scales)

            def get_order(var):
                # Ignore order for x/y: they have been scaled to numeric indices,
                # so any original order is no longer valid. Default ordering rules
                # sorted unique numbers will correctly reconstruct intended order
                # TODO This is tricky, make sure we add some tests for this
                if var not in "xy" and var in scales:
                    return getattr(scales[var], "order", None)

            if "width" in mark._mappable_props:
                width = mark._resolve(df, "width", None)
            else:
                width = df.get("width", 0.8)  # TODO what default
            if orient in df:
                df["width"] = width * scales[orient]._spacing(df[orient])

            if "baseline" in mark._mappable_props:
                # TODO what marks should have this?
                # If we can set baseline with, e.g., Bar(), then the
                # "other" (e.g. y for x oriented bars) parameterization
                # is somewhat ambiguous.
                baseline = mark._resolve(df, "baseline", None)
            else:
                # TODO unlike width, we might not want to add baseline to data
                # if the mark doesn't use it. Practically, there is a concern about
                # Mark abstraction like Area / Ribbon
                baseline = df.get("baseline", 0)
            df["baseline"] = baseline

            if move is not None:
                moves = move if isinstance(move, list) else [move]
                for move_step in moves:
                    move_by = getattr(move_step, "by", None)
                    if move_by is None:
                        move_by = grouping_properties
                    move_groupers = [*move_by, *default_grouping_vars]
                    if move_step.group_by_orient:
                        move_groupers.insert(0, orient)
                    order = {var: get_order(var) for var in move_groupers}
                    groupby = GroupBy(order)
                    df = move_step(df, groupby, orient)

            df = self._unscale_coords(subplots, df, orient)

            grouping_vars = mark._grouping_props + default_grouping_vars
            split_generator = self._setup_split_generator(
                grouping_vars, df, subplots
            )

            mark._plot(split_generator, scales, orient)

        # TODO is this the right place for this?
        for view in self._subplots:
            view["ax"].autoscale_view()

        if layer["legend"]:
            self._update_legend_contents(mark, data, scales)

    def _scale_coords(self, subplots: list[dict], df: DataFrame) -> DataFrame:
        # TODO stricter type on subplots

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .copy(deep=False)
            .drop(coord_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            with pd.option_context("mode.use_inf_as_null", True):
                axes_df = axes_df.dropna()
            for var, values in axes_df.items():
                scale = view[f"{var[0]}scale"]
                out_df.loc[values.index, var] = scale(values)

        return out_df

    def _unscale_coords(
        self, subplots: list[dict], df: DataFrame, orient: str,
    ) -> DataFrame:
        # TODO do we still have numbers in the variable name at this point?
        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        drop_cols = [*coord_cols, "width"] if "width" in df else coord_cols
        out_df = (
            df
            .drop(drop_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
            .copy(deep=False)
        )

        for view in subplots:
            view_df = self._filter_subplot_data(df, view)
            axes_df = view_df[coord_cols]
            for var, values in axes_df.items():

                axis = getattr(view["ax"], f"{var[0]}axis")
                # TODO see https://github.com/matplotlib/matplotlib/issues/22713
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, var] = inverted

                if var == orient and "width" in view_df:
                    width = view_df["width"]
                    out_df.loc[values.index, "width"] = (
                        transform(values + width / 2) - transform(values - width / 2)
                    )

        return out_df

    def _generate_pairings(
        self, data: PlotData, pair_variables: dict,
    ) -> Generator[
        tuple[list[dict], DataFrame, dict[str, Scale]], None, None
    ]:
        # TODO retype return with subplot_spec or similar

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [axis]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for view in self._subplots:
                if (view["x"] == x) and (view["y"] == y):
                    subplots.append(view)

            if data.frame.empty and data.frames:
                out_df = data.frames[(x, y)].copy()
            elif not pair_variables:
                out_df = data.frame.copy()
            else:
                if data.frame.empty and data.frames:
                    out_df = data.frames[(x, y)].copy()
                else:
                    out_df = data.frame.copy()

            scales = self._scales.copy()
            if x in out_df:
                scales["x"] = self._scales[x]
            if y in out_df:
                scales["y"] = self._scales[y]

            for axis, var in zip("xy", (x, y)):
                if axis != var:
                    out_df = out_df.rename(columns={var: axis})
                    cols = [col for col in out_df if re.match(rf"{axis}\d+", col)]
                    out_df = out_df.drop(cols, axis=1)

            yield subplots, out_df, scales

    def _get_subplot_index(self, df: DataFrame, subplot: dict) -> DataFrame:

        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df.index

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df.index[keep_rows]

    def _filter_subplot_data(self, df: DataFrame, subplot: dict) -> DataFrame:
        # TODO note redundancies with preceding function ... needs refactoring
        dims = df.columns.intersection(["col", "row"])
        if dims.empty:
            return df

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in dims:
            keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]

    def _setup_split_generator(
        self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO will need to recreate previous categorical plots

        grouping_keys = []
        grouping_vars = [
            v for v in grouping_vars if v in df and v not in ["col", "row"]
        ]
        for var in grouping_vars:
            order = getattr(self._scales[var], "order", None)
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator(keep_na=False) -> Generator:

            for view in subplots:

                axes_df = self._filter_subplot_data(df, view)

                with pd.option_context("mode.use_inf_as_null", True):
                    if keep_na:
                        # The simpler thing to do would be x.dropna().reindex(x.index).
                        # But that doesn't work with the way that the subset iteration
                        # is written below, which assumes data for grouping vars.
                        # Matplotlib (usually?) masks nan data, so this should "work".
                        # Downstream code can also drop these rows, at some speed cost.
                        present = axes_df.notna().all(axis=1)
                        axes_df = axes_df.assign(
                            x=axes_df["x"].where(present),
                            y=axes_df["y"].where(present),
                        )
                    else:
                        axes_df = axes_df.dropna()

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if view[dim] is not None:
                        subplot_keys[dim] = view[dim]

                if not grouping_vars or not any(grouping_keys):
                    yield subplot_keys, axes_df.copy(), view["ax"]
                    continue

                grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False)

                for key in itertools.product(*grouping_keys):

                    # Pandas fails with singleton tuple inputs
                    pd_key = key[0] if len(key) == 1 else key

                    try:
                        df_subset = grouped_df.get_group(pd_key)
                    except KeyError:
                        # TODO (from initial work on categorical plots refactor)
                        # We are adding this to allow backwards compatability
                        # with the empty artists that old categorical plots would
                        # add (before 0.12), which we may decide to break, in which
                        # case this option could be removed
                        df_subset = axes_df.loc[[]]

                    if df_subset.empty and not allow_empty:
                        continue

                    sub_vars = dict(zip(grouping_vars, key))
                    sub_vars.update(subplot_keys)

                    # TODO need copy(deep=...) policy (here, above, anywhere else?)
                    yield sub_vars, df_subset.copy(), view["ax"]

        return split_generator

    def _update_legend_contents(
        self, mark: Mark, data: PlotData, scales: dict[str, Scale]
    ) -> None:
        """Add legend artists / labels for one layer in the plot."""
        if data.frame.empty and data.frames:
            legend_vars = set()
            for frame in data.frames.values():
                legend_vars.update(frame.columns.intersection(scales))
        else:
            legend_vars = data.frame.columns.intersection(scales)

        # First pass: Identify the values that will be shown for each variable
        schema: list[tuple[
            tuple[str | None, str | int], list[str], tuple[list, list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            var_legend = scales[var]._legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    if data.ids[var] == part_id:
                        # Allow multiple plot semantics to represent same data variable
                        part_vars.append(var)
                        break
                else:
                    entry = (data.names[var], data.ids[var]), [var], (values, labels)
                    schema.append(entry)

        # Second pass, generate an artist corresponding to each value
        contents = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                artists.append(mark._legend_artist(variables, val, scales))
            contents.append((key, artists, labels))

        self._legend_contents.extend(contents)

    def _make_legend(self) -> None:
        """Create the legend artist(s) and add onto the figure."""
        # Combine artists representing same information across layers
        # Input list has an entry for each distinct variable in each layer
        # Output dict has an entry for each distinct variable
        merged_contents: dict[
            tuple[str | None, str | int], tuple[list[Artist], list[str]],
        ] = {}
        for key, artists, labels in self._legend_contents:
            # Key is (name, id); we need the id to resolve variable uniqueness,
            # but will need the name in the next step to title the legend
            if key in merged_contents:
                # Copy so inplace updates don't propagate back to legend_contents
                existing_artists = merged_contents[key][0]
                for i, artist in enumerate(existing_artists):
                    # Matplotlib accepts a tuple of artists and will overlay them
                    if isinstance(artist, tuple):
                        artist += artist[i],
                    else:
                        existing_artists[i] = artist, artists[i]
            else:
                merged_contents[key] = artists.copy(), labels

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title="" if name is None else name,
                loc="center left",
                bbox_to_anchor=(.98, .55),
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)
