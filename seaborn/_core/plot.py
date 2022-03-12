from __future__ import annotations

import io
import re
import itertools
from collections import abc
from distutils.version import LooseVersion

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt  # TODO defer import into Plot.show()

from seaborn._compat import set_scale_obj
from seaborn._core.data import PlotData
from seaborn._core.rules import categorical_order
from seaborn._core.scales import ScaleSpec
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.properties import PROPERTIES, Property

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any
    from collections.abc import Callable, Generator, Hashable
    from pandas import DataFrame, Index
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure, SubFigure
    from seaborn._marks.base import Mark
    from seaborn._stats.base import Stat
    from seaborn._core.move import Move
    from seaborn._core.typing import DataSource, VariableSpec, OrderSpec


class Plot:

    # TODO use TypedDict throughout?

    _data: PlotData
    _layers: list[dict]
    _scales: dict[str, ScaleSpec]

    _subplotspec: dict[str, Any]
    _facetspec: dict[str, Any]
    _pairspec: dict[str, Any]

    def __init__(
        self,
        # TODO rewrite with overload to clarify possible signatures?
        *args: DataSource | VariableSpec,
        data: DataSource = None,
        x: VariableSpec = None,
        y: VariableSpec = None,
        # TODO maybe enumerate variables for tab-completion/discoverability?
        # I think the main concern was being extensible ... possible to add
        # to the signature using inspect?
        **variables: VariableSpec,
    ):

        if args:
            data, x, y = self._resolve_positionals(args, data, x, y)
        if x is not None:
            variables["x"] = x
        if y is not None:
            variables["y"] = y

        self._data = PlotData(data, variables)
        self._layers = []
        self._scales = {}

        self._subplotspec = {}
        self._facetspec = {}
        self._pairspec = {}

        self._target = None

        # TODO
        self._inplace = False

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource, x: VariableSpec, y: VariableSpec,
    ) -> tuple[DataSource, VariableSpec, VariableSpec]:

        if len(args) > 3:
            err = "Plot accepts no more than 3 positional arguments (data, x, y)"
            raise TypeError(err)  # TODO PlotSpecError?
        elif len(args) == 3:
            data_, x_, y_ = args
        else:
            # TODO need some clearer way to differentiate data / vector here
            # Alternatively, could decide this is too flexible for its own good,
            # and require data to be in positional signature. I'm conflicted.
            have_data = isinstance(args[0], (abc.Mapping, pd.DataFrame))
            if len(args) == 2:
                if have_data:
                    data_, x_ = args
                    y_ = None
                else:
                    data_ = None
                    x_, y_ = args
            else:
                y_ = None
                if have_data:
                    data_ = args[0]
                    x_ = None
                else:
                    data_ = None
                    x_ = args[0]

        out = []
        for var, named, pos in zip(["data", "x", "y"], [data, x, y], [data_, x_, y_]):
            if pos is None:
                val = named
            else:
                if named is not None:
                    raise TypeError(f"`{var}` given by both name and position")
                val = pos
            out.append(val)
        data, x, y = out

        return data, x, y

    def __add__(self, other):

        # TODO restrict to Mark / Stat etc?
        raise TypeError("Sorry, this isn't ggplot! Perhaps try Plot.add?")

    def _repr_png_(self) -> tuple[bytes, dict[str, float]]:

        return self.plot()._repr_png_()

    # TODO _repr_svg_?

    def _clone(self) -> Plot:

        if self._inplace:
            return self

        new = Plot()

        # TODO any way to make sure data does not get mutated?
        new._data = self._data

        new._layers.extend(self._layers)
        new._scales.update(self._scales)

        new._subplotspec.update(self._subplotspec)
        new._facetspec.update(self._facetspec)
        new._pairspec.update(self._pairspec)

        new._target = self._target

        return new

    def inplace(self, val: bool | None = None) -> Plot:

        # TODO I am not convinced we need this

        if val is None:
            self._inplace = not self._inplace
        else:
            self._inplace = val
        return self

    def on(self, target: Axes | SubFigure | Figure) -> Plot:

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
        move: Move | None = None,
        orient: Literal["x", "y", "v", "h"] | None = None,
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:

        # TODO do a check here that mark has been initialized,
        # otherwise errors will be inscrutable

        # TODO currently it doesn't work to specify faceting for the first time in add()
        # and I think this would be too difficult. But it should not silently fail.

        # TODO decide how to allow Mark to have Stat/Move
        # if stat is None and hasattr(mark, "default_stat"):
        #     stat = mark.default_stat()

        # TODO if data is supplied it overrides the global data object
        # Another option would be to left join (layer_data, global_data)
        # after dropping the column intersection from global_data
        # (but join on what? always the index? that could get tricky...)

        new = self._clone()
        new._layers.append({
            "mark": mark,
            "stat": stat,
            "move": move,
            "source": data,
            "variables": variables,
            "orient": {"v": "x", "h": "y"}.get(orient, orient),  # type: ignore
        })

        return new

    def pair(
        self,
        x: list[Hashable] | Index[Hashable] | None = None,
        y: list[Hashable] | Index[Hashable] | None = None,
        wrap: int | None = None,
        cartesian: bool = True,  # TODO bikeshed name, maybe cross?
        # TODO other existing PairGrid things like corner?
    ) -> Plot:

        # TODO Problems to solve:
        #
        # - Unclear is how to handle the diagonal plots that PairGrid offers
        #
        # - Implementing this will require lots of downscale changes in figure setup,
        #   and especially the axis scaling, which will need to be pair specific

        # TODO lists of vectors currently work, but I'm not sure where best to test

        # TODO is is weird to call .pair() to create univariate plots?
        # i.e. Plot(data).pair(x=[...]). The basic logic is fine.
        # But maybe a different verb (e.g. Plot.spread) would be more clear?
        # Then Plot(data).pair(x=[...]) would show the given x vars vs all.

        pairspec: dict[str, Any] = {}

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
            for axis in "xy":
                if axis not in self._data:
                    pairspec[axis] = all_unused_columns
        else:

            axes = {"x": x, "y": y}
            for axis, arg in axes.items():
                if arg is not None:
                    if isinstance(arg, (str, int)):
                        err = f"You must pass a sequence of variable keys to `{axis}`"
                        raise TypeError(err)
                    pairspec[axis] = list(arg)

        pairspec["variables"] = {}
        pairspec["structure"] = {}
        for axis in "xy":
            keys = []
            for i, col in enumerate(pairspec.get(axis, [])):
                # TODO note that this assumes no variables are defined as {axis}{digit}
                # This could be a slight problem as matplotlib occasionally uses that
                # format for artists that take multiple parameters on each axis.
                # Perhaps we should set the internal pair variables to "_{axis}{index}"?
                key = f"{axis}{i}"
                keys.append(key)
                pairspec["variables"][key] = col

            if keys:
                pairspec["structure"][axis] = keys

        # TODO raise here if cartesian is False and len(x) != len(y)?
        pairspec["cartesian"] = cartesian
        pairspec["wrap"] = wrap

        new = self._clone()
        new._pairspec.update(pairspec)
        return new

    def facet(
        self,
        # TODO require kwargs?
        col: VariableSpec = None,
        row: VariableSpec = None,
        order: OrderSpec | dict[Literal["col", "row"], OrderSpec] = None,
        wrap: int | None = None,
    ) -> Plot:

        # Can't pass `None` here or it will disinherit the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        # TODO raise when wrap is specified with both col and row?

        col_order = row_order = None
        if isinstance(order, dict):
            col_order = order.get("col")
            if col_order is not None:
                col_order = list(col_order)
            row_order = order.get("row")
            if row_order is not None:
                row_order = list(row_order)
        elif order is not None:
            # TODO Allow order: list here when single facet var defined in constructor?
            # Thinking I'd rather not at this point; rather at general .order method?
            if col is not None:
                col_order = list(order)
            if row is not None:
                row_order = list(order)

        new = self._clone()
        new._facetspec.update({
            "source": None,
            "variables": variables,
            "col_order": col_order,
            "row_order": row_order,
            "wrap": wrap,
        })

        return new

    # TODO def twin()?

    def scale(self, **scales: ScaleSpec) -> Plot:

        new = self._clone()
        # TODO use update but double check it doesn't mutate parent of clone
        for var, scale in scales.items():
            new._scales[var] = scale
        return new

    def configure(
        self,
        figsize: tuple[float, float] | None = None,
        sharex: bool | Literal["row", "col"] | None = None,
        sharey: bool | Literal["row", "col"] | None = None,
    ) -> Plot:

        # TODO add an "auto" mode for figsize that roughly scales with the rcParams
        # figsize (so that works), but expands to prevent subplots from being squished
        # Also should we have height=, aspect=, exclusive with figsize? Or working
        # with figsize when only one is defined?

        new = self._clone()

        # TODO this is a hack; make a proper figure spec object
        new._figsize = figsize  # type: ignore

        subplot_keys = ["sharex", "sharey"]
        for key in subplot_keys:
            val = locals()[key]
            if val is not None:
                new._subplotspec[key] = val

        return new

    # TODO def legend (ugh)

    def theme(self) -> Plot:

        # TODO Plot-specific themes using the seaborn theming system
        # TODO should this also be where custom figure size goes?
        raise NotImplementedError
        new = self._clone()
        return new

    # TODO decorate? (or similar, for various texts) alt names: label?

    def save(self, fname, **kwargs) -> Plot:
        # TODO kws?
        self.plot().save(fname, **kwargs)
        return self

    def plot(self, pyplot=False) -> Plotter:

        # TODO if we have _target object, pyplot should be determined by whether it
        # is hooked into the pyplot state machine (how do we check?)

        plotter = Plotter(pyplot=pyplot)
        plotter._setup_data(self)
        plotter._setup_figure(self)
        plotter._setup_scales(self)

        for layer in plotter._layers:
            plotter._plot_layer(self, layer)

        # TODO should this go here?
        plotter._make_legend()  # TODO does this return?

        # TODO this should be configurable
        if not plotter._figure.get_constrained_layout():
            plotter._figure.set_tight_layout(True)

        return plotter

    def show(self, **kwargs) -> None:

        # TODO make pyplot configurable at the class level, and when not using,
        # import IPython.display and call on self to populate cell output?

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024

        self.plot(pyplot=True)
        plt.show(**kwargs)

    # TODO? Have this print a textual summary of how the plot is defined?
    # Could be nice to stick in the middle of a pipeline for debugging
    # def tell(self) -> Plot:
    #    return self


class Plotter:

    def __init__(self, pyplot=False):

        self.pyplot = pyplot
        self._legend_contents: list[
            tuple[str, str | int], list[Artist], list[str],
        ] = []

    def save(self, fname, **kwargs) -> Plotter:
        # TODO type fname as string or path; handle Path objects if matplotlib can't
        kwargs.setdefault("dpi", 96)
        self._figure.savefig(fname, **kwargs)
        return self

    def show(self, **kwargs) -> None:
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?

    # def draw?

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
        dpi = 96
        buffer = io.BytesIO()
        self._figure.savefig(buffer, dpi=dpi * 2, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        scaling = .85
        w, h = self._figure.get_size_inches()
        metadata = {"width": w * dpi * scaling, "height": h * dpi * scaling}
        return data, metadata

    def _setup_data(self, p: Plot) -> None:

        self._data = (
            p._data
            .join(
                p._facetspec.get("source"),
                p._facetspec.get("variables"),
            )
            .join(
                p._pairspec.get("source"),
                p._pairspec.get("variables"),
            )
        )

        # TODO join with mapping spec
        self._layers = []
        for layer in p._layers:
            self._layers.append({
                "data": self._data.join(layer.get("source"), layer.get("variables")),
                **layer,
            })

    def _setup_figure(self, p: Plot) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        # TODO use context manager with theme that has been set
        # TODO (maybe wrap THIS function with context manager; would be cleaner)

        self._subplots = subplots = Subplots(
            p._subplotspec, p._facetspec, p._pairspec, self._data,
        )

        # --- Figure initialization
        figure_kws = {"figsize": getattr(p, "_figsize", None)}  # TODO fix
        self._figure = subplots.init_figure(self.pyplot, figure_kws, p._target)

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
                    self._data.names.get(axis_key),
                    *[layer["data"].names.get(axis_key) for layer in self._layers],
                ]
                label = next((name for name in names if name is not None), None)
                ax.set(**{f"{axis}label": label})

                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or axis in p._pairspec and bool(p._pairspec.get("wrap"))
                    or not p._pairspec.get("cartesian", True)
                )
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = (
                    show_axis_label
                    or p._subplotspec.get(f"share{axis}") not in (
                        True, "all", {"x": "col", "y": "row"}[axis]
                    )
                )
                plt.setp(axis_obj.get_majorticklabels(), visible=show_tick_labels)
                plt.setp(axis_obj.get_minorticklabels(), visible=show_tick_labels)

            # TODO title template should be configurable
            # TODO Also we want right-side titles for row facets in most cases
            # TODO should configure() accept a title= kwarg (for single subplot plots)?
            # Let's have what we currently call "margin titles" but properly using the
            # ax.set_title interface (see my gist)
            title_parts = []
            for dim in ["row", "col"]:
                if sub[dim] is not None:
                    name = self._data.names.get(dim, f"_{dim}_")
                    title_parts.append(f"{name} = {sub[dim]}")

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and p._facetspec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)

    def _setup_scales(self, p: Plot) -> None:

        # Identify all of the variables that will be used at some point in the plot
        df = self._data.frame
        variables = list(df)
        for layer in self._layers:
            variables.extend(c for c in layer["data"].frame if c not in variables)

        # Catch cases where a variable is explicitly scaled but has no data,
        # which is *likely* to be a user error (i.e. a typo or mis-specified plot).
        # It's possible we'd want to allow the coordinate axes to be scaled without
        # data, which would let the Plot interface be used to set up an empty figure.
        # So we could revisit this if that seems useful.
        undefined = set(p._scales) - set(variables)
        if undefined:
            err = f"No data found for variable(s) with explicit scale: {undefined}"
            # TODO decide whether this is too strict. Maybe a warning?
            # raise RuntimeError(err)  # FIXME:PlotSpecError

        self._scales = {}

        for var in variables:

            # Get the data all the distinct appearances of this variable.
            var_values = pd.concat([
                df.get(var),
                # Only use variables that are *added* at the layer-level
                *(x["data"].frame.get(var)
                  for x in self._layers if var in x["variables"])
            ], axis=0, join="inner", ignore_index=True).rename(var)

            # Determine whether this is an coordinate variable
            # (i.e., x/y, paired x/y, or derivative such as xmax)
            m = re.match(r"^(?P<prefix>(?P<axis>[x|y])\d*).*", var)
            if m is None:
                axis = None
            else:
                var = m.group("prefix")
                axis = m.group("axis")

            # TODO what is the best way to allow undefined properties?
            # i.e. it is useful for extensions and non-graphical variables.
            prop = PROPERTIES.get(var if axis is None else axis, Property())

            if var in p._scales:
                arg = p._scales[var]
                if isinstance(arg, ScaleSpec):
                    scale = arg
                elif arg is None:
                    # TODO what is the cleanest way to implement identity scale?
                    # We don't really need a ScaleSpec, and Identity() will be
                    # overloaded anyway (but maybe a general Identity object
                    # that can be used as Scale/Mark/Stat/Move?)
                    self._scales[var] = Scale([], [], None, "identity", None)
                    continue
                else:
                    scale = prop.infer_scale(arg, var_values)
            else:
                scale = prop.default_scale(var_values)

            # Initialize the data-dependent parameters of the scale
            # Note that this returns a copy and does not mutate the original
            # This dictionary is used by the semantic mappings
            self._scales[var] = scale.setup(var_values, prop)

            # The mappings are always shared across subplots, but the coordinate
            # scaling can be independent (i.e. with share{x/y} = False).
            # So the coordinate scale setup is more complicated, and the rest of the
            # code is only used for coordinate scales.
            if axis is None:
                continue

            share_state = self._subplots.subplot_spec[f"share{axis}"]

            # Shared categorical axes are broken on matplotlib<3.4.0.
            # https://github.com/matplotlib/matplotlib/pull/18308
            # This only affects us when sharing *paired* axes.
            # While it would be possible to hack a workaround together,
            # this is a novel/niche behavior, so we will just raise.
            if LooseVersion(mpl.__version__) < "3.4.0":
                paired_axis = axis in p._pairspec
                cat_scale = self._scales[var].scale_type == "categorical"
                ok_dim = {"x": "col", "y": "row"}[axis]
                shared_axes = share_state not in [False, "none", ok_dim]
                if paired_axis and cat_scale and shared_axes:
                    err = "Sharing paired categorical axes requires matplotlib>=3.4.0"
                    raise RuntimeError(err)

            # Loop over every subplot and assign its scale if it's not in the axis cache
            for subplot in self._subplots:

                # This happens when Plot.pair was used
                if subplot[axis] != var:
                    continue

                axis_obj = getattr(subplot["ax"], f"{axis}axis")

                # Now we need to identify the right data rows to setup the scale with

                # The all-shared case is easiest, every subplot sees all the data
                if share_state in [True, "all"]:
                    axis_scale = scale.setup(var_values, prop, axis=axis_obj)
                    subplot[f"{axis}scale"] = axis_scale

                # Otherwise, we need to setup separate scales for different subplots
                else:
                    # Fully independent axes are easy, we use each subplot's data
                    if share_state in [False, "none"]:
                        subplot_data = self._filter_subplot_data(df, subplot)
                    # Sharing within row/col is more complicated
                    elif share_state in df:
                        subplot_data = df[df[share_state] == subplot[share_state]]
                    else:
                        subplot_data = df

                    # Same operation as above, but using the reduced dataset
                    subplot_values = var_values.loc[subplot_data.index]
                    axis_scale = scale.setup(subplot_values, prop, axis=axis_obj)
                    subplot[f"{axis}scale"] = axis_scale

                # TODO should this just happen within scale.setup?
                # Currently it is disabling the formatters that we set in scale.setup
                # The other option (using currently) is to define custom matplotlib
                # scales that don't change other axis properties
                set_scale_obj(subplot["ax"], axis, axis_scale.matplotlib_scale)

    def _plot_layer(
        self,
        p: Plot,
        layer: dict[str, Any],  # TODO layer should be a TypedDict
    ) -> None:

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        # TODO or test that value is not Coordinate? Or test /for/ something?
        grouping_properties = [v for v in PROPERTIES if v not in "xy"]

        data = layer["data"]
        mark = layer["mark"]
        stat = layer["stat"]
        move = layer["move"]

        pair_variables = p._pairspec.get("structure", {})

        # TODO should default order of properties be fixed?
        # Another option: use order they were defined in the spec?

        full_df = data.frame
        for subplots, df, scales in self._generate_pairings(full_df, pair_variables):

            orient = layer["orient"] or mark._infer_orient(scales)

            with (
                mark.use(self._scales, orient)
                # TODO this doesn't work if stat is None
                # stat.use(mappings=self._mappings, orient=orient),
            ):

                df = self._scale_coords(subplots, df)

                def get_order(var):
                    # Ignore order for x/y: they have been scaled to numeric indices,
                    # so any original order is no longer valid. Default ordering rules
                    # sorted unique numbers will correctly reconstruct intended order
                    # TODO This is tricky, make sure we add some tests for this
                    if var not in "xy" and var in scales:
                        return scales[var].order

                if stat is not None:
                    grouping_vars = grouping_properties + default_grouping_vars
                    if stat.group_by_orient:
                        grouping_vars.insert(0, orient)
                    groupby = GroupBy({var: get_order(var) for var in grouping_vars})
                    df = stat(df, groupby, orient)

                # TODO get this from the Mark, otherwise scale by natural spacing?
                # (But what about sparse categoricals? categorical always width/height=1
                # Should default width/height be 1 and then get scaled by Mark.width?
                # Also note tricky thing, width attached to mark does not get rescaled
                # during dodge, but then it dominates during feature resolution
                if "width" not in df:
                    df["width"] = 0.8
                if "height" not in df:
                    df["height"] = 0.8

                if move is not None:
                    moves = move if isinstance(move, list) else [move]
                    for move in moves:
                        move_groupers = [
                            orient,
                            *(getattr(move, "by", None) or grouping_properties),
                            *default_grouping_vars,
                        ]
                        order = {var: get_order(var) for var in move_groupers}
                        groupby = GroupBy(order)
                        df = move(df, groupby, orient)

                df = self._unscale_coords(subplots, df)

                grouping_vars = mark.grouping_vars + default_grouping_vars
                split_generator = self._setup_split_generator(
                    grouping_vars, df, subplots
                )

                mark._plot(split_generator)

        # TODO disabling while hacking on scales
        with mark.use(self._scales, None):  # TODO will we ever need orient?
            self._update_legend_contents(mark, data)

    def _scale_coords(
        self,
        subplots: list[dict],  # TODO retype with a SubplotSpec or similar
        df: DataFrame,
    ) -> DataFrame:

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .copy(deep=False)
            .drop(coord_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for subplot in subplots:
            axes_df = self._filter_subplot_data(df, subplot)[coord_cols]
            with pd.option_context("mode.use_inf_as_null", True):
                axes_df = axes_df.dropna()  # TODO do we actually need/want this?
            for var, values in axes_df.items():
                scale = subplot[f"{var[0]}scale"]
                out_df.loc[values.index, var] = scale(values)

        return out_df

    def _unscale_coords(
        self,
        subplots: list[dict],  # TODO retype with a SubplotSpec or similar
        df: DataFrame
    ) -> DataFrame:

        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", c)]
        out_df = (
            df
            .drop(coord_cols, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for subplot in subplots:
            axes_df = self._filter_subplot_data(df, subplot)[coord_cols]
            for var, values in axes_df.items():
                scale = subplot[f"{var[0]}scale"]
                out_df.loc[values.index, var] = scale.invert_transform(axes_df[var])

        return out_df

    def _generate_pairings(
        self,
        df: DataFrame,
        pair_variables: dict,
    ) -> Generator[
        # TODO type scales dict more strictly when we get rid of original Scale
        tuple[list[dict], DataFrame, dict], None, None
    ]:
        # TODO retype return with SubplotSpec or similar

        if not pair_variables:
            # TODO casting to list because subplots below is a list
            # Maybe a cleaner way to do this?
            yield list(self._subplots), df, self._scales
            return

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [None]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for sub in self._subplots:
                if (x is None or sub["x"] == x) and (y is None or sub["y"] == y):
                    subplots.append(sub)

            reassignments = {}
            for axis, prefix in zip("xy", [x, y]):
                if prefix is not None:
                    reassignments.update({
                        # Complex regex business to support e.g. x0max
                        re.sub(rf"^{prefix}(.*)$", rf"{axis}\1", col): df[col]
                        for col in df if col.startswith(prefix)
                    })

            scales = self._scales.copy()
            scales.update(
                {new: self._scales[old.name] for new, old in reassignments.items()}
            )

            yield subplots, df.assign(**reassignments), scales

    def _filter_subplot_data(
        self,
        df: DataFrame,
        subplot: dict,
    ) -> DataFrame:

        keep_rows = pd.Series(True, df.index, dtype=bool)
        for dim in ["col", "row"]:
            if dim in df:
                keep_rows &= df[dim] == subplot[dim]
        return df[keep_rows]

    def _setup_split_generator(
        self,
        grouping_vars: list[str],
        df: DataFrame,
        subplots: list[dict[str, Any]],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO will need to recreate previous categorical plots

        grouping_keys = []
        grouping_vars = [
            v for v in grouping_vars if v in df and v not in ["col", "row"]
        ]
        for var in grouping_vars:
            order = self._scales[var].order
            if order is None:
                order = categorical_order(df[var])
            grouping_keys.append(order)

        def split_generator() -> Generator:

            for subplot in subplots:

                axes_df = self._filter_subplot_data(df, subplot)

                subplot_keys = {}
                for dim in ["col", "row"]:
                    if subplot[dim] is not None:
                        subplot_keys[dim] = subplot[dim]

                if not grouping_vars or not any(grouping_keys):
                    yield subplot_keys, axes_df.copy(), subplot["ax"]
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
                    yield sub_vars, df_subset.copy(), subplot["ax"]

        return split_generator

    def _update_legend_contents(self, mark: Mark, data: PlotData) -> None:
        """Add legend artists / labels for one layer in the plot."""
        legend_vars = data.frame.columns.intersection(self._scales)

        # First pass: Identify the values that will be shown for each variable
        schema: list[tuple[
            tuple[str | None, str | int], list[str], tuple[list, list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            var_legend = self._scales[var].legend
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
                artists.append(mark._legend_artist(variables, val))
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
                existing_artists = merged_contents[key][0].copy()
                for i, artist in enumerate(existing_artists):
                    # Matplotlib accepts a tuple of artists and will overlay them
                    if isinstance(artist, tuple):
                        artist += artist[i],
                    else:
                        artist = artist, artists[i]
                    # Update list that is a value in the merged_contents dict in place
                    existing_artists[i] = artist
            else:
                merged_contents[key] = artists, labels

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title=name,  # TODO don't show "None" as title
                loc="center left",
                bbox_to_anchor=(.98, .55),
            )

            # TODO: This is an illegal hack accessing private attributes on the legend
            # We need to sort out how we are going to handle this given that lack of a
            # proper API to do things like position legends relative to each other
            if base_legend:
                base_legend._legend_box._children.extend(legend._legend_box._children)
            else:
                base_legend = legend
                self._figure.legends.append(legend)
