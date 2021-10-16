from __future__ import annotations

import re
import io
import itertools
from copy import deepcopy
from distutils.version import LooseVersion

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt  # TODO defer import into Plot.show()

from seaborn._core.rules import categorical_order
from seaborn._core.data import PlotData
from seaborn._core.subplots import Subplots
from seaborn._core.mappings import (
    ColorSemantic,
    BooleanSemantic,
    MarkerSemantic,
    LineStyleSemantic,
    LineWidthSemantic,
)
from seaborn._core.scales import (
    ScaleWrapper,
    CategoricalScale,
    DatetimeScale,
    norm_from_scale
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any
    from collections.abc import Callable, Generator, Iterable, Hashable
    from pandas import DataFrame, Series, Index
    from matplotlib.axes import Axes
    from matplotlib.color import Normalize
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.scale import ScaleBase
    from seaborn._core.mappings import Semantic, SemanticMapping
    from seaborn._marks.base import Mark
    from seaborn._stats.base import Stat
    from seaborn._core.typing import (
        DataSource,
        PaletteSpec,
        VariableSpec,
        OrderSpec,
        NormSpec,
    )


SEMANTICS = {  # TODO should this be pluggable?
    "color": ColorSemantic(),
    "facecolor": ColorSemantic(variable="facecolor"),
    "edgecolor": ColorSemantic(variable="edgecolor"),
    "marker": MarkerSemantic(),
    "linestyle": LineStyleSemantic(),
    "fill": BooleanSemantic(variable="fill"),
    "linewidth": LineWidthSemantic(),
}


class Plot:

    _data: PlotData
    _layers: list[Layer]
    _semantics: dict[str, Semantic]
    _mappings: dict[str, SemanticMapping]  # TODO keys as Literal, or use TypedDict?
    _scales: dict[str, ScaleWrapper]

    # TODO use TypedDict here
    _subplotspec: dict[str, Any]
    _facetspec: dict[str, Any]
    _pairspec: dict[str, Any]

    _figure: Figure

    def __init__(
        self,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        self._data = PlotData(data, variables)
        self._layers = []

        self._scales = {}
        self._semantics = {}

        self._subplotspec = {}
        self._facetspec = {}
        self._pairspec = {}

        self._target = None

    def on(self, target: Axes | SubFigure | Figure) -> Plot:

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
                f"You passed an object of class {target.__class__} instead."
            )
            raise TypeError(err)

        self._target = target

        return self

    def add(
        self,
        mark: Mark,
        stat: Stat | None = None,
        orient: Literal["x", "y", "v", "h"] = "x",  # TODO "auto" as defined by Mark?
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:

        # TODO FIXME:layer change the layer object to a simple dictionary,
        # there's almost no logic in the class and it will make copy/update less awkward

        # TODO do a check here that mark has been initialized,
        # otherwise errors will be inscrutable

        if stat is None and mark.default_stat is not None:
            # TODO We need some way to say "do no stat transformation" that is different
            # from "use the default". That's basically an IdentityStat.

            # Default stat needs to be initialized here so that its state is
            # not modified across multiple plots. If a Mark wants to define a default
            # stat with non-default params, it should use functools.partial
            stat = mark.default_stat()

        orient_map = {"v": "x", "h": "y"}
        orient = orient_map.get(orient, orient)  # type: ignore  # mypy false positive?
        mark.orient = orient  # type: ignore  # mypy false positive?
        if stat is not None:
            stat.orient = orient  # type: ignore  # mypy false positive?

        self._layers.append(Layer(mark, stat, data, variables))

        return self

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

        # TODO add data kwarg here? (it's everywhere else...)

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

            if self._data._source_data is None:
                err = "You must pass `data` in the constructor to use default pairing."
                raise RuntimeError(err)

            all_unused_columns = [
                key for key in self._data._source_data
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

        self._pairspec.update(pairspec)
        return self

    def facet(
        self,
        # TODO require kwargs?
        col: VariableSpec = None,
        row: VariableSpec = None,
        col_order: OrderSpec = None,  # TODO single order param
        row_order: OrderSpec = None,
        wrap: int | None = None,
        data: DataSource = None,
    ) -> Plot:

        # Can't pass `None` here or it will disinherit the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row

        # TODO Alternately use the following parameterization for order
        # `order: list[Hashable] | dict[Literal['col', 'row'], list[Hashable]]
        # this is more convenient for the (dominant?) case where there is one
        # faceting variable

        self._facetspec.update({
            "source": data,
            "variables": variables,
            "col_order": None if col_order is None else list(col_order),
            "row_order": None if row_order is None else list(row_order),
            "wrap": wrap,
        })

        return self

    def map_color(
        self,
        # TODO accept variable specification here?
        palette: PaletteSpec = None,
        order: OrderSpec = None,
        norm: NormSpec = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # If we do ... maybe we don't even need to write these methods, but can
        # instead programatically add them based on central dict of mapping objects.
        # ALSO TODO should these be initialized with defaults?
        # TODO if we define default semantics, we can use that
        # for initialization and make this more abstract (assuming kwargs match?)
        self._semantics["color"] = ColorSemantic(palette)
        if order is not None:
            self.scale_categorical("color", order=order)
        elif norm is not None:
            self.scale_numeric("color", norm=norm)
        return self

    def map_facecolor(
        self,
        palette: PaletteSpec = None,
        order: OrderSpec = None,
        norm: NormSpec = None,
    ) -> Plot:

        self._semantics["facecolor"] = ColorSemantic(palette, variable="facecolor")
        if order is not None:
            self.scale_categorical("facecolor", order=order)
        elif norm is not None:
            self.scale_numeric("facecolor", norm=norm)
        return self

    def map_edgecolor(
        self,
        palette: PaletteSpec = None,
        order: OrderSpec = None,
        norm: NormSpec = None,
    ) -> Plot:

        self._semantics["edgecolor"] = ColorSemantic(palette, variable="edgecolor")
        if order is not None:
            self.scale_categorical("edgecolor", order=order)
        elif norm is not None:
            self.scale_numeric("edgecolor", norm=norm)
        return self

    def map_fill(
        self,
        values: list | dict | None = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["fill"] = BooleanSemantic(values, variable="fill")
        if order is not None:
            self.scale_categorical("fill", order=order)
        return self

    def map_marker(
        self,
        shapes: list | dict | None = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["marker"] = MarkerSemantic(shapes, variable="marker")
        if order is not None:
            self.scale_categorical("marker", order=order)
        return self

    def map_linestyle(
        self,
        styles: list | dict | None = None,
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["linestyle"] = LineStyleSemantic(styles, variable="linestyle")
        if order is not None:
            self.scale_categorical("linestyle", order=order)
        return self

    def map_linewidth(
        self,
        values: tuple[float, float] | list[float] | dict[Any, float] | None = None,
        norm: Normalize | None = None,
        # TODO clip?
        order: OrderSpec = None,
    ) -> Plot:

        self._semantics["linewidth"] = LineWidthSemantic(values, variable="linewidth")
        if order is not None:
            self.scale_categorical("linewidth", order=order)
        elif norm is not None:
            self.scale_numeric("linewidth", norm=norm)
        return self

    # TODO have map_gradient?
    # This could be used to add another color-like dimension
    # and also the basis for what mappings like stat.density -> rgba do

    # TODO originally we had planned to have a scale_native option that would default
    # to matplotlib. I don't fully remember why. Is this still something we need?

    def scale_numeric(
        self,
        var: str,
        scale: str | ScaleBase = "linear",
        norm: NormSpec = None,
        # TODO add clip? Useful for e.g., making sure lines don't get too thick.
        # (If we add clip, should we make the legend say like ``> value`)?
        **kwargs
    ) -> Plot:

        # TODO XXX FIXME matplotlib scales sometimes default to
        # filling invalid outputs with large out of scale numbers
        # (e.g. default behavior for LogScale is 0 -> -10000)
        # This will cause MAJOR PROBLEMS for statistical transformations
        # Solution? I think it's fine to special-case scale="log" in
        # Plot.scale_numeric and force `nonpositive="mask"` and remove
        # NAs after scaling (cf GH2454).
        # And then add a warning in the docstring that the users must
        # ensure that ScaleBase derivatives mask out of bounds data

        # TODO use norm for setting axis limits? Or otherwise share an interface?

        # TODO or separate norm as a Normalize object and limits as a tuple?
        # (If we have one we can create the other)

        # TODO expose parameter for internal dtype achieved during scale.cast?

        # TODO we want to be able to call this on numbers-as-strings data and
        # have it work the way you would expect.

        if isinstance(scale, str):
            # Matplotlib scales require an Axis object for backwards compatability,
            # but it is not used, aside from extraction of the axis_name in LogScale.
            # This can be removed when the minimum matplotlib is raised to 3.4,
            # and a simple string (`var`) can be passed.
            class Axis:
                axis_name = var
            scale = mpl.scale.scale_factory(scale, Axis(), **kwargs)

        if norm is None:
            # TODO what about when we want to infer the scale from the norm?
            # e.g. currently you pass LogNorm to get a log normalization...
            norm = norm_from_scale(scale, norm)
        self._scales[var] = ScaleWrapper(scale, "numeric", norm)
        return self

    def scale_categorical(
        self,
        var: str,
        order: Series | Index | Iterable | None = None,
        formatter: Callable | None = None,
    ) -> Plot:

        # TODO how to set limits/margins "nicely"? (i.e. 0.5 data units, past extremes)
        # TODO similarly, should this modify grid state like current categorical plots?
        # TODO "smart"/data-dependant ordering (e.g. order by median of y variable)

        if order is not None:
            order = list(order)

        scale = CategoricalScale(var, order, formatter)
        self._scales[var] = ScaleWrapper(scale, "categorical")
        return self

    def scale_datetime(self, var) -> Plot:

        scale = DatetimeScale(var)
        self._scales[var] = ScaleWrapper(scale, "datetime")

        # TODO what else should this do?
        # We should pass kwargs to the DateTime cast probably.
        # Should we also explicitly expose more of the pd.to_datetime interface?

        # It will be nice to have more control over the formatting of the ticks
        # which is pretty annoying in standard matplotlib.

        # Should datetime data ever have anything other than a linear scale?
        # The only thing I can really think of are geologic/astro plots that
        # use a reverse log scale, (but those are usually in units of years).

        return self

    def scale_identity(self, var) -> Plot:

        raise NotImplementedError("TODO")

    def theme(self) -> Plot:

        # TODO Plot-specific themes using the seaborn theming system
        # TODO should this also be where custom figure size goes?
        raise NotImplementedError
        return self

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

        # TODO figsize has no actual effect here

        subplot_keys = ["sharex", "sharey"]
        for key in subplot_keys:
            val = locals()[key]
            if val is not None:
                self._subplotspec[key] = val

        return self

    def resize(self, val):

        # TODO I don't think this is the interface we ultimately want to use, but
        # I want to be able to do this for demonstration now. If we do want this
        # could think about how to have "auto" sizing based on number of subplots
        self._figsize = val
        return self

    def plot(self, pyplot=False) -> Plot:

        self._setup_data()
        self._setup_scales()
        self._setup_mappings()
        self._setup_figure(pyplot)

        for layer in self._layers:
            layer_mappings = {k: v for k, v in self._mappings.items() if k in layer}
            self._plot_layer(layer, layer_mappings)

        # TODO this should be configurable
        if not self._figure.get_constrained_layout():
            self._figure.set_tight_layout(True)

        # TODO many methods will (confusingly) have no effect if invoked after
        # Plot.plot is (manually) called. We should have some way of raising from
        # within those methods to provide more useful feedback.

        return self

    def clone(self) -> Plot:

        if hasattr(self, "_figure"):
            raise RuntimeError("Cannot clone after calling `Plot.plot`.")
        elif self._target is not None:
            raise RuntimeError("Cannot clone after calling `Plot.on`.")
        return deepcopy(self)

    def show(self, **kwargs) -> None:

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        if self._target is None:
            self.clone().plot(pyplot=True)
        else:
            self.plot(pyplot=True)
        plt.show(**kwargs)

    def save(self) -> Plot:  # TODO perhaps this should not return self?

        raise NotImplementedError()
        return self

    # ================================================================================ #
    # End of public API
    # ================================================================================ #

    def _setup_data(self):

        self._data = (
            self._data
            .concat(
                self._facetspec.get("source"),
                self._facetspec.get("variables"),
            )
            .concat(
                self._pairspec.get("source"),
                self._pairspec.get("variables"),
            )
        )

        # TODO concat with mapping spec

        for layer in self._layers:
            # TODO FIXME:mutable we need to make this not modify the existing object
            # TODO one idea is add() inserts a dict into _layerspec or something
            layer.data = self._data.concat(layer.source, layer.variables)

    def _setup_scales(self) -> None:

        # TODO currently typoing variable name in `scale_*`, or scaling a variable that
        # isn't defined anywhere, silently does nothing. We should raise/warn on that.

        variables = set(self._data.frame)
        for layer in self._layers:
            variables |= set(layer.data.frame)

        for var in (var for var in variables if var not in self._scales):
            all_values = pd.concat([
                self._data.frame.get(var),
                # TODO important to check for var in x.variables, not just in x
                # Because we only want to concat if a variable was *added* here
                *(y.data.frame.get(var) for y in self._layers if var in y.variables)
            ], ignore_index=True)

            # TODO eventually this will be updating a different dictionary
            self._scales[var] = ScaleWrapper.from_inferred_type(all_values)

        # TODO Think about how this is going to handle situations where we have
        # e.g. ymin and ymax but no y specified. I think in that situation one
        # would expect to control the y scale with scale_numeric("y").
        # Actually, if one calls that explicitly, it works. But if they don't,
        # then no scale gets created for y.

    def _setup_mappings(self) -> None:

        # TODO we should setup default mappings here based on whether a mapping
        # variable appears in at least one of the layer data but isn't in self._mappings
        # Source of what mappings to check can be some dictionary of default mappings?
        defined = [v for v in SEMANTICS if any(v in y for y in self._layers)]

        self._mappings = {}
        for var in defined:

            semantic = self._semantics.get(var) or SEMANTICS[var]

            all_values = pd.concat([
                self._data.frame.get(var),
                # TODO important to check for var in x.variables, not just in x
                # Because we only want to concat if a variable was *added* here
                # TODO note copy=pasted from setup_scales code!
                *(x.data.frame.get(var) for x in self._layers if var in x.variables)
            ], ignore_index=True)
            scale = self._scales.get(var)
            self._mappings[var] = semantic.setup(all_values, scale)

    def _setup_figure(self, pyplot: bool = False) -> None:

        # --- Parsing the faceting/pairing parameterization to specify figure grid

        # TODO use context manager with theme that has been set
        # TODO (maybe wrap THIS function with context manager; would be cleaner)

        # Get the full set of assigned variables, whether from constructor or methods
        setup_data = (
            self._data
            .concat(
                self._facetspec.get("source"),
                self._facetspec.get("variables"),
            ).concat(
                self._pairspec.get("source"),  # Currently always None
                self._pairspec.get("variables"),
            )
        )

        self._subplots = subplots = Subplots(
            self._subplotspec, self._facetspec, self._pairspec, setup_data
        )

        # --- Figure initialization
        figure_kws = {"figsize": getattr(self, "_figsize", None)}  # TODO fix
        self._figure = subplots.init_figure(pyplot, figure_kws, self._target)

        # --- Assignment of scales
        for sub in subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]
                if axis_key in self._scales:
                    scale = self._scales[axis_key]._scale
                    if LooseVersion(mpl.__version__) < "3.4":
                        # The ability to pass a BaseScale instance to
                        # Axes.set_{axis}scale was added to matplotlib in version 3.4.0:
                        # https://github.com/matplotlib/matplotlib/pull/19089
                        # Workaround: use the scale name, which is restrictive only
                        # if the user wants to define a custom scale.
                        # Additionally, setting the scale after updating units breaks in
                        # some cases on older versions of matplotlib (/ older pandas?)
                        # so only do it if necessary.
                        axis_obj = getattr(ax, f"{axis}axis")
                        if axis_obj.get_scale() != scale.name:
                            ax.set(**{f"{axis}scale": scale.name})
                    else:
                        ax.set(**{f"{axis}scale": scale})

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
                    setup_data.names.get(axis_key),
                    *[layer.data.names.get(axis_key) for layer in self._layers],
                ]
                label = next((name for name in names if name is not None), None)
                ax.set(**{f"{axis}label": label})

                axis_obj = getattr(ax, f"{axis}axis")
                visible_side = {"x": "bottom", "y": "left"}.get(axis)
                show_axis_label = (
                    sub[visible_side]
                    or axis in self._pairspec and bool(self._pairspec.get("wrap"))
                    or not self._pairspec.get("cartesian", True)
                )
                axis_obj.get_label().set_visible(show_axis_label)
                show_tick_labels = (
                    show_axis_label
                    or self._subplotspec.get(f"share{axis}") not in (
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
                    name = setup_data.names.get(dim, f"_{dim}_")
                    title_parts.append(f"{name} = {sub[dim]}")

            has_col = sub["col"] is not None
            has_row = sub["row"] is not None
            show_title = (
                has_col and has_row
                or (has_col or has_row) and self._facetspec.get("wrap")
                or (has_col and sub["top"])
                # TODO or has_row and sub["right"] and <right titles>
                or has_row  # TODO and not <right titles>
            )
            if title_parts:
                title = " | ".join(title_parts)
                title_text = ax.set_title(title)
                title_text.set_visible(show_title)

    def _plot_layer(self, layer: Layer, mappings: dict[str, SemanticMapping]) -> None:

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?

        data = layer.data
        mark = layer.mark
        stat = layer.stat

        full_df = data.frame
        for subplots, scales, df in self._generate_pairings(full_df):

            df = self._scale_coords(subplots, scales, df)

            if stat is not None:
                grouping_vars = stat.grouping_vars + default_grouping_vars
                df = self._apply_stat(df, grouping_vars, stat)

            df = mark._adjust(df)

            # Our statistics happen on the scale we want, but then matplotlib is going
            # to re-handle the scaling, so we need to invert before handing off
            df = self._unscale_coords(scales, df)

            grouping_vars = mark.grouping_vars + default_grouping_vars
            generate_splits = self._setup_split_generator(
                grouping_vars, df, mappings, subplots
            )

            layer.mark._plot(generate_splits, mappings)

    def _apply_stat(
        self, df: DataFrame, grouping_vars: list[str], stat: Stat
    ) -> DataFrame:

        stat.setup(df)  # TODO pass scales here?

        # TODO how can we special-case fast aggregations? (i.e. mean, std, etc.)
        # IDEA: have Stat identify as an aggregator? (Through Mixin or attribute)
        # e.g. if stat.aggregates ...
        stat_grouping_vars = [var for var in grouping_vars if var in df]
        # TODO I don't think we always want to group by the default orient axis?
        # Better to have the Stat declare when it wants that to happen
        if stat.orient not in stat_grouping_vars:
            stat_grouping_vars.append(stat.orient)

        # TODO rewrite this whole thing, I think we just need to avoid groupby/apply
        df = (
            df
            .groupby(stat_grouping_vars)
            .apply(stat)
        )
        # TODO next because of https://github.com/pandas-dev/pandas/issues/34809
        for var in stat_grouping_vars:
            if var in df.index.names:
                df = (
                    df
                    .drop(var, axis=1, errors="ignore")
                    .reset_index(var)
                )
        df = df.reset_index(drop=True)  # TODO not always needed, can we limit?
        return df

    def _scale_coords(
        self,
        subplots: list[dict],  # TODO retype with a SubplotSpec or similar
        scales: dict[str, ScaleWrapper],  # TODO same idea, but ScaleSpec
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
                axes_df = axes_df.dropna()
            self._scale_coords_single(axes_df, out_df, scales, subplot["ax"])

        return out_df

    def _scale_coords_single(
        self,
        coord_df: DataFrame,
        out_df: DataFrame,
        scales: dict[str, ScaleWrapper],
        ax: Axes,
    ) -> None:

        # TODO modify out_df in place or return and handle externally?
        for var, values in coord_df.items():

            # TODO Explain the logic of this method thoroughly
            # It is clever, but a bit confusing!

            scale = scales[var]
            axis_obj = getattr(ax, f"{var[0]}axis")

            # TODO this is no longer valid with the way the semantic order overrides
            # Perhaps better to have the scale always be the source of the order info
            # but have a step where the order specified in the mapping overrides it?
            # Alternately, use self._orderings here?
            if scale.order is not None:
                values = values[values.isin(scale.order)]

            # TODO FIXME:feedback wrap this in a try/except and reraise with
            # more information about what variable caused the problem
            values = scale.cast(values)
            axis_obj.update_units(categorical_order(values))  # TODO think carefully

            # TODO it seems wrong that we need to cast to float here,
            # but convert_units sometimes outputs an object array (e.g. w/Int64 values)
            scaled = scale.forward(axis_obj.convert_units(values).astype(float))
            out_df.loc[values.index, var] = scaled

    def _unscale_coords(
        self,
        scales: dict[str, ScaleWrapper],
        df: DataFrame
    ) -> DataFrame:

        # Note this is now different from what's in scale_coords as the dataframe
        # that comes into this method will have pair columns reassigned to x/y
        coord_df = df.filter(regex="(^x)|(^y)")
        out_df = (
            df
            .drop(coord_df.columns, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for var, col in coord_df.items():
            axis = var[0]  # TODO check this logic
            out_df[var] = scales[axis].reverse(coord_df[var])

        return out_df

    def _generate_pairings(
        self,
        df: DataFrame
    ) -> Generator[
        tuple[list[dict], dict[str, ScaleWrapper], DataFrame], None, None
    ]:
        # TODO retype return with SubplotSpec or similar

        pair_variables = self._pairspec.get("structure", {})

        if not pair_variables:
            # TODO casting to list because subplots below is a list
            # Maybe a cleaner way to do this?
            yield list(self._subplots), self._scales, df
            return

        iter_axes = itertools.product(*[
            pair_variables.get(axis, [None]) for axis in "xy"
        ])

        for x, y in iter_axes:

            subplots = []
            for sub in self._subplots:
                if (x is None or sub["x"] == x) and (y is None or sub["y"] == y):
                    subplots.append(sub)

            scales = {}
            for axis, prefix in zip("xy", [x, y]):
                key = axis if prefix is None else prefix
                if key in self._scales:
                    scales[axis] = self._scales[key]

            reassignments = {}
            for axis, prefix in zip("xy", [x, y]):
                if prefix is not None:
                    reassignments.update({
                        # Complex regex business to support e.g. x0max
                        re.sub(rf"^{prefix}(.*)$", rf"{axis}\1", col): df[col]
                        for col in df if col.startswith(prefix)
                    })

            yield subplots, scales, df.assign(**reassignments)

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
        mappings: dict[str, SemanticMapping],
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

        def generate_splits() -> Generator:

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

                    yield sub_vars, df_subset.copy(), subplot["ax"]

        return generate_splits

    def _repr_png_(self) -> bytes:

        # TODO better to do this through a Jupyter hook? e.g.
        # ipy = IPython.core.formatters.get_ipython()
        # fmt = ipy.display_formatter.formatters["text/html"]
        # fmt.for_type(Plot, ...)

        # TODO Would like to allow for svg too ... how to configure?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # Preferred behavior is to clone self so that showing a Plot in the REPL
        # does not interfere with adding further layers onto it in the next cell.
        # But we can still show a Plot where the user has manually invoked .plot()
        if hasattr(self, "_figure"):
            figure = self._figure
        elif self._target is None:
            figure = self.clone().plot()._figure
        else:
            figure = self.plot()._figure

        buffer = io.BytesIO()

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()


class Layer:

    data: PlotData

    def __init__(
        self,
        mark: Mark,
        stat: Stat | None,
        source: DataSource | None,
        variables: dict[str, VariableSpec],
    ):

        self.mark = mark
        self.stat = stat
        self.source = source
        self.variables = variables

    def __contains__(self, key: str) -> bool:
        if hasattr(self, "data"):
            return key in self.data
        return False
