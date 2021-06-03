from __future__ import annotations

import io
import itertools

import pandas as pd
import matplotlib as mpl

from ..axisgrid import FacetGrid
from .rules import categorical_order
from .data import PlotData
from .mappings import HueMapping

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Literal, Any
    from collections.abc import Hashable, Mapping, Generator
    from pandas import DataFrame
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.scale import ScaleBase as Scale
    from matplotlib.colors import Normalize
    from .mappings import SemanticMapping
    from .typing import Vector, PaletteSpec
    from .._marks.base import Mark
    from .._stats.base import Stat


class Plot:

    _data: PlotData
    _layers: list[Layer]
    _mappings: dict[str, SemanticMapping]  # TODO keys as Literal, or use TypedDict?
    _scales: dict[str, Scale]

    _figure: Figure
    _ax: Optional[Axes]
    # _facets: Optional[FacetGrid]  # TODO would have a circular import?

    def __init__(
        self,
        data: Optional[DataFrame | Mapping] = None,
        **variables: Optional[Hashable | Vector],
    ):

        # Note that we can't assume wide-form here if variables does not contain x or y
        # because those might get assigned in long-form fashion per layer.
        # TODO I am thinking about just not supporting wide-form data in this interface
        # and handling the reshaping in the functional interface externally

        self._data = PlotData(data, variables)
        self._layers = []
        self._mappings = {}  # TODO initialize with defaults?

        # TODO right place for defaults? (Best to be consistent with mappings)
        self._scales = {
            "x": mpl.scale.LinearScale("x"),
            "y": mpl.scale.LinearScale("y")
        }

    def on(self) -> Plot:

        # TODO  Provisional name for a method that accepts an existing Axes object,
        # and possibly one that does all of the figure/subplot configuration
        raise NotImplementedError()
        return self

    def add(
        self,
        mark: Mark,
        stat: Stat = None,
        data: Optional[DataFrame | Mapping] = None,
        variables: Optional[dict[str, Optional[Hashable | Vector]]] = None,
        orient: Literal["x", "y", "v", "h"] = "x",
    ) -> Plot:

        # TODO what if in wide-form mode, we convert to long-form
        # based on the transform that mark defines?
        layer_data = self._data.concat(data, variables)

        if stat is None:
            stat = mark.default_stat

        orient = {"v": "x", "h": "y"}.get(orient, orient)
        mark.orient = orient
        if stat is not None:
            stat.orient = orient

        self._layers.append(Layer(layer_data, mark, stat))

        return self

    def facet(
        self,
        col: Optional[Hashable | Vector] = None,
        row: Optional[Hashable | Vector] = None,
        col_order: Optional[list] = None,
        row_order: Optional[list] = None,
        col_wrap: Optional[int] = None,
        data: Optional[DataFrame | Mapping] = None,
        # TODO what other parameters? sharex/y?
    ) -> Plot:

        # Note: can't pass `None` here or it will undo the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row
        data = self._data.concat(data, variables)

        # TODO raise here if neither col nor row are defined?

        # TODO do we want to allow this method to be optional and create
        # facets if col or row are defined in Plot()? More convenient...

        # TODO another option would be to have this signature be like
        # facet(dim, order, wrap, share)
        # and expect to call it twice for column and row faceting
        # (or have facet_col, facet_row)?

        # TODO what should this data structure be?
        # We can't initialize a FacetGrid here because that will open a figure
        orders = {"col": col_order, "row": row_order}

        facetspec = {}
        for dim in ["col", "row"]:
            if dim in data:
                facetspec[dim] = {
                    "data": data.frame[dim],
                    "order": categorical_order(data.frame[dim], orders[dim]),
                    "name": data.names[dim],
                }

        # TODO accept row_wrap too? If so, move into above logic
        # TODO alternately, change to wrap?
        if "col" in facetspec:
            facetspec["col"]["wrap"] = col_wrap

        self._facetspec = facetspec
        self._facetdata = data  # TODO messy, but needed if variables are added here

        return self

    def map_hue(
        self,
        palette: Optional[PaletteSpec] = None,
        order: Optional[list] = None,
        norm: Optional[Normalize] = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # ALSO TODO should these be initialized with defaults?
        self._mappings["hue"] = HueMapping(palette, order, norm)
        return self

    def scale_numeric(self, axis, scale="linear", **kwargs) -> Plot:

        scale = mpl.scale.scale_factory(scale, axis, **kwargs)
        self._scales[axis] = scale

        return self

    def theme(self) -> Plot:

        # TODO We want to be able to use the existing seaborn themeing system
        # to do plot-specific theming
        raise NotImplementedError()
        return self

    def plot(self) -> Plot:

        # TODO a rough sketch ...

        # TODO one option is to loop over the layers here and use them to
        # initialize and scaling/mapping we need to do (using parameters)
        # possibly previously set and stored through calls to map_hue etc.
        # Alternately (and probably a better idea), we could concatenate
        # the layer data and then pass that to the Mapping objects to
        # set them up. Note that if strings are passed in one layer and
        # floats in another, this will turn the whole variable into a
        # categorical. That might make sense but it's different from if you
        # plot twice once with strings and then once with numbers.
        # Another option would be to raise if layers have different variable
        # types (this is basically what ggplot does), but that adds complexity.

        # === TODO clean series of setup functions (TODO bikeshed names)
        self._setup_figure()

        # ===

        # TODO we need to be able to show a blank figure
        if not self._layers:
            return self

        mappings = self._setup_mappings()

        for layer in self._layers:

            # TODO alt. assign as attribute on Layer?
            layer_mappings = {k: v for k, v in mappings.items() if k in layer}

            # TODO very messy but needed to concat with variables added in .facet()
            # Demands serious rethinking!
            if hasattr(self, "_facetdata"):
                layer.data = layer.data.concat(
                    self._facetdata.frame,
                    {v: v for v in ["col", "row"] if v in self._facetdata}
                )

            self._plot_layer(layer, layer_mappings)

        return self

    def _setup_figure(self):

        # TODO add external API for parameterizing figure, etc.
        # TODO add external API for parameterizing FacetGrid if using
        # TODO add external API for passing existing ax (maybe in same method)
        # TODO add object that handles the "FacetGrid or single Axes?" abstractions

        if not hasattr(self, "_facetspec"):
            self.facet()  # TODO a good way to activate defaults?

        # TODO use context manager with theme that has been set
        # TODO (or maybe wrap THIS function with context manager; would be cleaner)

        if self._facetspec:

            facet_data = pd.DataFrame()
            facet_vars = {}
            for dim in ["row", "col"]:
                if dim in self._facetspec:
                    name = self._facetspec[dim]["name"]
                    facet_data[name] = self._facetspec[dim]["data"]
                    facet_vars[dim] = name
                if dim == "col":
                    facet_vars["col_wrap"] = self._facetspec[dim]["wrap"]
            grid = FacetGrid(facet_data, **facet_vars, pyplot=False)
            grid.set_titles()

            self._figure = grid.fig
            self._ax = None
            self._facets = grid

        else:

            self._figure = mpl.figure.Figure()
            self._ax = self._figure.add_subplot()
            self._facets = None

        axes_list = list(self._facets.axes.flat) if self._ax is None else [self._ax]
        for ax in axes_list:
            ax.set_xscale(self._scales["x"])
            ax.set_yscale(self._scales["y"])

        # TODO good place to do this? (needs to handle FacetGrid)
        obj = self._ax if self._facets is None else self._facets
        for axis in "xy":
            name = self._data.names.get(axis, None)
            if name is not None:
                obj.set(**{f"{axis}label": name})

        # TODO in current _attach, we initialize the units at this point
        # TODO we will also need to incorporate the scaling that (could) be set

    def _setup_mappings(self) -> dict[str, SemanticMapping]:  # TODO literal key

        all_data = pd.concat([layer.data.frame for layer in self._layers])

        # TODO should mappings hold *all* mappings, and generalize to, e.g.
        # AxisMapping, FacetMapping?
        # One reason this might not work: FacetMapping would need to map
        # col *and* row to get the axes it is looking for.

        # TODO this is a real hack
        class GroupMapping:
            def train(self, vector):
                self.levels = categorical_order(vector)

        # TODO defaults can probably be set up elsewhere
        default_mappings = {  # TODO central source for this!
            "hue": HueMapping,
            "group": GroupMapping,
        }
        for var, mapping in default_mappings.items():
            if var in all_data and var not in self._mappings:
                self._mappings[var] = mapping()  # TODO refactor w/above

        mappings = {}
        for var, mapping in self._mappings.items():
            if var in all_data:
                mapping.train(all_data[var])  # TODO return self?
                mappings[var] = mapping

        return mappings

    def _plot_layer(self, layer, mappings):

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        grouping_vars = layer.mark.grouping_vars + default_grouping_vars

        data = layer.data
        stat = layer.stat

        df = self._scale_coords(data.frame)

        # TODO how to we handle orientation?
        # TODO how can we special-case fast aggregations? (i.e. mean, std, etc.)
        # TODO should we pass the grouping variables to the Stat and let it handle that?
        if stat is not None:  # TODO or default to Identity, but we'll have groupby cost
            stat_grouping_vars = [var for var in grouping_vars if var in data]
            if stat.orient not in stat_grouping_vars:
                stat_grouping_vars.append(stat.orient)
            df = (
                df
                .groupby(stat_grouping_vars)
                .apply(stat)
                # TODO next because of https://github.com/pandas-dev/pandas/issues/34809
                .drop(stat_grouping_vars, axis=1, errors="ignore")
                .reset_index(stat_grouping_vars)
                .reset_index(drop=True)  # TODO not always needed, can we limit?
            )

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        df = self._unscale_coords(df)

        # TODO this might make debugging annoying ... should we create new layer object?
        data.frame = df

        # TODO the layer.data somehow needs to pick up variables added in Plot.facet()
        splitgen = self._make_splitgen(grouping_vars, data, mappings)

        layer.mark._plot(splitgen, mappings)

    def _assign_axes(self, df: DataFrame) -> Axes:
        """Given a faceted DataFrame, find the Axes object for each entry."""
        df = df.filter(regex="row|col")

        if len(df.columns) > 1:
            zipped = zip(df["row"], df["col"])
            facet_keys = pd.Series(zipped, index=df.index)
        else:
            facet_keys = df.squeeze().astype("category")

        return facet_keys.map(self._facets.axes_dict)

    def _scale_coords(self, df):

        # TODO we will want to scale/unscale xmin, xmax, which i *think* this catches?
        coord_df = df.filter(regex="x|y")

        # TODO any reason to scale the semantics here?
        out_df = df.drop(coord_df.columns, axis=1).copy(deep=False)

        with pd.option_context("mode.use_inf_as_null", True):
            coord_df = coord_df.dropna()

        if self._ax is not None:
            self._scale_coords_single(coord_df, out_df, self._ax)
        else:
            axes_map = self._assign_axes(df)
            grouped = coord_df.groupby(axes_map, sort=False)
            for ax, ax_df in grouped:
                self._scale_coords_single(ax_df, out_df, ax)

        # TODO do we need to handle nas again, e.g. if negative values
        # went into a log transform?
        # cf GH2454

        return out_df

    def _scale_coords_single(self, coord_df, out_df, ax):

        # TODO modify out_df in place or return and handle externally?

        # TODO this looped through "yx" in original core ... why?
        # for var in "yx":
        #     if var not in coord_df:
        #        continue
        for var, col in coord_df.items():

            axis = var[0]
            axis_obj = getattr(ax, f"{axis}axis")

            # TODO should happen upstream, in setup_figure(?), but here for now
            # will need to account for order; we don't have that yet
            axis_obj.update_units(col)

            # TODO subset categories based on whether specified in order
            ...

            transform = self._scales[axis].get_transform().transform
            scaled = transform(axis_obj.convert_units(col))
            out_df.loc[col.index, var] = scaled

    def _unscale_coords(self, df):

        # TODO copied from _scale function; refactor!
        # TODO we will want to scale/unscale xmin, xmax, which i *think* this catches?
        coord_df = df.filter(regex="x|y")
        out_df = df.drop(coord_df.columns, axis=1).copy(deep=False)
        for var, col in coord_df.items():
            axis = var[0]
            invert_scale = self._scales[axis].get_transform().inverted().transform
            out_df[var] = invert_scale(coord_df[var])

        if self._ax is not None:
            self._unscale_coords_single(coord_df, out_df, self._ax)
        else:
            # TODO the only reason this structure exists in the forward scale func
            # is to support unshared categorical axes. I don't think there is any
            # situation where numeric axes would have different *transforms*.
            # So we should be able to do this in one step in all cases, once
            # we are storing information about the scaling centrally.
            axes_map = self._assign_axes(df)
            grouped = coord_df.groupby(axes_map, sort=False)
            for ax, ax_df in grouped:
                self._unscale_coords_single(ax_df, out_df, ax)

        return out_df

    def _unscale_coords_single(self, coord_df, out_df, ax):

        for var, col in coord_df.items():

            axis = var[0]
            axis_obj = getattr(ax, f"{axis}axis")
            inverse_transform = axis_obj.get_transform().inverted().transform
            unscaled = inverse_transform(col)
            out_df.loc[col.index, var] = unscaled

    def _make_splitgen(
        self,
        grouping_vars,
        data,
        mappings,
    ):  # TODO typing

        allow_empty = False  # TODO

        df = data.frame
        # TODO join with axes_map to simplify logic below?

        ax = self._ax
        facets = self._facets

        grouping_vars = [var for var in grouping_vars if var in data]
        if grouping_vars:
            grouped_df = df.groupby(grouping_vars, sort=False, as_index=False)

        levels = {v: m.levels for v, m in mappings.items()}
        if facets is not None:
            for dim in ["col", "row"]:
                if dim in grouping_vars:
                    levels[dim] = getattr(facets, f"{dim}_names")

        grouping_keys = []
        for var in grouping_vars:
            grouping_keys.append(levels.get(var, []))

        iter_keys = itertools.product(*grouping_keys)

        def splitgen() -> Generator[dict[str, Any], DataFrame, Axes]:

            if not grouping_vars:
                yield {}, df.copy(), ax
                return

            for key in iter_keys:

                # Pandas fails with singleton tuple inputs
                pd_key = key[0] if len(key) == 1 else key

                try:
                    df_subset = grouped_df.get_group(pd_key)
                except KeyError:
                    # XXX we are adding this to allow backwards compatability
                    # with the empty artists that old categorical plots would
                    # add (before 0.12), which we may decide to break, in which
                    # case this option could be removed
                    df_subset = df.loc[[]]

                if df_subset.empty and not allow_empty:
                    continue

                sub_vars = dict(zip(grouping_vars, key))

                # TODO can we use axes_map here?
                row = sub_vars.get("row", None)
                col = sub_vars.get("col", None)
                if row is not None and col is not None:
                    use_ax = facets.axes_dict[(row, col)]
                elif row is not None:
                    use_ax = facets.axes_dict[row]
                elif col is not None:
                    use_ax = facets.axes_dict[col]
                else:
                    use_ax = ax
                yield sub_vars, df_subset.copy(), use_ax

        return splitgen

    def show(self) -> Plot:

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise. In this vision, it would
        # make sense to specify whether or not to use pyplot at the initial Plot().
        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        import matplotlib.pyplot as plt  # type: ignore
        self.plot()
        plt.show()

        return self

    def save(self) -> Plot:  # or to_file or similar to match pandas?

        raise NotImplementedError()
        return self

    def _repr_png_(self) -> bytes:

        # TODO better to do this through a Jupyter hook?
        # TODO Would like to allow for svg too ... how to configure?
        # TODO We want to skip if the plot has otherwise been shown, but tricky...

        # TODO we need some way of not plotting multiple times
        if not hasattr(self, "_figure"):
            self.plot()

        buffer = io.BytesIO()

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        self._figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()


class Layer:

    # Does this need to be anything other than a simple container for these attributes?
    # Could use a Dataclass I guess?

    def __init__(self, data: PlotData, mark: Mark, stat: Stat = None):

        self.data = data
        self.mark = mark
        self.stat = stat

    def __contains__(self, key: Hashable) -> bool:
        return key in self.data
