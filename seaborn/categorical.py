from textwrap import dedent
from numbers import Number
import warnings
from colorsys import rgb_to_hls
from functools import partial

import numpy as np
import pandas as pd
try:
    from scipy.stats import gaussian_kde
    _no_scipy = False
except ImportError:
    from .external.kde import gaussian_kde
    _no_scipy = True

import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt

from seaborn._core.typing import default, deprecated
from seaborn._oldcore import (
    variable_type,
    infer_orient,
    categorical_order,
)
from seaborn.relational import _RelationalPlotter
from seaborn import utils
from seaborn.utils import (
    remove_na,
    desaturate,
    _check_argument,
    _draw_figure,
    _default_color,
    _normal_quantile_func,
    _normalize_kwargs,
)
from seaborn._statistics import EstimateAggregator
from seaborn.palettes import color_palette, husl_palette, light_palette, dark_palette
from seaborn.axisgrid import FacetGrid, _facet_docs


__all__ = [
    "catplot",
    "stripplot", "swarmplot",
    "boxplot", "violinplot", "boxenplot",
    "pointplot", "barplot", "countplot",
]


# Subclassing _RelationalPlotter for the legend machinery,
# but probably should move that more centrally
class _CategoricalPlotterNew(_RelationalPlotter):

    semantics = "x", "y", "hue", "units"

    wide_structure = {"x": "@columns", "y": "@values"}
    flat_structure = {"y": "@values"}

    _legend_func = "scatter"
    _legend_attributes = ["color"]

    def __init__(
        self,
        data=None,
        variables={},
        order=None,
        orient=None,
        require_numeric=False,
        legend="auto",
    ):

        super().__init__(data=data, variables=variables)

        # This method takes care of some bookkeeping that is necessary because the
        # original categorical plots (prior to the 2021 refactor) had some rules that
        # don't fit exactly into the logic of _core. It may be wise to have a second
        # round of refactoring that moves the logic deeper, but this will keep things
        # relatively sensible for now.

        # For wide data, orient determines assignment to x/y differently from the
        # wide_structure rules in _core. If we do decide to make orient part of the
        # _core variable assignment, we'll want to figure out how to express that.
        if self.input_format == "wide" and orient in ["h", "y"]:
            self.plot_data = self.plot_data.rename(columns={"x": "y", "y": "x"})
            orig_variables = set(self.variables)
            orig_x = self.variables.pop("x", None)
            orig_y = self.variables.pop("y", None)
            orig_x_type = self.var_types.pop("x", None)
            orig_y_type = self.var_types.pop("y", None)
            if "x" in orig_variables:
                self.variables["y"] = orig_x
                self.var_types["y"] = orig_x_type
            if "y" in orig_variables:
                self.variables["x"] = orig_y
                self.var_types["x"] = orig_y_type

        # The concept of an "orientation" is important to the original categorical
        # plots, but there's no provision for it in _core, so we need to do it here.
        # Note that it could be useful for the other functions in at least two ways
        # (orienting a univariate distribution plot from long-form data and selecting
        # the aggregation axis in lineplot), so we may want to eventually refactor it.
        self.orient = infer_orient(
            x=self.plot_data.get("x", None),
            y=self.plot_data.get("y", None),
            orient=orient,
            require_numeric=require_numeric,
        )

        self.legend = legend

        # Short-circuit in the case of an empty plot
        if not self.has_xy_data:
            return

        # Categorical plots can be "univariate" in which case they get an anonymous
        # category label on the opposite axis. Note: this duplicates code in the core
        # scale_categorical function. We need to do it here because of the next line.
        if self.orient not in self.variables:
            self.variables[self.orient] = None
            self.var_types[self.orient] = "categorical"
            self.plot_data[self.orient] = ""

        # Categorical variables have discrete levels that we need to track
        cat_levels = categorical_order(self.plot_data[self.orient], order)
        self.var_levels[self.orient] = cat_levels

    def _hue_backcompat(self, color, palette, hue_order, force_hue=False):
        """Implement backwards compatibility for hue parametrization.

        Note: the force_hue parameter is used so that functions can be shown to
        pass existing tests during refactoring and then tested for new behavior.
        It can be removed after completion of the work.

        """
        # The original categorical functions applied a palette to the categorical axis
        # by default. We want to require an explicit hue mapping, to be more consistent
        # with how things work elsewhere now. I don't think there's any good way to
        # do this gently -- because it's triggered by the default value of hue=None,
        # users would always get a warning, unless we introduce some sentinel "default"
        # argument for this change. That's possible, but asking users to set `hue=None`
        # on every call is annoying.
        # We are keeping the logic for implementing the old behavior in with the current
        # system so that (a) we can punt on that decision and (b) we can ensure that
        # refactored code passes old tests.
        default_behavior = color is None or palette is not None
        if force_hue and "hue" not in self.variables and default_behavior:
            self._redundant_hue = True
            self.plot_data["hue"] = self.plot_data[self.orient]
            self.variables["hue"] = self.variables[self.orient]
            self.var_types["hue"] = "categorical"
            hue_order = self.var_levels[self.orient]

            # Because we convert the categorical axis variable to string,
            # we need to update a dictionary palette too
            if isinstance(palette, dict):
                palette = {str(k): v for k, v in palette.items()}

        else:
            self._redundant_hue = False

        # Previously, categorical plots had a trick where color= could seed the palette.
        # Because that's an explicit parameterization, we are going to give it one
        # release cycle with a warning before removing.
        if "hue" in self.variables and palette is None and color is not None:
            if not isinstance(color, str):
                color = mpl.colors.to_hex(color)
            palette = f"dark:{color}"
            msg = (
                "\n\nSetting a gradient palette using color= is deprecated and will be "
                f"removed in v0.14.0. Set `palette='{palette}'` for the same effect.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

        return palette, hue_order

    def _palette_without_hue_backcompat(self, palette, hue_order):
        """Provide one cycle where palette= implies hue= when not provided"""
        if "hue" not in self.variables and palette is not None:
            msg = (
                "\n\nPassing `palette` without assigning `hue` is deprecated "
                f"and will be removed in v0.14.0. Assign the `{self.orient}` variable "
                "to `hue` and set `legend=False` for the same effect.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

            self.legend = False
            self.plot_data["hue"] = self.plot_data[self.orient]
            self.variables["hue"] = self.variables.get(self.orient)
            self.var_types["hue"] = self.var_types.get(self.orient)

            hue_order = self.var_levels.get(self.orient)
            self._var_levels.pop("hue", None)

        return hue_order

    def _point_kwargs_backcompat(self, scale, join, kwargs):
        """Provide two cycles where scale= and join= work, but redirect to kwargs."""
        if scale is not deprecated:
            lw = mpl.rcParams["lines.linewidth"] * 1.8 * scale
            mew = lw * .75
            ms = lw * 2

            msg = (
                "\n\n"
                "The `scale` parameter is deprecated and will be removed in v0.15.0. "
                "You can now control the size of each plot element using matplotlib "
                "`Line2D` parameters (e.g., `linewidth`, `markersize`, etc.)."
                "\n"
            )
            warnings.warn(msg, stacklevel=3)
            kwargs.update(linewidth=lw, markeredgewidth=mew, markersize=ms)

        if join is not deprecated:
            msg = (
                "\n\n"
                "The `join` parameter is deprecated and will be removed in v0.15.0."
            )
            if not join:
                msg += (
                    " You can remove the line between points with `linestyle='none'`."
                )
                kwargs.update(linestyle="")
            msg += "\n"
            warnings.warn(msg, stacklevel=3)

    def _err_kws_backcompat(self, err_kws, errcolor, errwidth, capsize):
        """Provide two cycles where existing signature-level err_kws are handled."""
        def deprecate_err_param(name, key, val):
            if val is deprecated:
                return
            suggest = f"err_kws={{'{key}': {val!r}}}"
            msg = (
                f"\n\nThe `{name}` parameter is deprecated. And will be removed "
                f"in v0.15.0. Pass `{suggest}` instead.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=4)
            err_kws[key] = val

        if errcolor is not None:
            deprecate_err_param("errcolor", "color", errcolor)
        deprecate_err_param("errwidth", "linewidth", errwidth)

        if capsize is None:
            capsize = 0
            msg = (
                "\n\nPassing `capsize=None` is deprecated and will be removed "
                "in v0.15.0. Pass `capsize=0` to disable caps.\n"
            )
            warnings.warn(msg, FutureWarning, stacklevel=3)

        return err_kws, capsize

    def _get_gray(self, colors):
        """Get a grayscale value that looks good with color."""
        if not len(colors):
            return None
        colors = [mpl.colors.to_rgb(c) for c in colors]
        unique_colors = np.unique(colors, axis=0)
        light_vals = [rgb_to_hls(*rgb[:3])[1] for rgb in unique_colors]
        lum = min(light_vals) * .6
        return (lum, lum, lum)

    def _map_prop_with_hue(self, name, value, fallback, plot_kws):
        """Support pointplot behavior of modifying the marker/linestyle with hue."""
        if value is default:
            value = plot_kws.pop(name, fallback)

        if (levels := self._hue_map.levels) is None:
            mapping = {None: value}
        else:
            if isinstance(value, list):
                mapping = {k: v for k, v in zip(levels, value)}
            else:
                mapping = {k: value for k in levels}

        return mapping

    def _adjust_cat_axis(self, ax, axis):
        """Set ticks and limits for a categorical variable."""
        # Note: in theory, this could happen in _attach for all categorical axes
        # But two reasons not to do that:
        # - If it happens before plotting, autoscaling messes up the plot limits
        # - It would change existing plots from other seaborn functions
        if self.var_types[axis] != "categorical":
            return

        # If both x/y data are empty, the correct way to set up the plot is
        # somewhat undefined; because we don't add null category data to the plot in
        # this case we don't *have* a categorical axis (yet), so best to just bail.
        if self.plot_data[axis].empty:
            return

        # We can infer the total number of categories (including those from previous
        # plots that are not part of the plot we are currently making) from the number
        # of ticks, which matplotlib sets up while doing unit conversion. This feels
        # slightly risky, as if we are relying on something that may be a matplotlib
        # implementation detail. But I cannot think of a better way to keep track of
        # the state from previous categorical calls (see GH2516 for context)
        n = len(getattr(ax, f"get_{axis}ticks")())

        if axis == "x":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, n - .5, auto=None)
        else:
            ax.yaxis.grid(False)
            # Note limits that correspond to previously-inverted y axis
            ax.set_ylim(n - .5, -.5, auto=None)

    def _dodge_needed(self):
        """Return True when use of `hue` would cause overlaps."""
        groupers = list({self.orient, "col", "row"} & set(self.variables))
        if "hue" in self.variables:
            orient = self.plot_data[groupers].value_counts()
            paired = self.plot_data[[*groupers, "hue"]].value_counts()
            return orient.size != paired.size
        return False

    def _invert_scale(self, data, vars=("x", "y")):
        """Undo log scaling after computation so data are plotted correctly."""
        for var in vars:
            if self._log_scaled(var):
                for suf in ["", "min", "max"]:
                    if (col := f"{var}{suf}") in data:
                        data[col] = np.power(10, data[col])

    def _configure_legend(self, ax, func, common_kws=None, semantic_kws=None):

        if self.legend == "auto":
            show_legend = not self._redundant_hue and self.input_format != "wide"
        else:
            show_legend = bool(self.legend)

        if show_legend:
            self.add_legend_data(ax, func, common_kws, semantic_kws)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(title=self.legend_title)

    @property
    def _native_width(self):
        """Return unit of width separating categories on native numeric scale."""
        unique_values = np.unique(self.comp_data[self.orient])
        if len(unique_values) > 1:
            native_width = np.nanmin(np.diff(unique_values))
        else:
            native_width = 1
        return native_width

    def _nested_offsets(self, width, dodge):
        """Return offsets for each hue level for dodged plots."""
        offsets = None
        if "hue" in self.variables and self._hue_map.levels is not None:
            n_levels = len(self._hue_map.levels)
            if dodge:
                each_width = width / n_levels
                offsets = np.linspace(0, width - each_width, n_levels)
                offsets -= offsets.mean()
            else:
                offsets = np.zeros(n_levels)
        return offsets

    # Note that the plotting methods here aim (in most cases) to produce the
    # exact same artists as the original (pre 0.12) version of the code, so
    # there is some weirdness that might not otherwise be clean or make sense in
    # this context, such as adding empty artists for combinations of variables
    # with no observations

    def plot_strips(
        self,
        jitter,
        dodge,
        color,
        edgecolor,
        plot_kws,
    ):

        width = .8 * self._native_width
        offsets = self._nested_offsets(width, dodge)

        if jitter is True:
            jlim = 0.1
        else:
            jlim = float(jitter)
        if "hue" in self.variables and dodge and self._hue_map.levels is not None:
            jlim /= len(self._hue_map.levels)
        jlim *= self._native_width
        jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)

        iter_vars = [self.orient]
        if dodge:
            iter_vars.append("hue")

        ax = self.ax
        dodge_move = jitter_move = 0

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):

            if offsets is not None and (offsets != 0).any():
                dodge_move = offsets[sub_data["hue"].map(self._hue_map.levels.index)]

            jitter_move = jitterer(size=len(sub_data)) if len(sub_data) > 1 else 0

            adjusted_data = sub_data[self.orient] + dodge_move + jitter_move
            sub_data[self.orient] = adjusted_data
            self._invert_scale(sub_data)

            ax = self._get_axes(sub_vars)
            points = ax.scatter(sub_data["x"], sub_data["y"], color=color, **plot_kws)

            if "hue" in self.variables:
                points.set_facecolors(self._hue_map(sub_data["hue"]))

            if edgecolor == "gray":  # XXX TODO change to "auto"
                points.set_edgecolors(self._get_gray(points.get_facecolors()))
            else:
                points.set_edgecolors(edgecolor)

        self._configure_legend(ax, ax.scatter)

    def plot_swarms(
        self,
        dodge,
        color,
        edgecolor,
        warn_thresh,
        plot_kws,
    ):

        width = .8 * self._native_width
        offsets = self._nested_offsets(width, dodge)

        iter_vars = [self.orient]
        if dodge:
            iter_vars.append("hue")

        ax = self.ax
        point_collections = {}
        dodge_move = 0

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):

            if offsets is not None:
                dodge_move = offsets[sub_data["hue"].map(self._hue_map.levels.index)]

            if not sub_data.empty:
                sub_data[self.orient] = sub_data[self.orient] + dodge_move

            self._invert_scale(sub_data)
            ax = self._get_axes(sub_vars)
            points = ax.scatter(sub_data["x"], sub_data["y"], color=color, **plot_kws)

            if "hue" in self.variables:
                points.set_facecolors(self._hue_map(sub_data["hue"]))

            if edgecolor == "gray":  # XXX TODO change to "auto"
                points.set_edgecolors(self._get_gray(points.get_facecolors()))
            else:
                points.set_edgecolors(edgecolor)

            if not sub_data.empty:
                point_collections[(ax, sub_data[self.orient].iloc[0])] = points

        beeswarm = Beeswarm(
            width=width, orient=self.orient, warn_thresh=warn_thresh,
        )
        for (ax, center), points in point_collections.items():
            if points.get_offsets().shape[0] > 1:

                def draw(points, renderer, *, center=center):

                    beeswarm(points, center)

                    if self.orient == "y":
                        scalex = False
                        scaley = ax.get_autoscaley_on()
                    else:
                        scalex = ax.get_autoscalex_on()
                        scaley = False

                    # This prevents us from undoing the nice categorical axis limits
                    # set in _adjust_cat_axis, because that method currently leave
                    # the autoscale flag in its original setting. It may be better
                    # to disable autoscaling there to avoid needing to do this.
                    fixed_scale = self.var_types[self.orient] == "categorical"
                    ax.update_datalim(points.get_datalim(ax.transData))
                    if not fixed_scale and (scalex or scaley):
                        ax.autoscale_view(scalex=scalex, scaley=scaley)

                    super(points.__class__, points).draw(renderer)

                points.draw = draw.__get__(points)

        _draw_figure(ax.figure)
        self._configure_legend(ax, ax.scatter)

    def plot_boxes(
        self,
        width,
        dodge,
        gap,
        fill,
        whis,
        color,
        linecolor,
        linewidth,
        fliersize,
        saturation,
        plot_kws,
    ):

        iter_vars = ["hue"]
        value_var = {"x": "y", "y": "x"}[self.orient]

        if linecolor is None:
            if "hue" in self.variables:
                line_color = self._get_gray(list(self._hue_map.lookup_table.values()))
            else:
                line_color = self._get_gray([color])

        def get_props(element, artist=mpl.lines.Line2D):
            return _normalize_kwargs(plot_kws.pop(f"{element}props", {}), artist)

        if not fill and linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]

        box_artist = mpl.patches.Rectangle if fill else mpl.lines.Line2D
        props = {
            "box": get_props("box", box_artist),
            "median": get_props("median"),
            "whisker": get_props("whisker"),
            "flier": get_props("flier"),
            "cap": get_props("cap"),
        }

        props["flier"].setdefault("markersize", fliersize)

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):

            grouped = sub_data.groupby(self.orient)[value_var]
            value_data = [x.to_numpy() for _, x in grouped]
            stats = pd.DataFrame(mpl.cbook.boxplot_stats(value_data, whis=whis))
            positions = grouped.grouper.result_index.to_numpy(dtype=float)

            if self._log_scaled(self.orient):
                positions = 10 ** positions
            if self._log_scaled(value_var):
                stats = stats.apply(lambda x: 10 ** x)

            real_width = width * self._native_width
            if dodge:
                hue_idx = self._hue_map.levels.index(sub_vars["hue"])
                real_width /= len(self._hue_map.levels)

                full_width = real_width * len(self._hue_map.levels)
                offset = real_width * hue_idx + real_width / 2 - full_width / 2
                if self._log_scaled(self.orient):
                    positions = 10 ** (np.log10(positions) + offset)
                else:
                    positions += offset

            if gap:
                real_width *= 1 - gap

            if "hue" in sub_vars:
                main_color = self._hue_map(sub_vars["hue"])
                if saturation != 1:
                    main_color = desaturate(main_color, saturation)
            else:
                main_color = color

            # TODO how to handle solid / empty fliers?

            if fill:
                boxprops = {
                    "facecolor": main_color, "edgecolor": line_color, **props["box"]
                }
                medianprops = {"color": line_color, **props["median"]}
                whiskerprops = {"color": line_color, **props["whisker"]}
                flierprops = {"markeredgecolor": line_color, **props["flier"]}
                capprops = {"color": line_color, **props["cap"]}
            else:
                boxprops = {"color": main_color, **props["box"]}
                medianprops = {"color": main_color, **props["median"]}
                whiskerprops = {"color": main_color, **props["whisker"]}
                flierprops = {"markeredgecolor": main_color, **props["flier"]}
                capprops = {"color": main_color, **props["cap"]}

            if linewidth is not None:
                for prop_dict in [boxprops, medianprops, whiskerprops, capprops]:
                    prop_dict.setdefault("linewidth", linewidth)

            ax = self._get_axes(sub_vars)
            artists = ax.bxp(
                bxpstats=stats.to_dict("records"),
                positions=positions,
                widths=0 if self._log_scaled(self.orient) else real_width,
                patch_artist=fill,
                vert=self.orient == "x",
                manage_ticks=False,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                flierprops=flierprops,
                capprops=capprops,
                **plot_kws,
            )

            ori_idx = ["x", "y"].index(self.orient)
            if self._log_scaled(self.orient):
                for i, pos in enumerate(positions):
                    p0 = 10 ** (np.log10(pos) - real_width / 2)
                    p1 = 10 ** (np.log10(pos) + real_width / 2)
                    if artists["boxes"]:
                        box_artist = artists["boxes"][i]
                        if fill:
                            box_verts = box_artist.get_path().vertices.T
                        else:
                            box_verts = box_artist.get_data()
                        box_verts[ori_idx][0] = p0
                        box_verts[ori_idx][3:] = p0
                        box_verts[ori_idx][1:3] = p1
                        if not fill:
                            box_artist.set_data(box_verts)
                        ax.update_datalim(np.transpose(box_verts))

                    if artists["medians"]:
                        verts = artists["medians"][i].get_xydata().T
                        verts[ori_idx][:] = p0, p1
                        artists["medians"][i].set_data(verts)

                    if artists["caps"]:
                        capwidth = plot_kws.get("capwidths", 0.5 * real_width)
                        for line in artists["caps"][2 * i:2 * i + 2]:
                            p0 = 10 ** (np.log10(pos) - capwidth / 2)
                            p1 = 10 ** (np.log10(pos) + capwidth / 2)
                            verts = line.get_xydata().T
                            verts[ori_idx][:] = p0, p1
                            line.set_data(verts)

            ax.add_container(BoxPlotContainer(artists))

    def plot_points(
        self,
        aggregator,
        markers,
        linestyles,
        dodge,
        color,
        capsize,
        err_kws,
        plot_kws,
    ):

        agg_var = {"x": "y", "y": "x"}[self.orient]
        iter_vars = ["hue"]

        plot_kws = _normalize_kwargs(plot_kws, mpl.lines.Line2D)
        plot_kws.setdefault("linewidth", mpl.rcParams["lines.linewidth"] * 1.8)
        plot_kws.setdefault("markeredgewidth", plot_kws["linewidth"] * 0.75)
        plot_kws.setdefault("markersize", plot_kws["linewidth"] * np.sqrt(2 * np.pi))

        markers = self._map_prop_with_hue("marker", markers, "o", plot_kws)
        linestyles = self._map_prop_with_hue("linestyle", linestyles, "-", plot_kws)

        positions = self.var_levels[self.orient]
        if self.var_types[self.orient] == "categorical":
            positions = [i for i, _ in enumerate(positions)]
        else:
            if self._log_scaled(self.orient):
                positions = np.log10(positions)
            if self.var_types[self.orient] == "datetime":
                positions = mpl.dates.date2num(positions)
        positions = pd.Index(positions, name=self.orient)

        n_hue_levels = 0 if self._hue_map.levels is None else len(self._hue_map.levels)
        if dodge is True:
            dodge = .025 * n_hue_levels

        ax = self.ax

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):

            agg_data = sub_data if sub_data.empty else (
                sub_data
                .groupby(self.orient)
                .apply(aggregator, agg_var)
                .reindex(positions)
                .reset_index()
            )

            if dodge:
                hue_idx = self._hue_map.levels.index(sub_vars["hue"])
                offset = -dodge * (n_hue_levels - 1) / 2 + dodge * hue_idx
                agg_data[self.orient] += offset * self._native_width

            self._invert_scale(agg_data)

            sub_kws = plot_kws.copy()
            sub_kws.update(
                marker=markers[sub_vars.get("hue")],
                linestyle=linestyles[sub_vars.get("hue")],
                color=self._hue_map(sub_vars["hue"]) if "hue" in sub_vars else color,
            )

            ax = self._get_axes(sub_vars)
            line, = ax.plot(agg_data["x"], agg_data["y"], **sub_kws)

            sub_err_kws = err_kws.copy()
            line_props = line.properties()
            for prop in ["color", "linewidth", "alpha", "zorder"]:
                sub_err_kws.setdefault(prop, line_props[prop])
            if aggregator.error_method is not None:
                self.plot_errorbars(ax, agg_data, capsize, sub_err_kws)

        semantic_kws = {"hue": {"marker": markers, "linestyle": linestyles}}
        self._configure_legend(ax, ax.plot, sub_kws, semantic_kws)

    def plot_bars(
        self,
        aggregator,
        dodge,
        width,
        color,
        saturation,
        capsize,
        err_kws,
        plot_kws,
    ):

        agg_var = {"x": "y", "y": "x"}[self.orient]
        iter_vars = ["hue"]

        ax = self.ax

        if self._hue_map.levels is None:
            dodge = False

        if dodge and capsize is not None:
            capsize = capsize / len(self._hue_map.levels)

        err_kws.setdefault("color", ".26")
        err_kws.setdefault("linewidth", 1.5 * mpl.rcParams["lines.linewidth"])

        for sub_vars, sub_data in self.iter_data(iter_vars,
                                                 from_comp_data=True,
                                                 allow_empty=True):

            agg_data = sub_data if sub_data.empty else (
                sub_data
                .groupby(self.orient)
                .apply(aggregator, agg_var)
                .reset_index()
            )

            real_width = width * self._native_width
            if dodge:
                hue_idx = self._hue_map.levels.index(sub_vars["hue"])
                real_width /= len(self._hue_map.levels)

                full_width = real_width * len(self._hue_map.levels)
                offset = real_width * hue_idx + real_width / 2 - full_width / 2
                agg_data[self.orient] += offset

            agg_data["edge"] = agg_data[self.orient] - real_width / 2
            if self._log_scaled(self.orient):
                agg_data["edge"] = 10 ** agg_data["edge"]
                right_edge = 10 ** (agg_data[self.orient] + real_width / 2)
                real_width = right_edge - agg_data["edge"]
            self._invert_scale(agg_data)

            ax = self._get_axes(sub_vars)

            if self.orient == "x":
                bar_func = ax.bar
                kws = dict(x=agg_data["edge"], height=agg_data["y"], width=real_width)
            else:
                bar_func = ax.barh
                kws = dict(y=agg_data["edge"], width=agg_data["x"], height=real_width)

            container = bar_func(
                **kws,
                color=color,
                align="edge",
                **plot_kws,
            )

            if "hue" in self.variables:
                mapped_color = self._hue_map(sub_vars["hue"])
                if saturation < 1:
                    mapped_color = desaturate(mapped_color, saturation)
                plt.setp(container, facecolor=mapped_color)

            if aggregator.error_method is not None:
                self.plot_errorbars(ax, agg_data, capsize, err_kws.copy())

        self._configure_legend(ax, ax.fill_between)

    def plot_errorbars(self, ax, data, capsize, err_kws):

        var = {"x": "y", "y": "x"}[self.orient]
        for row in data.to_dict("records"):

            row = dict(row)
            pos = np.array([row[self.orient], row[self.orient]])
            val = np.array([row[f"{var}min"], row[f"{var}max"]])

            cw = capsize * self._native_width / 2
            if self._log_scaled(self.orient):
                log_pos = np.log10(pos)
                cap = 10 ** (log_pos[0] - cw), 10 ** (log_pos[1] + cw)
            else:
                cap = pos[0] - cw, pos[1] + cw

            if capsize:
                pos = np.concatenate([
                    [*cap, np.nan], pos, [np.nan, *cap]
                ])
                val = np.concatenate([
                    [val[0], val[0], np.nan], val, [np.nan, val[-1], val[-1]],
                ])

            if self.orient == "x":
                args = pos, val
            else:
                args = val, pos
            ax.plot(*args, **err_kws)


class _CategoricalAggPlotter(_CategoricalPlotterNew):

    flat_structure = {"x": "@index", "y": "@values"}


class _CategoricalFacetPlotter(_CategoricalPlotterNew):
    semantics = _CategoricalPlotterNew.semantics + ("col", "row")


class _CategoricalAggFacetPlotter(_CategoricalAggPlotter, _CategoricalFacetPlotter):
    # Ugh, this is messy
    pass


class _CategoricalPlotter:

    width = .8
    default_palette = "light"
    require_numeric = True

    def establish_variables(self, x=None, y=None, hue=None, data=None,
                            orient=None, order=None, hue_order=None,
                            units=None):
        """Convert input specification into a common representation."""
        # Option 1:
        # We are plotting a wide-form dataset
        # -----------------------------------
        if x is None and y is None:

            # Do a sanity check on the inputs
            if hue is not None:
                error = "Cannot use `hue` without `x` and `y`"
                raise ValueError(error)

            # No hue grouping with wide inputs
            plot_hues = None
            hue_title = None
            hue_names = None

            # No statistical units with wide inputs
            plot_units = None

            # We also won't get a axes labels here
            value_label = None
            group_label = None

            # Option 1a:
            # The input data is a Pandas DataFrame
            # ------------------------------------

            if isinstance(data, pd.DataFrame):

                # Order the data correctly
                if order is None:
                    order = []
                    # Reduce to just numeric columns
                    for col in data:
                        if variable_type(data[col]) == "numeric":
                            order.append(col)
                plot_data = data[order]
                group_names = order
                group_label = data.columns.name

                # Convert to a list of arrays, the common representation
                iter_data = plot_data.items()
                plot_data = [np.asarray(s, float) for k, s in iter_data]

            # Option 1b:
            # The input data is an array or list
            # ----------------------------------

            else:

                # We can't reorder the data
                if order is not None:
                    error = "Input data must be a pandas object to reorder"
                    raise ValueError(error)

                # The input data is an array
                if hasattr(data, "shape"):
                    if len(data.shape) == 1:
                        if np.isscalar(data[0]):
                            plot_data = [data]
                        else:
                            plot_data = list(data)
                    elif len(data.shape) == 2:
                        nr, nc = data.shape
                        if nr == 1 or nc == 1:
                            plot_data = [data.ravel()]
                        else:
                            plot_data = [data[:, i] for i in range(nc)]
                    else:
                        error = ("Input `data` can have no "
                                 "more than 2 dimensions")
                        raise ValueError(error)

                # Check if `data` is None to let us bail out here (for testing)
                elif data is None:
                    plot_data = [[]]

                # The input data is a flat list
                elif np.isscalar(data[0]):
                    plot_data = [data]

                # The input data is a nested list
                # This will catch some things that might fail later
                # but exhaustive checks are hard
                else:
                    plot_data = data

                # Convert to a list of arrays, the common representation
                plot_data = [np.asarray(d, float) for d in plot_data]

                # The group names will just be numeric indices
                group_names = list(range(len(plot_data)))

            # Figure out the plotting orientation
            orient = "y" if str(orient)[0] in "hy" else "x"

        # Option 2:
        # We are plotting a long-form dataset
        # -----------------------------------

        else:

            # See if we need to get variables from `data`
            if data is not None:
                x = data.get(x, x)
                y = data.get(y, y)
                hue = data.get(hue, hue)
                units = data.get(units, units)

            # Validate the inputs
            for var in [x, y, hue, units]:
                if isinstance(var, str):
                    err = f"Could not interpret input '{var}'"
                    raise ValueError(err)

            # Figure out the plotting orientation
            orient = infer_orient(x, y, orient, require_numeric=self.require_numeric)

            # Option 2a:
            # We are plotting a single set of data
            # ------------------------------------
            if x is None or y is None:

                # Determine where the data are
                vals = y if x is None else x

                # Put them into the common representation
                plot_data = [np.asarray(vals)]

                # Get a label for the value axis
                if hasattr(vals, "name"):
                    value_label = vals.name
                else:
                    value_label = None

                # This plot will not have group labels or hue nesting
                groups = None
                group_label = None
                group_names = []
                plot_hues = None
                hue_names = None
                hue_title = None
                plot_units = None

            # Option 2b:
            # We are grouping the data values by another variable
            # ---------------------------------------------------
            else:

                # Determine which role each variable will play
                if orient == "x":
                    vals, groups = y, x
                else:
                    vals, groups = x, y

                # Get the categorical axis label
                group_label = None
                if hasattr(groups, "name"):
                    group_label = groups.name

                # Get the order on the categorical axis
                group_names = categorical_order(groups, order)

                # Group the numeric data
                plot_data, value_label = self._group_longform(vals, groups,
                                                              group_names)

                # Now handle the hue levels for nested ordering
                if hue is None:
                    plot_hues = None
                    hue_title = None
                    hue_names = None
                else:

                    # Get the order of the hue levels
                    hue_names = categorical_order(hue, hue_order)

                    # Group the hue data
                    plot_hues, hue_title = self._group_longform(hue, groups,
                                                                group_names)

                # Now handle the units for nested observations
                if units is None:
                    plot_units = None
                else:
                    plot_units, _ = self._group_longform(units, groups,
                                                         group_names)

        # Assign object attributes
        # ------------------------
        self.orient = orient
        self.plot_data = plot_data
        self.group_label = group_label
        self.value_label = value_label
        self.group_names = group_names
        self.plot_hues = plot_hues
        self.hue_title = hue_title
        self.hue_names = hue_names
        self.plot_units = plot_units

    def _group_longform(self, vals, grouper, order):
        """Group a long-form variable by another with correct order."""
        # Ensure that the groupby will work
        if not isinstance(vals, pd.Series):
            if isinstance(grouper, pd.Series):
                index = grouper.index
            else:
                index = None
            vals = pd.Series(vals, index=index)

        # Group the val data
        grouped_vals = vals.groupby(grouper)
        out_data = []
        for g in order:
            try:
                g_vals = grouped_vals.get_group(g)
            except KeyError:
                g_vals = np.array([])
            out_data.append(g_vals)

        # Get the vals axis label
        label = vals.name

        return out_data, label

    def establish_colors(self, color, palette, saturation):
        """Get a list of colors for the main component of the plots."""
        if self.hue_names is None:
            n_colors = len(self.plot_data)
        else:
            n_colors = len(self.hue_names)

        # Determine the main colors
        if color is None and palette is None:
            # Determine whether the current palette will have enough values
            # If not, we'll default to the husl palette so each is distinct
            current_palette = utils.get_color_cycle()
            if n_colors <= len(current_palette):
                colors = color_palette(n_colors=n_colors)
            else:
                colors = husl_palette(n_colors, l=.7)  # noqa

        elif palette is None:
            # When passing a specific color, the interpretation depends
            # on whether there is a hue variable or not.
            # If so, we will make a blend palette so that the different
            # levels have some amount of variation.
            if self.hue_names is None:
                colors = [color] * n_colors
            else:
                if self.default_palette == "light":
                    colors = light_palette(color, n_colors)
                elif self.default_palette == "dark":
                    colors = dark_palette(color, n_colors)
                else:
                    raise RuntimeError("No default palette specified")
        else:

            # Let `palette` be a dict mapping level to color
            if isinstance(palette, dict):
                if self.hue_names is None:
                    levels = self.group_names
                else:
                    levels = self.hue_names
                palette = [palette[l] for l in levels]

            colors = color_palette(palette, n_colors)

        # Desaturate a bit because these are patches
        if saturation < 1:
            colors = color_palette(colors, desat=saturation)

        # Convert the colors to a common representations
        rgb_colors = color_palette(colors)

        # Determine the gray color to use for the lines framing the plot
        light_vals = [rgb_to_hls(*c)[1] for c in rgb_colors]
        lum = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((lum, lum, lum))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray

    @property
    def hue_offsets(self):
        """A list of center positions for plots when hue nesting is used."""
        n_levels = len(self.hue_names)
        if self.dodge:
            each_width = self.width / n_levels
            offsets = np.linspace(0, self.width - each_width, n_levels)
            offsets -= offsets.mean()
        else:
            offsets = np.zeros(n_levels)

        return offsets

    @property
    def nested_width(self):
        """A float with the width of plot elements when hue nesting is used."""
        if self.dodge:
            width = self.width / len(self.hue_names) * .98
        else:
            width = self.width
        return width

    def annotate_axes(self, ax):
        """Add descriptive labels to an Axes object."""
        if self.orient == "x":
            xlabel, ylabel = self.group_label, self.value_label
        else:
            xlabel, ylabel = self.value_label, self.group_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        group_names = self.group_names
        if not group_names:
            group_names = ["" for _ in range(len(self.plot_data))]

        if self.orient == "x":
            ax.set_xticks(np.arange(len(self.plot_data)))
            ax.set_xticklabels(group_names)
        else:
            ax.set_yticks(np.arange(len(self.plot_data)))
            ax.set_yticklabels(group_names)

        if self.orient == "x":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, len(self.plot_data) - .5, auto=None)
        else:
            ax.yaxis.grid(False)
            ax.set_ylim(-.5, len(self.plot_data) - .5, auto=None)

        if self.hue_names is not None:
            ax.legend(loc="best", title=self.hue_title)

    def add_legend_data(self, ax, color, label):
        """Add a dummy patch object so we can get legend data."""
        rect = plt.Rectangle([0, 0], 0, 0,
                             linewidth=self.linewidth / 2,
                             edgecolor=self.gray,
                             facecolor=color,
                             label=label)
        ax.add_patch(rect)


class _BoxPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 orient, color, palette, saturation,
                 width, dodge, fliersize, linewidth):

        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)

        self.dodge = dodge
        self.width = width
        self.fliersize = fliersize

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

    def draw_boxplot(self, ax, kws):
        """Use matplotlib to draw a boxplot on an Axes."""
        vert = self.orient == "x"

        props = {}
        for obj in ["box", "whisker", "cap", "median", "flier"]:
            props[obj] = kws.pop(obj + "props", {})

        for i, group_data in enumerate(self.plot_data):

            if self.plot_hues is None:

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                # Draw a single box or a set of boxes
                # with a single level of grouping
                box_data = np.asarray(remove_na(group_data))

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                artist_dict = ax.boxplot(box_data,
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[i],
                                         widths=self.width,
                                         **kws)
                color = self.colors[i]
                self.restyle_boxplot(artist_dict, color, props)
            else:
                # Draw nested groups of boxes
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):

                    # Add a legend for this hue level
                    if not i:
                        self.add_legend_data(ax, self.colors[j], hue_level)

                    # Handle case where there is data at this level
                    if group_data.size == 0:
                        continue

                    hue_mask = self.plot_hues[i] == hue_level
                    box_data = np.asarray(remove_na(group_data[hue_mask]))

                    # Handle case where there is no non-null data
                    if box_data.size == 0:
                        continue

                    center = i + offsets[j]
                    artist_dict = ax.boxplot(box_data,
                                             vert=vert,
                                             patch_artist=True,
                                             positions=[center],
                                             widths=self.nested_width,
                                             **kws)
                    self.restyle_boxplot(artist_dict, self.colors[j], props)
                    # Add legend data, but just for one set of boxes

    def restyle_boxplot(self, artist_dict, color, props):
        """Take a drawn matplotlib boxplot and make it look nice."""
        for box in artist_dict["boxes"]:
            box.update(dict(facecolor=color,
                            zorder=.9,
                            edgecolor=self.gray,
                            linewidth=self.linewidth))
            box.update(props["box"])
        for whisk in artist_dict["whiskers"]:
            whisk.update(dict(color=self.gray,
                              linewidth=self.linewidth,
                              linestyle="-"))
            whisk.update(props["whisker"])
        for cap in artist_dict["caps"]:
            cap.update(dict(color=self.gray,
                            linewidth=self.linewidth))
            cap.update(props["cap"])
        for med in artist_dict["medians"]:
            med.update(dict(color=self.gray,
                            linewidth=self.linewidth))
            med.update(props["median"])
        for fly in artist_dict["fliers"]:
            fly.update(dict(markerfacecolor=self.gray,
                            marker="d",
                            markeredgecolor=self.gray,
                            markersize=self.fliersize))
            fly.update(props["flier"])

    def plot(self, ax, boxplot_kws):
        """Make the plot."""
        self.draw_boxplot(ax, boxplot_kws)
        self.annotate_axes(ax)
        if self.orient == "y":
            ax.invert_yaxis()


class _ViolinPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 bw, cut, scale, scale_hue, gridsize,
                 width, inner, split, dodge, orient, linewidth,
                 color, palette, saturation):

        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)
        self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge

        if inner is not None:
            if not any([inner.startswith("quart"),
                        inner.startswith("box"),
                        inner.startswith("stick"),
                        inner.startswith("point")]):
                err = f"Inner style '{inner}' not recognized"
                raise ValueError(err)
        self.inner = inner

        if split and self.hue_names is not None and len(self.hue_names) != 2:
            msg = "There must be exactly two hue levels to use `split`.'"
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

    def estimate_densities(self, bw, cut, scale, scale_hue, gridsize):
        """Find the support and density for all of the data."""
        # Initialize data structures to keep track of plotting data
        if self.hue_names is None:
            support = []
            density = []
            counts = np.zeros(len(self.plot_data))
            max_density = np.zeros(len(self.plot_data))
        else:
            support = [[] for _ in self.plot_data]
            density = [[] for _ in self.plot_data]
            size = len(self.group_names), len(self.hue_names)
            counts = np.zeros(size)
            max_density = np.zeros(size)

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single level of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                # Strip missing datapoints
                kde_data = remove_na(group_data)

                # Handle special case of no data at this level
                if kde_data.size == 0:
                    support.append(np.array([]))
                    density.append(np.array([1.]))
                    counts[i] = 0
                    max_density[i] = 0
                    continue

                # Handle special case of a single unique datapoint
                elif np.unique(kde_data).size == 1:
                    support.append(np.unique(kde_data))
                    density.append(np.array([1.]))
                    counts[i] = 1
                    max_density[i] = 0
                    continue

                # Fit the KDE and get the used bandwidth size
                kde, bw_used = self.fit_kde(kde_data, bw)

                # Determine the support grid and get the density over it
                support_i = self.kde_support(kde_data, bw_used, cut, gridsize)
                density_i = kde.evaluate(support_i)

                # Update the data structures with these results
                support.append(support_i)
                density.append(density_i)
                counts[i] = kde_data.size
                max_density[i] = density_i.max()

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                for j, hue_level in enumerate(self.hue_names):

                    # Handle special case of no data at this category level
                    if not group_data.size:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Select out the observations for this hue level
                    hue_mask = self.plot_hues[i] == hue_level

                    # Strip missing datapoints
                    kde_data = remove_na(group_data[hue_mask])

                    # Handle special case of no data at this level
                    if kde_data.size == 0:
                        support[i].append(np.array([]))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 0
                        max_density[i, j] = 0
                        continue

                    # Handle special case of a single unique datapoint
                    elif np.unique(kde_data).size == 1:
                        support[i].append(np.unique(kde_data))
                        density[i].append(np.array([1.]))
                        counts[i, j] = 1
                        max_density[i, j] = 0
                        continue

                    # Fit the KDE and get the used bandwidth size
                    kde, bw_used = self.fit_kde(kde_data, bw)

                    # Determine the support grid and get the density over it
                    support_ij = self.kde_support(kde_data, bw_used,
                                                  cut, gridsize)
                    density_ij = kde.evaluate(support_ij)

                    # Update the data structures with these results
                    support[i].append(support_ij)
                    density[i].append(density_ij)
                    counts[i, j] = kde_data.size
                    max_density[i, j] = density_ij.max()

        # Scale the height of the density curve.
        # For a violinplot the density is non-quantitative.
        # The objective here is to scale the curves relative to 1 so that
        # they can be multiplied by the width parameter during plotting.

        if scale == "area":
            self.scale_area(density, max_density, scale_hue)

        elif scale == "width":
            self.scale_width(density)

        elif scale == "count":
            self.scale_count(density, counts, scale_hue)

        else:
            raise ValueError(f"scale method '{scale}' not recognized")

        # Set object attributes that will be used while plotting
        self.support = support
        self.density = density

    def fit_kde(self, x, bw):
        """Estimate a KDE for a vector of data with flexible bandwidth."""
        kde = gaussian_kde(x, bw)

        # Extract the numeric bandwidth from the KDE object
        bw_used = kde.factor

        # At this point, bw will be a numeric scale factor.
        # To get the actual bandwidth of the kernel, we multiple by the
        # unbiased standard deviation of the data, which we will use
        # elsewhere to compute the range of the support.
        bw_used = bw_used * x.std(ddof=1)

        return kde, bw_used

    def kde_support(self, x, bw, cut, gridsize):
        """Define a grid of support for the violin."""
        support_min = x.min() - bw * cut
        support_max = x.max() + bw * cut
        return np.linspace(support_min, support_max, gridsize)

    def scale_area(self, density, max_density, scale_hue):
        """Scale the relative area under the KDE curve.

        This essentially preserves the "standard" KDE scaling, but the
        resulting maximum density will be 1 so that the curve can be
        properly multiplied by the violin width.

        """
        if self.hue_names is None:
            for d in density:
                if d.size > 1:
                    d /= max_density.max()
        else:
            for i, group in enumerate(density):
                for d in group:
                    if scale_hue:
                        max = max_density[i].max()
                    else:
                        max = max_density.max()
                    if d.size > 1:
                        d /= max

    def scale_width(self, density):
        """Scale each density curve to the same height."""
        if self.hue_names is None:
            for d in density:
                d /= d.max()
        else:
            for group in density:
                for d in group:
                    d /= d.max()

    def scale_count(self, density, counts, scale_hue):
        """Scale each density curve by the number of observations."""
        if self.hue_names is None:
            if counts.max() == 0:
                d = 0
            else:
                for count, d in zip(counts, density):
                    d /= d.max()
                    d *= count / counts.max()
        else:
            for i, group in enumerate(density):
                for j, d in enumerate(group):
                    if counts[i].max() == 0:
                        d = 0
                    else:
                        count = counts[i, j]
                        if scale_hue:
                            scaler = count / counts[i].max()
                        else:
                            scaler = count / counts.max()
                        d /= d.max()
                        d *= scaler

    @property
    def dwidth(self):

        if self.hue_names is None or not self.dodge:
            return self.width / 2
        elif self.split:
            return self.width / 2
        else:
            return self.width / (2 * len(self.hue_names))

    def draw_violins(self, ax):
        """Draw the violins onto `ax`."""
        fill_func = ax.fill_betweenx if self.orient == "x" else ax.fill_between
        for i, group_data in enumerate(self.plot_data):

            kws = dict(edgecolor=self.gray, linewidth=self.linewidth)

            # Option 1: we have a single level of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                support, density = self.support[i], self.density[i]

                # Handle special case of no observations in this bin
                if support.size == 0:
                    continue

                # Handle special case of a single observation
                elif support.size == 1:
                    val = support.item()
                    d = density.item()
                    self.draw_single_observation(ax, i, val, d)
                    continue

                # Draw the violin for this group
                grid = np.ones(self.gridsize) * i
                fill_func(support,
                          grid - density * self.dwidth,
                          grid + density * self.dwidth,
                          facecolor=self.colors[i],
                          **kws)

                # Draw the interior representation of the data
                if self.inner is None:
                    continue

                # Get a nan-free vector of datapoints
                violin_data = remove_na(group_data)

                # Draw box and whisker information
                if self.inner.startswith("box"):
                    self.draw_box_lines(ax, violin_data, i)

                # Draw quartile lines
                elif self.inner.startswith("quart"):
                    self.draw_quartiles(ax, violin_data, support, density, i)

                # Draw stick observations
                elif self.inner.startswith("stick"):
                    self.draw_stick_lines(ax, violin_data, support, density, i)

                # Draw point observations
                elif self.inner.startswith("point"):
                    self.draw_points(ax, violin_data, i)

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):

                    support, density = self.support[i][j], self.density[i][j]
                    kws["facecolor"] = self.colors[j]

                    # Add legend data, but just for one set of violins
                    if not i:
                        self.add_legend_data(ax, self.colors[j], hue_level)

                    # Handle the special case where we have no observations
                    if support.size == 0:
                        continue

                    # Handle the special case where we have one observation
                    elif support.size == 1:
                        val = support.item()
                        d = density.item()
                        if self.split:
                            d = d / 2
                        at_group = i + offsets[j]
                        self.draw_single_observation(ax, at_group, val, d)
                        continue

                    # Option 2a: we are drawing a single split violin
                    # -----------------------------------------------

                    if self.split:

                        grid = np.ones(self.gridsize) * i
                        if j:
                            fill_func(support,
                                      grid,
                                      grid + density * self.dwidth,
                                      **kws)
                        else:
                            fill_func(support,
                                      grid - density * self.dwidth,
                                      grid,
                                      **kws)

                        # Draw the interior representation of the data
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])

                        # Draw quartile lines
                        if self.inner.startswith("quart"):
                            self.draw_quartiles(ax, violin_data,
                                                support, density, i,
                                                ["left", "right"][j])

                        # Draw stick observations
                        elif self.inner.startswith("stick"):
                            self.draw_stick_lines(ax, violin_data,
                                                  support, density, i,
                                                  ["left", "right"][j])

                        # The box and point interior plots are drawn for
                        # all data at the group level, so we just do that once
                        if j and any(self.plot_hues[0] == hue_level):
                            continue

                        # Get the whole vector for this group level
                        violin_data = remove_na(group_data)

                        # Draw box and whisker information
                        if self.inner.startswith("box"):
                            self.draw_box_lines(ax, violin_data, i)

                        # Draw point observations
                        elif self.inner.startswith("point"):
                            self.draw_points(ax, violin_data, i)

                    # Option 2b: we are drawing full nested violins
                    # -----------------------------------------------

                    else:
                        grid = np.ones(self.gridsize) * (i + offsets[j])
                        fill_func(support,
                                  grid - density * self.dwidth,
                                  grid + density * self.dwidth,
                                  **kws)

                        # Draw the interior representation
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])

                        # Draw box and whisker information
                        if self.inner.startswith("box"):
                            self.draw_box_lines(ax, violin_data, i + offsets[j])

                        # Draw quartile lines
                        elif self.inner.startswith("quart"):
                            self.draw_quartiles(ax, violin_data,
                                                support, density,
                                                i + offsets[j])

                        # Draw stick observations
                        elif self.inner.startswith("stick"):
                            self.draw_stick_lines(ax, violin_data,
                                                  support, density,
                                                  i + offsets[j])

                        # Draw point observations
                        elif self.inner.startswith("point"):
                            self.draw_points(ax, violin_data, i + offsets[j])

    def draw_single_observation(self, ax, at_group, at_quant, density):
        """Draw a line to mark a single observation."""
        d_width = density * self.dwidth
        if self.orient == "x":
            ax.plot([at_group - d_width, at_group + d_width],
                    [at_quant, at_quant],
                    color=self.gray,
                    linewidth=self.linewidth)
        else:
            ax.plot([at_quant, at_quant],
                    [at_group - d_width, at_group + d_width],
                    color=self.gray,
                    linewidth=self.linewidth)

    def draw_box_lines(self, ax, data, center):
        """Draw boxplot information at center of the density."""
        # Compute the boxplot statistics
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * (q75 - q25)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])

        # Draw a boxplot using lines and a point
        if self.orient == "x":
            ax.plot([center, center], [h1, h2],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([center, center], [q25, q75],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(center, q50,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))
        else:
            ax.plot([h1, h2], [center, center],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([q25, q75], [center, center],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(q50, center,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))

    def draw_quartiles(self, ax, data, support, density, center, split=False):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])

        self.draw_to_density(ax, center, q25, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)
        self.draw_to_density(ax, center, q50, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 3] * 2)
        self.draw_to_density(ax, center, q75, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)

    def draw_points(self, ax, data, center):
        """Draw individual observations as points at middle of the violin."""
        kws = dict(s=np.square(self.linewidth * 2),
                   color=self.gray,
                   edgecolor=self.gray)

        grid = np.ones(len(data)) * center

        if self.orient == "x":
            ax.scatter(grid, data, **kws)
        else:
            ax.scatter(data, grid, **kws)

    def draw_stick_lines(self, ax, data, support, density,
                         center, split=False):
        """Draw individual observations as sticks at width of density."""
        for val in data:
            self.draw_to_density(ax, center, val, support, density, split,
                                 linewidth=self.linewidth * .5)

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * .99

        kws["color"] = self.gray

        if self.orient == "x":
            if split == "left":
                ax.plot([center - width, center], [val, val], **kws)
            elif split == "right":
                ax.plot([center, center + width], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width, center], **kws)
            elif split == "right":
                ax.plot([val, val], [center, center + width], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def plot(self, ax):
        """Make the violin plot."""
        self.draw_violins(ax)
        self.annotate_axes(ax)
        if self.orient == "y":
            ax.invert_yaxis()


class _LVPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 orient, color, palette, saturation,
                 width, dodge, k_depth, linewidth, scale, outlier_prop,
                 trust_alpha, showfliers=True):

        self.width = width
        self.dodge = dodge
        self.saturation = saturation

        k_depth_methods = ['proportion', 'tukey', 'trustworthy', 'full']
        if not (k_depth in k_depth_methods or isinstance(k_depth, Number)):
            msg = (f'k_depth must be one of {k_depth_methods} or a number, '
                   f'but {k_depth} was passed.')
            raise ValueError(msg)
        self.k_depth = k_depth

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

        scales = ['linear', 'exponential', 'area']
        if scale not in scales:
            msg = f'scale must be one of {scales}, but {scale} was passed.'
            raise ValueError(msg)
        self.scale = scale

        if ((outlier_prop > 1) or (outlier_prop <= 0)):
            msg = f'outlier_prop {outlier_prop} not in range (0, 1]'
            raise ValueError(msg)
        self.outlier_prop = outlier_prop

        if not 0 < trust_alpha < 1:
            msg = f'trust_alpha {trust_alpha} not in range (0, 1) '
            raise ValueError(msg)
        self.trust_alpha = trust_alpha

        self.showfliers = showfliers

        self.establish_variables(x, y, hue, data, orient, order, hue_order)
        self.establish_colors(color, palette, saturation)

    def _lv_box_ends(self, vals):
        """Get the number of data points and calculate `depth` of
        letter-value plot."""
        vals = np.asarray(vals)
        # Remove infinite values while handling a 'object' dtype
        # that can come from pd.Float64Dtype() input
        with pd.option_context('mode.use_inf_as_na', True):
            vals = vals[~pd.isnull(vals)]
        n = len(vals)
        p = self.outlier_prop

        # Select the depth, i.e. number of boxes to draw, based on the method
        if self.k_depth == 'full':
            # extend boxes to 100% of the data
            k = int(np.log2(n)) + 1
        elif self.k_depth == 'tukey':
            # This results with 5-8 points in each tail
            k = int(np.log2(n)) - 3
        elif self.k_depth == 'proportion':
            k = int(np.log2(n)) - int(np.log2(n * p)) + 1
        elif self.k_depth == 'trustworthy':
            point_conf = 2 * _normal_quantile_func(1 - self.trust_alpha / 2) ** 2
            k = int(np.log2(n / point_conf)) + 1
        else:
            k = int(self.k_depth)  # allow having k as input
        # If the number happens to be less than 1, set k to 1
        if k < 1:
            k = 1

        # Calculate the upper end for each of the k boxes
        upper = [100 * (1 - 0.5 ** (i + 1)) for i in range(k, 0, -1)]
        # Calculate the lower end for each of the k boxes
        lower = [100 * (0.5 ** (i + 1)) for i in range(k, 0, -1)]
        # Stitch the box ends together
        percentile_ends = [(i, j) for i, j in zip(lower, upper)]
        box_ends = [np.percentile(vals, q) for q in percentile_ends]
        return box_ends, k

    def _lv_outliers(self, vals, k):
        """Find the outliers based on the letter value depth."""
        box_edge = 0.5 ** (k + 1)
        perc_ends = (100 * box_edge, 100 * (1 - box_edge))
        edges = np.percentile(vals, perc_ends)
        lower_out = vals[np.where(vals < edges[0])[0]]
        upper_out = vals[np.where(vals > edges[1])[0]]
        return np.concatenate((lower_out, upper_out))

    def _width_functions(self, width_func):
        # Dictionary of functions for computing the width of the boxes
        width_functions = {'linear': lambda h, i, k: (i + 1.) / k,
                           'exponential': lambda h, i, k: 2**(-k + i - 1),
                           'area': lambda h, i, k: (1 - 2**(-k + i - 2)) / h}
        return width_functions[width_func]

    def _lvplot(self, box_data, positions,
                color=[255. / 256., 185. / 256., 0.],
                widths=1, ax=None, box_kws=None,
                flier_kws=None,
                line_kws=None):

        # -- Default keyword dicts - based on
        # distributions.plot_univariate_histogram
        box_kws = {} if box_kws is None else box_kws.copy()
        flier_kws = {} if flier_kws is None else flier_kws.copy()
        line_kws = {} if line_kws is None else line_kws.copy()

        # Set the default kwargs for the boxes
        box_default_kws = dict(edgecolor=self.gray,
                               linewidth=self.linewidth)
        for k, v in box_default_kws.items():
            box_kws.setdefault(k, v)

        # Set the default kwargs for the lines denoting medians
        line_default_kws = dict(
            color=".15", alpha=0.45, solid_capstyle="butt", linewidth=self.linewidth
        )
        for k, v in line_default_kws.items():
            line_kws.setdefault(k, v)

        # Set the default kwargs for the outliers scatterplot
        flier_default_kws = dict(marker='d', color=self.gray)
        for k, v in flier_default_kws.items():
            flier_kws.setdefault(k, v)

        vert = self.orient == "x"
        x = positions[0]
        box_data = np.asarray(box_data)

        # If we only have one data point, plot a line
        if len(box_data) == 1:
            line_kws.update({
                'color': box_kws['edgecolor'],
                'linestyle': box_kws.get('linestyle', '-'),
                'linewidth': max(box_kws["linewidth"], line_kws["linewidth"])
            })
            ys = [box_data[0], box_data[0]]
            xs = [x - widths / 2, x + widths / 2]
            if vert:
                xx, yy = xs, ys
            else:
                xx, yy = ys, xs
            ax.plot(xx, yy, **line_kws)
        else:
            # Get the number of data points and calculate "depth" of
            # letter-value plot
            box_ends, k = self._lv_box_ends(box_data)

            # Anonymous functions for calculating the width and height
            # of the letter value boxes
            width = self._width_functions(self.scale)

            # Function to find height of boxes
            def height(b):
                return b[1] - b[0]

            # Functions to construct the letter value boxes
            def vert_perc_box(x, b, i, k, w):
                rect = Patches.Rectangle((x - widths * w / 2, b[0]),
                                         widths * w,
                                         height(b), fill=True)
                return rect

            def horz_perc_box(x, b, i, k, w):
                rect = Patches.Rectangle((b[0], x - widths * w / 2),
                                         height(b), widths * w,
                                         fill=True)
                return rect

            # Scale the width of the boxes so the biggest starts at 1
            w_area = np.array([width(height(b), i, k)
                               for i, b in enumerate(box_ends)])
            w_area = w_area / np.max(w_area)

            # Calculate the medians
            y = np.median(box_data)

            # Calculate the outliers and plot (only if showfliers == True)
            outliers = []
            if self.showfliers:
                outliers = self._lv_outliers(box_data, k)
            hex_color = mpl.colors.rgb2hex(color)

            if vert:
                box_func = vert_perc_box
                xs_median = [x - widths / 2, x + widths / 2]
                ys_median = [y, y]
                xs_outliers = np.full(len(outliers), x)
                ys_outliers = outliers

            else:
                box_func = horz_perc_box
                xs_median = [y, y]
                ys_median = [x - widths / 2, x + widths / 2]
                xs_outliers = outliers
                ys_outliers = np.full(len(outliers), x)

            # Plot the medians
            ax.plot(
                xs_median,
                ys_median,
                **line_kws
            )

            # Plot outliers (if any)
            if len(outliers) > 0:
                ax.scatter(xs_outliers, ys_outliers,
                           **flier_kws
                           )

            # Construct a color map from the input color
            rgb = [hex_color, (1, 1, 1)]
            cmap = mpl.colors.LinearSegmentedColormap.from_list('new_map', rgb)
            # Make sure that the last boxes contain hue and are not pure white
            rgb = [hex_color, cmap(.85)]
            cmap = mpl.colors.LinearSegmentedColormap.from_list('new_map', rgb)

            # Update box_kws with `cmap` if not defined in dict until now
            box_kws.setdefault('cmap', cmap)

            boxes = [box_func(x, b[0], i, k, b[1])
                     for i, b in enumerate(zip(box_ends, w_area))]

            collection = PatchCollection(boxes, **box_kws)

            # Set the color gradation, first box will have color=hex_color
            collection.set_array(np.array(np.linspace(1, 0, len(boxes))))

            # Plot the boxes
            ax.add_collection(collection)

    def draw_letter_value_plot(self, ax, box_kws=None, flier_kws=None,
                               line_kws=None):
        """Use matplotlib to draw a letter value plot on an Axes."""

        for i, group_data in enumerate(self.plot_data):

            if self.plot_hues is None:

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                # Draw a single box or a set of boxes
                # with a single level of grouping
                box_data = remove_na(group_data)

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                color = self.colors[i]

                self._lvplot(box_data,
                             positions=[i],
                             color=color,
                             widths=self.width,
                             ax=ax,
                             box_kws=box_kws,
                             flier_kws=flier_kws,
                             line_kws=line_kws)

            else:
                # Draw nested groups of boxes
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):

                    # Add a legend for this hue level
                    if not i:
                        self.add_legend_data(ax, self.colors[j], hue_level)

                    # Handle case where there is data at this level
                    if group_data.size == 0:
                        continue

                    hue_mask = self.plot_hues[i] == hue_level
                    box_data = remove_na(group_data[hue_mask])

                    # Handle case where there is no non-null data
                    if box_data.size == 0:
                        continue

                    color = self.colors[j]
                    center = i + offsets[j]
                    self._lvplot(box_data,
                                 positions=[center],
                                 color=color,
                                 widths=self.nested_width,
                                 ax=ax,
                                 box_kws=box_kws,
                                 flier_kws=flier_kws,
                                 line_kws=line_kws)

        # Autoscale the values axis to make sure all patches are visible
        ax.autoscale_view(scalex=self.orient == "y", scaley=self.orient == "x")

    def plot(self, ax, box_kws, flier_kws, line_kws):
        """Make the plot."""
        self.draw_letter_value_plot(ax, box_kws, flier_kws, line_kws)
        self.annotate_axes(ax)
        if self.orient == "y":
            ax.invert_yaxis()


_categorical_docs = dict(

    # Shared narrative docs
    categorical_narrative=dedent("""\
    .. note::
        This function always treats one of the variables as categorical and
        draws data at ordinal positions (0, 1, ... n) on the relevant axis,
        even when the data has a numeric or date type.

    See the :ref:`tutorial <categorical_tutorial>` for more information.\
    """),

    new_categorical_narrative=dedent("""\
    .. note::
        By default, this function treats one of the variables as categorical
        and draws data at ordinal positions (0, 1, ... n) on the relevant axis.
        As of v0.13, this can be disabled by setting `native_scale=True`.

    See the :ref:`tutorial <categorical_tutorial>` for more information.\
    """),

    # Shared function parameters
    input_params=dedent("""\
    x, y, hue : names of variables in `data` or vector data
        Inputs for plotting long-form data. See examples for interpretation.\
    """),
    string_input_params=dedent("""\
    x, y, hue : names of variables in `data`
        Inputs for plotting long-form data. See examples for interpretation.\
    """),
    categorical_data=dedent("""\
    data : DataFrame, Series, dict, array, or list of arrays
        Dataset for plotting. If `x` and `y` are absent, this is
        interpreted as wide-form. Otherwise it is expected to be long-form.\
    """),
    long_form_data=dedent("""\
    data : DataFrame
        Long-form (tidy) dataset for plotting. Each column should correspond
        to a variable, and each row should correspond to an observation.\
    """),
    order_vars=dedent("""\
    order, hue_order : lists of strings
        Order to plot the categorical levels in; otherwise the levels are
        inferred from the data objects.\
    """),
    stat_api_params=dedent("""\
    estimator : string or callable that maps vector -> scalar
        Statistical function to estimate within each categorical bin.
    errorbar : string, (string, number) tuple, callable or None
        Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
        with a method name and a level parameter, or a function that maps from a
        vector to a (min, max) interval, or None to hide errorbar.

        .. versionadded:: v0.12.0
    n_boot : int
        Number of bootstrap samples used to compute confidence intervals.
    units : name of variable in `data` or vector data
        Identifier of sampling units, which will be used to perform a
        multilevel bootstrap and account for repeated measures design.
    seed : int, `numpy.random.Generator`, or `numpy.random.RandomState`
        Seed or random number generator for reproducible bootstrapping.\
    """),
    ci=dedent("""\
    ci : float
        Level of the confidence interval to show, in [0, 100].

        .. deprecated:: v0.12.0
            Use `errorbar=("ci", ...) `.
    """),
    orient=dedent("""\
    orient : "v" | "h" | "x" | "y"
        Orientation of the plot (vertical or horizontal). This is usually
        inferred based on the type of the input variables, but it can be used
        to resolve ambiguity when both `x` and `y` are numeric or when
        plotting wide-form data.

        .. versionchanged:: v0.13.0
            Added 'x'/'y' as options, equivalent to 'v'/'h'.
    """),
    color=dedent("""\
    color : matplotlib color
        Single color for the elements in the plot.\
    """),
    palette=dedent("""\
    palette : palette name, list, dict, or :class:`matplotlib.colors.Colormap`
        Color palette that maps the hue variable. If the palette is a dictionary,
        keys should be names of levels and values should be matplotlib colors.
        The type/value will sometimes force a qualitative/quantitative mapping.\
    """),
    hue_norm=dedent("""\
    hue_norm : tuple or :class:`matplotlib.colors.Normalize` object
        Normalization in data units for colormap applied to the `hue`
        variable when it is numeric. Not relevant if `hue` is categorical.

        .. versionadded:: v0.12.0\
    """),
    saturation=dedent("""\
    saturation : float
        Proportion of the original saturation to draw colors at. Large patches
        often look better with slightly desaturated colors, but set this to
        `1` if you want the plot colors to perfectly match the input color.\
    """),
    capsize=dedent("""\
    capsize : float
        Width of the "caps" on error bars, relative to bar spacing.\
    """),
    errcolor=dedent("""\
    errcolor : matplotlib color
        Color used for the error bar lines.

        .. deprecated:: 0.13.0
            Use `err_kws={'color': ...}`.\
    """),
    errwidth=dedent("""\
    errwidth : float
        Thickness of error bar lines (and caps), in points.

        .. deprecated:: 0.13.0
            Use `err_kws={'linewidth': ...}`.\
    """),
    width=dedent("""\
    width : float
        Width of a full element when not using hue nesting, or width of all the
        elements for one level of the major grouping variable.\
    """),
    dodge=dedent("""\
    dodge : bool
        When hue nesting is used, whether elements should be shifted along the
        categorical axis.\
    """),
    linewidth=dedent("""\
    linewidth : float
        Width of the gray lines that frame the plot elements.\
    """),
    native_scale=dedent("""\
    native_scale : bool
        When True, numeric or datetime values on the categorical axis will maintain
        their original scaling rather than being converted to fixed indices.

        .. versionadded:: v0.13.0\
    """),
    formatter=dedent("""\
    formatter : callable
        Function for converting categorical data into strings. Affects both grouping
        and tick labels.

        .. versionadded:: v0.13.0\
    """),
    legend=dedent("""\
    legend : "auto", "brief", "full", or False
        How to draw the legend. If "brief", numeric `hue` and `size`
        variables will be represented with a sample of evenly spaced values.
        If "full", every group will get an entry in the legend. If "auto",
        choose between brief or full representation based on number of levels.
        If `False`, no legend data is added and no legend is drawn.

        .. versionadded:: v0.13.0\
    """),
    err_kws=dedent("""\
    err_kws : dict
        Parameters of :class:`matplotlib.lines.Line2D`, for the error bar artists.

        .. versionadded:: v0.13.0\
    """),
    ax_in=dedent("""\
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
    ax_out=dedent("""\
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
    """),

    # Shared see also
    boxplot=dedent("""\
    boxplot : A traditional box-and-whisker plot with a similar API.\
    """),
    violinplot=dedent("""\
    violinplot : A combination of boxplot and kernel density estimation.\
    """),
    stripplot=dedent("""\
    stripplot : A scatterplot where one variable is categorical. Can be used
                in conjunction with other plots to show each observation.\
    """),
    swarmplot=dedent("""\
    swarmplot : A categorical scatterplot where the points do not overlap. Can
                be used with other plots to show each observation.\
    """),
    barplot=dedent("""\
    barplot : Show point estimates and confidence intervals using bars.\
    """),
    countplot=dedent("""\
    countplot : Show the counts of observations in each categorical bin.\
    """),
    pointplot=dedent("""\
    pointplot : Show point estimates and confidence intervals using scatterplot
                glyphs.\
    """),
    catplot=dedent("""\
    catplot : Combine a categorical plot with a :class:`FacetGrid`.\
    """),
    boxenplot=dedent("""\
    boxenplot : An enhanced boxplot for larger datasets.\
    """),

)

_categorical_docs.update(_facet_docs)


def boxplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75, width=.8,
    gap=0,  # TODO new, document, place next to dodge and width, add to bar?
    fill=True,  # TODO new, document
    linecolor=None,  # TODO new, document
    linewidth=None, fliersize=None,
    dodge="auto", hue_norm=None, whis=1.5,
    native_scale=False, formatter=None, legend="auto",
    ax=None,
    **kwargs
):

    p = _CategoricalPlotterNew(
        data=data,
        variables=_CategoricalPlotterNew.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if dodge == "auto":
        # Needs to be before scale_categorical changes the coordinate series dtype
        dodge = p._dodge_needed()

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    color = _default_color(
        ax.fill_between, hue, color,
        {k: v for k, v in kwargs.items() if k in ["c", "color", "fc", "facecolor"]},
    )
    if color is not None and saturation < 1:
        color = desaturate(color, saturation)

    p.plot_boxes(
        width=width,
        dodge=dodge,
        gap=gap,
        fill=fill,
        whis=whis,
        color=color,
        linecolor=linecolor,
        linewidth=linewidth,
        fliersize=fliersize,
        saturation=saturation,
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


boxplot.__doc__ = dedent("""\
    Draw a box plot to show distributions with respect to categories.

    A box plot (or box-and-whisker plot) shows the distribution of quantitative
    data in a way that facilitates comparisons between variables or across
    levels of a categorical variable. The box shows the quartiles of the
    dataset while the whiskers extend to show the rest of the distribution,
    except for points that are determined to be "outliers" using a method
    that is a function of the inter-quartile range.

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {orient}
    {color}
    {palette}
    {saturation}
    {width}
    {dodge}
    fliersize : float
        Size of the markers used to indicate outlier observations.
    {linewidth}
    whis : float
        Maximum length of the plot whiskers as proportion of the
        interquartile range. Whiskers extend to the furthest datapoint
        within that range. More extreme points are marked as outliers.
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.boxplot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {violinplot}
    {stripplot}
    {swarmplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/boxplot.rst

    """).format(**_categorical_docs)


def violinplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
    width=.8, inner="box", split=False, dodge=True, orient=None,
    linewidth=None, color=None, palette=None, saturation=.75,
    ax=None, **kwargs,
):

    plotter = _ViolinPlotter(x, y, hue, data, order, hue_order,
                             bw, cut, scale, scale_hue, gridsize,
                             width, inner, split, dodge, orient, linewidth,
                             color, palette, saturation)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax


violinplot.__doc__ = dedent("""\
    Draw a combination of boxplot and kernel density estimate.

    A violin plot plays a similar role as a box and whisker plot. It shows the
    distribution of quantitative data across several levels of one (or more)
    categorical variables such that those distributions can be compared. Unlike
    a box plot, in which all of the plot components correspond to actual
    datapoints, the violin plot features a kernel density estimation of the
    underlying distribution.

    This can be an effective and attractive way to show multiple distributions
    of data at once, but keep in mind that the estimation procedure is
    influenced by the sample size, and violins for relatively small samples
    might look misleadingly smooth.

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    bw : {{'scott', 'silverman', float}}
        Either the name of a reference rule or the scale factor to use when
        computing the kernel bandwidth. The actual kernel size will be
        determined by multiplying the scale factor by the standard deviation of
        the data within each bin.
    cut : float
        Distance, in units of bandwidth size, to extend the density past the
        extreme datapoints. Set to 0 to limit the violin range within the range
        of the observed data (i.e., to have the same effect as ``trim=True`` in
        ``ggplot``.
    scale : {{"area", "count", "width"}}
        The method used to scale the width of each violin. If ``area``, each
        violin will have the same area. If ``count``, the width of the violins
        will be scaled by the number of observations in that bin. If ``width``,
        each violin will have the same width.
    scale_hue : bool
        When nesting violins using a ``hue`` variable, this parameter
        determines whether the scaling is computed within each level of the
        major grouping variable (``scale_hue=True``) or across all the violins
        on the plot (``scale_hue=False``).
    gridsize : int
        Number of points in the discrete grid used to compute the kernel
        density estimate.
    {width}
    inner : {{"box", "quartile", "point", "stick", None}}
        Representation of the datapoints in the violin interior. If ``box``,
        draw a miniature boxplot. If ``quartiles``, draw the quartiles of the
        distribution.  If ``point`` or ``stick``, show each underlying
        datapoint. Using ``None`` will draw unadorned violins.
    split : bool
        When using hue nesting with a variable that takes two levels, setting
        ``split`` to True will draw half of a violin for each level. This can
        make it easier to directly compare the distributions.
    {dodge}
    {orient}
    {linewidth}
    {color}
    {palette}
    {saturation}
    {ax_in}

    Returns
    -------
    {ax_out}

    See Also
    --------
    {boxplot}
    {stripplot}
    {swarmplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/violinplot.rst

    """).format(**_categorical_docs)


def boxenplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75,
    width=.8, dodge=True, k_depth='tukey', linewidth=None,
    scale='exponential', outlier_prop=0.007, trust_alpha=0.05,
    showfliers=True,
    ax=None, box_kws=None, flier_kws=None, line_kws=None,
):
    plotter = _LVPlotter(x, y, hue, data, order, hue_order,
                         orient, color, palette, saturation,
                         width, dodge, k_depth, linewidth, scale,
                         outlier_prop, trust_alpha, showfliers)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, box_kws, flier_kws, line_kws)
    return ax


boxenplot.__doc__ = dedent("""\
    Draw an enhanced box plot for larger datasets.

    This style of plot was originally named a "letter value" plot because it
    shows a large number of quantiles that are defined as "letter values".  It
    is similar to a box plot in plotting a nonparametric representation of a
    distribution in which all features correspond to actual observations. By
    plotting more quantiles, it provides more information about the shape of
    the distribution, particularly in the tails. For a more extensive
    explanation, you can read the paper that introduced the plot:
    https://vita.had.co.nz/papers/letter-value-plot.html

    {categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {orient}
    {color}
    {palette}
    {saturation}
    {width}
    {dodge}
    k_depth : {{"tukey", "proportion", "trustworthy", "full"}} or scalar
        The number of boxes, and by extension number of percentiles, to draw.
        All methods are detailed in Wickham's paper. Each makes different
        assumptions about the number of outliers and leverages different
        statistical properties. If "proportion", draw no more than
        `outlier_prop` extreme observations. If "full", draw `log(n) +1` boxes.
    {linewidth}
    scale : {{"exponential", "linear", "area"}}
        Method to use for the width of the letter value boxes. All give similar
        results visually. "linear" reduces the width by a constant linear
        factor, "exponential" uses the proportion of data not covered, "area"
        is proportional to the percentage of data covered.
    outlier_prop : float
        Proportion of data believed to be outliers. Must be in the range
        (0, 1]. Used to determine the number of boxes to plot when
        `k_depth="proportion"`.
    trust_alpha : float
        Confidence level for a box to be plotted. Used to determine the
        number of boxes to plot when `k_depth="trustworthy"`. Must be in the
        range (0, 1).
    showfliers : bool
        If False, suppress the plotting of outliers.
    {ax_in}
    box_kws: dict
        Keyword arguments for the box artists; passed to
        :class:`matplotlib.patches.Rectangle`.
    line_kws: dict
        Keyword arguments for the line denoting the median; passed to
        :meth:`matplotlib.axes.Axes.plot`.
    flier_kws: dict
        Keyword arguments for the scatter denoting the outlier observations;
        passed to :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {violinplot}
    {boxplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/boxenplot.rst

    """).format(**_categorical_docs)


def stripplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    jitter=True, dodge=False, orient=None, color=None, palette=None,
    size=5, edgecolor="gray", linewidth=0,
    hue_norm=None, native_scale=False, formatter=None, legend="auto",
    ax=None, **kwargs
):

    p = _CategoricalPlotterNew(
        data=data,
        variables=_CategoricalPlotterNew.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    color = _default_color(ax.scatter, hue, color, kwargs)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # XXX Copying possibly bad default decisions from original code for now
    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)

    kwargs.update(
        s=size ** 2,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    p.plot_strips(
        jitter=jitter,
        dodge=dodge,
        color=color,
        edgecolor=edgecolor,
        plot_kws=kwargs,
    )

    # XXX this happens inside a plotting method in the distribution plots
    # but maybe it's better out here? Alternatively, we have an open issue
    # suggesting that _attach could add default axes labels, which seems smart.
    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


stripplot.__doc__ = dedent("""\
    Draw a categorical scatterplot using jitter to reduce overplotting.

    A strip plot can be drawn on its own, but it is also a good complement
    to a box or violin plot in cases where you want to show all observations
    along with some representation of the underlying distribution.

    {new_categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    jitter : float, ``True``/``1`` is special-cased
        Amount of jitter (only along the categorical axis) to apply. This
        can be useful when you have many points and they overlap, so that
        it is easier to see the distribution. You can specify the amount
        of jitter (half the width of the uniform random variable support),
        or just use ``True`` for a good default.
    dodge : bool
        When using ``hue`` nesting, setting this to ``True`` will separate
        the strips for different hue levels along the categorical axis.
        Otherwise, the points for each level will be plotted on top of
        each other.
    {orient}
    {color}
    {palette}
    size : float
        Radius of the markers, in points.
    edgecolor : matplotlib color, "gray" is special-cased
        Color of the lines around each point. If you pass ``"gray"``, the
        brightness is determined by the color palette used for the body
        of the points. Note that `stripplot` has `linewidth=0` by default,
        so edge colors are only visible with nonzero line width.
    {linewidth}
    {hue_norm}
    {native_scale}
    {formatter}
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {swarmplot}
    {boxplot}
    {violinplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/stripplot.rst

    """).format(**_categorical_docs)


def swarmplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    dodge=False, orient=None, color=None, palette=None,
    size=5, edgecolor="gray", linewidth=0, hue_norm=None,
    native_scale=False, formatter=None, legend="auto", warn_thresh=.05,
    ax=None, **kwargs
):

    p = _CategoricalPlotterNew(
        data=data,
        variables=_CategoricalPlotterNew.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    if not p.has_xy_data:
        return ax

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    color = _default_color(ax.scatter, hue, color, kwargs)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # XXX Copying possibly bad default decisions from original code for now
    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)

    if linewidth is None:
        linewidth = size / 10

    kwargs.update(dict(
        s=size ** 2,
        linewidth=linewidth,
    ))

    p.plot_swarms(
        dodge=dodge,
        color=color,
        edgecolor=edgecolor,
        warn_thresh=warn_thresh,
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


swarmplot.__doc__ = dedent("""\
    Draw a categorical scatterplot with points adjusted to be non-overlapping.

    This function is similar to :func:`stripplot`, but the points are adjusted
    (only along the categorical axis) so that they don't overlap. This gives a
    better representation of the distribution of values, but it does not scale
    well to large numbers of observations. This style of plot is sometimes
    called a "beeswarm".

    A swarm plot can be drawn on its own, but it is also a good complement
    to a box or violin plot in cases where you want to show all observations
    along with some representation of the underlying distribution.

    {new_categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    dodge : bool
        When using ``hue`` nesting, setting this to ``True`` will separate
        the strips for different hue levels along the categorical axis.
        Otherwise, the points for each level will be plotted in one swarm.
    {orient}
    {color}
    {palette}
    size : float
        Radius of the markers, in points.
    edgecolor : matplotlib color, "gray" is special-cased
        Color of the lines around each point. If you pass ``"gray"``, the
        brightness is determined by the color palette used for the body
        of the points.
    {linewidth}
    {native_scale}
    {formatter}
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed through to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {boxplot}
    {violinplot}
    {stripplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/swarmplot.rst

    """).format(**_categorical_docs)


def barplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, units=None, seed=None,
    orient=None, color=None, palette=None, saturation=.75, hue_norm=None, width=.8,
    dodge="auto", native_scale=False, formatter=None, legend="auto", capsize=0,
    err_kws=None, ci=deprecated, errcolor=deprecated, errwidth=deprecated,
    ax=None, **kwargs,
):

    errorbar = utils._deprecate_ci(errorbar, ci)

    # Be backwards compatible with len passed directly, which
    # does not work in Series.agg (maybe a pandas bug?)
    if estimator is len:
        estimator = "size"

    p = _CategoricalAggPlotter(
        data=data,
        variables=_CategoricalAggPlotter.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if dodge == "auto":
        # Needs to be before scale_categorical changes the coordinate series dtype
        dodge = p._dodge_needed()

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    color = _default_color(ax.bar, hue, color, kwargs)
    if color is not None and saturation < 1:
        color = desaturate(color, saturation)

    aggregator = EstimateAggregator(estimator, errorbar, n_boot=n_boot, seed=seed)
    err_kws = {} if err_kws is None else _normalize_kwargs(err_kws, mpl.lines.Line2D)

    # Deprecations to remove in v0.15.0.
    err_kws, capsize = p._err_kws_backcompat(err_kws, errcolor, errwidth, capsize)

    p.plot_bars(
        aggregator=aggregator,
        dodge=dodge,
        width=width,
        color=color,
        saturation=saturation,
        capsize=capsize,
        err_kws=err_kws,
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


barplot.__doc__ = dedent("""\
    Show point estimates and errors as rectangular bars.

    A bar plot represents an aggregate or statistical estimate for a numeric
    variable with the height of each rectangle and indicates the uncertainty
    around that estimate using an error bar. Bar plots include 0 in the
    axis range, and they are a good choice when 0 is a meaningful value
    for the variable to take.

    {new_categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {stat_api_params}
    {orient}
    {color}
    {palette}
    {saturation}
    {hue_norm}
    {width}
    {capsize}
    {dodge}
    {native_scale}
    {formatter}
    {legend}
    {err_kws}
    {ci}
    {errcolor}
    {errwidth}
    {ax_in}
    kwargs : key, value mappings
        Other parameters are passed through to :class:`matplotlib.patches.Rectangle`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    {countplot}
    {pointplot}
    {catplot}

    Notes
    -----

    For datasets where 0 is not a meaningful value, a :func:`pointplot` will
    allow you to focus on differences between levels of one or more categorical
    variables.

    It is also important to keep in mind that a bar plot shows only the mean (or
    other aggregate) value, but it is often more informative to show the
    distribution of values at each level of the categorical variables. In those
    cases, approaches such as a :func:`boxplot` or :func:`violinplot` may be
    more appropriate.

    Examples
    --------
    .. include:: ../docstrings/barplot.rst


    """).format(**_categorical_docs)


def pointplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, units=None, seed=None,
    color=None, palette=None, hue_norm=None, markers=default, linestyles=default,
    dodge=False, native_scale=False, orient=None, capsize=0,
    formatter=None, legend="auto", err_kws=None,
    ci=deprecated, errwidth=deprecated, join=deprecated, scale=deprecated,
    ax=None,
    **kwargs,
):

    errorbar = utils._deprecate_ci(errorbar, ci)

    p = _CategoricalAggPlotter(
        data=data,
        variables=_CategoricalAggPlotter.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    color = _default_color(ax.plot, hue, color, kwargs)

    aggregator = EstimateAggregator(estimator, errorbar, n_boot=n_boot, seed=seed)
    err_kws = {} if err_kws is None else _normalize_kwargs(err_kws, mpl.lines.Line2D)

    # Deprecations to remove in v0.15.0.
    p._point_kwargs_backcompat(scale, join, kwargs)
    err_kws, capsize = p._err_kws_backcompat(err_kws, None, errwidth, capsize)

    p.plot_points(
        aggregator=aggregator,
        markers=markers,
        linestyles=linestyles,
        dodge=dodge,
        color=color,
        capsize=capsize,
        err_kws=err_kws,
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


pointplot.__doc__ = dedent("""\
    Show point estimates and errors using lines with markers.

    A point plot represents an estimate of central tendency for a numeric
    variable by the position of the dot and provides some indication of the
    uncertainty around that estimate using error bars.

    Point plots can be more useful than bar plots for focusing comparisons
    between different levels of one or more categorical variables. They are
    particularly adept at showing interactions: how the relationship between
    levels of one categorical variable changes across levels of a second
    categorical variable. The lines that join each point from the same `hue`
    level allow interactions to be judged by differences in slope, which is
    easier for the eyes than comparing the heights of several groups of points
    or bars.

    {new_categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {stat_api_params}
    {color}
    {palette}
    markers : string or list of strings
        Markers to use for each of the `hue` levels.
    linestyles : string or list of strings
        Line styles to use for each of the `hue` levels.
    dodge : bool or float
        Amount to separate the points for each level of the ``hue`` variable
        along the categorical axis.
    {native_scale}
    {orient}
    {capsize}
    {formatter}
    {legend}
    {err_kws}
    {ci}
    {errwidth}
    join : bool
        If `True`, draw lines will be drawn between point estimates.

        .. deprecated:: v0.13.0
            Set `linestyle="none"` to remove the lines between the points.
    scale : float
        Scale factor for the plot elements.

        .. deprecated:: v0.13.0
            Control element sizes with :class:`matplotlib.lines.Line2D` parameters.
    {ax_in}
    kwargs : key, value mappings
        Other parameters are passed through to :class:`matplotlib.lines.Line2D`.

        .. versionadded:: v0.13.0

    Returns
    -------
    {ax_out}

    See Also
    --------
    {barplot}
    {catplot}

    Notes
    -----
    It is important to keep in mind that a point plot shows only the mean (or
    other estimator) value, but in many cases it may be more informative to
    show the distribution of values at each level of the categorical variables.
    In that case, other approaches such as a box or violin plot may be more
    appropriate.

    Examples
    --------
    .. include:: ../docstrings/pointplot.rst

    """).format(**_categorical_docs)


def countplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75, hue_norm=None,
    stat="count", width=.8, dodge="auto", native_scale=False, formatter=None,
    legend="auto", ax=None, **kwargs
):

    if x is None and y is not None:
        orient = "y"
        x = 1
    elif x is not None and y is None:
        orient = "x"
        y = 1
    elif x is not None and y is not None:
        raise TypeError("Cannot pass values for both `x` and `y`.")

    p = _CategoricalAggPlotter(
        data=data,
        variables=_CategoricalAggPlotter.get_semantics(locals()),
        order=order,
        orient=orient,
        require_numeric=False,
        legend=legend,
    )

    if ax is None:
        ax = plt.gca()

    if p.plot_data.empty:
        return ax

    if dodge == "auto":
        # Needs to be before scale_categorical changes the coordinate series dtype
        dodge = p._dodge_needed()

    if p.var_types.get(p.orient) == "categorical" or not native_scale:
        p.scale_categorical(p.orient, order=order, formatter=formatter)

    p._attach(ax)

    # Deprecations to remove in v0.14.0.
    hue_order = p._palette_without_hue_backcompat(palette, hue_order)
    palette, hue_order = p._hue_backcompat(color, palette, hue_order)

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    color = _default_color(ax.bar, hue, color, kwargs)
    if color is not None and saturation < 1:
        color = desaturate(color, saturation)

    count_axis = {"x": "y", "y": "x"}[p.orient]
    if p.input_format == "wide":
        p.plot_data[count_axis] = 1

    _check_argument("stat", ["count", "percent", "probability", "proportion"], stat)
    p.variables[count_axis] = stat
    if stat != "count":
        denom = 100 if stat == "percent" else 1
        p.plot_data[count_axis] /= len(p.plot_data) / denom

    aggregator = EstimateAggregator("sum", errorbar=None)

    p.plot_bars(
        aggregator=aggregator,
        dodge=dodge,
        width=width,
        color=color,
        saturation=saturation,
        capsize=0,
        err_kws={},
        plot_kws=kwargs,
    )

    p._add_axis_labels(ax)
    p._adjust_cat_axis(ax, axis=p.orient)

    return ax


countplot.__doc__ = dedent("""\
    Show the counts of observations in each categorical bin using bars.

    A count plot can be thought of as a histogram across a categorical, instead
    of quantitative, variable. The basic API and options are identical to those
    for :func:`barplot`, so you can compare counts across nested variables.

    Note that :func:`histplot` function offers similar functionality with additional
    features (e.g. bar stacking), although its default behavior is somewhat different.

    {new_categorical_narrative}

    Parameters
    ----------
    {categorical_data}
    {input_params}
    {order_vars}
    {orient}
    {color}
    {palette}
    {saturation}
    {hue_norm}
    stat : {{'count', 'percent', 'proportion', 'probability'}}
        Statistic to compute; when not `'count'`, bar heights will be normalized so that
        they sum to 100 (for `'percent'`) or 1 (otherwise) across the plot.

        .. versionadded:: v0.13.0
    {width}
    {dodge}
    {native_scale}
    {formatter}
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other parameters are passed through to :class:`matplotlib.patches.Rectangle`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    histplot : Bin and count observations with additional options.
    {barplot}
    {catplot}

    Examples
    --------

    .. include:: ../docstrings/countplot.rst
    """).format(**_categorical_docs)


def catplot(
    data=None, *, x=None, y=None, hue=None, row=None, col=None,
    col_wrap=None, estimator="mean", errorbar=("ci", 95), n_boot=1000,
    units=None, seed=None, order=None, hue_order=None, row_order=None,
    col_order=None, height=5, aspect=1, kind="strip", native_scale=False,
    formatter=None, orient=None, color=None, palette=None, hue_norm=None,
    legend="auto", legend_out=True, sharex=True, sharey=True,
    margin_titles=False, facet_kws=None, ci="deprecated",
    **kwargs
):

    # Determine the plotting function
    try:
        plot_func = globals()[kind + "plot"]
    except KeyError:
        err = f"Plot kind '{kind}' is not recognized"
        raise ValueError(err)

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = ("catplot is a figure-level function and does not accept "
               f"target axes. You may wish to try {kind}plot")
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    refactored_kinds = ["strip", "swarm", "point", "bar", "count"]
    desaturated_kinds = ["bar", "count"]
    undodged_kinds = ["strip", "swarm", "point"]

    if kind in refactored_kinds:

        if kind in ["bar", "point", "count"]:
            Plotter = _CategoricalAggFacetPlotter
        else:
            Plotter = _CategoricalFacetPlotter

        if kind == "count":
            if x is None and y is not None:
                orient = "y"
                x = 1
            elif x is not None and y is None:
                orient = "x"
                y = 1
            elif x is not None and y is not None:
                raise ValueError("Cannot pass values for both `x` and `y`.")

        p = Plotter(
            data=data,
            variables=Plotter.get_semantics(locals()),
            order=order,
            orient=orient,
            require_numeric=False,
            legend=legend,
        )

        # XXX Copying a fair amount from displot, which is not ideal

        for var in ["row", "col"]:
            # Handle faceting variables that lack name information
            if var in p.variables and p.variables[var] is None:
                p.variables[var] = f"_{var}_"

        # Adapt the plot_data dataframe for use with FacetGrid
        data = p.plot_data.rename(columns=p.variables)
        data = data.loc[:, ~data.columns.duplicated()]

        col_name = p.variables.get("col", None)
        row_name = p.variables.get("row", None)

        if facet_kws is None:
            facet_kws = {}

        g = FacetGrid(
            data=data, row=row_name, col=col_name,
            col_wrap=col_wrap, row_order=row_order,
            col_order=col_order, height=height,
            sharex=sharex, sharey=sharey,
            aspect=aspect,
            **facet_kws,
        )

        # Capture this here because scale_categorical is going to insert a (null)
        # x variable even if it is empty. It's not clear whether that needs to
        # happen or if disabling that is the cleaner solution.
        has_xy_data = p.has_xy_data

        if not native_scale or p.var_types[p.orient] == "categorical":
            p.scale_categorical(p.orient, order=order, formatter=formatter)

        p._attach(g)

        if not has_xy_data:
            return g

        # Deprecations to remove in v0.14.0.
        hue_order = p._palette_without_hue_backcompat(palette, hue_order)
        palette, hue_order = p._hue_backcompat(color, palette, hue_order)

        p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

        # Set a default color
        # Otherwise each artist will be plotted separately and trip the color cycle
        saturation = kwargs.pop("saturation", 0.75)
        if hue is None:
            if color is None:
                color = "C0"
            if kind in desaturated_kinds and saturation < 1:
                color = desaturate(color, saturation)
        edgecolor = kwargs.pop("edgecolor", "gray")  # XXX TODO default

        width = kwargs.pop("width", 0.8)
        dodge = kwargs.pop("dodge", False if kind in undodged_kinds else "auto")
        if dodge == "auto":
            dodge = p._dodge_needed()

        if kind == "strip":

            # TODO get these defaults programmatically?
            jitter = kwargs.pop("jitter", True)

            # XXX Copying possibly bad default decisions from original code for now
            plot_kws = kwargs.copy()
            plot_kws.setdefault("zorder", 3)
            plot_kws.setdefault("s", plot_kws.pop("size", 5) ** 2)
            plot_kws.setdefault("linewidth", 0)

            p.plot_strips(
                jitter=jitter,
                dodge=dodge,
                color=color,
                edgecolor=edgecolor,
                plot_kws=plot_kws,
            )

        elif kind == "swarm":

            # TODO get these defaults programmatically?
            warn_thresh = kwargs.pop("warn_thresh", .05)

            # XXX Copying possibly bad default decisions from original code for now
            plot_kws = kwargs.copy()
            plot_kws.setdefault("zorder", 3)
            plot_kws.setdefault("s", plot_kws.pop("size", 5) ** 2)

            if plot_kws.setdefault("linewidth", 0) is None:
                plot_kws["linewidth"] = np.sqrt(plot_kws["s"]) / 10

            p.plot_swarms(
                dodge=dodge,
                color=color,
                edgecolor=edgecolor,
                warn_thresh=warn_thresh,
                plot_kws=plot_kws,
            )

        elif kind == "point":

            aggregator = EstimateAggregator(
                estimator, errorbar, n_boot=n_boot, seed=seed
            )

            markers = kwargs.pop("markers", default)
            linestyles = kwargs.pop("linestyles", default)
            # Uncomment when removing deprecation backcompat
            # capsize = kwargs.pop("capsize", 0)
            # err_kws = _normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D)

            # Deprecations to remove in v0.15.0.
            p._point_kwargs_backcompat(
                kwargs.pop("scale", deprecated),
                kwargs.pop("join", deprecated),
                kwargs
            )
            err_kws, capsize = p._err_kws_backcompat(
                _normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D),
                None,
                errwidth=kwargs.pop("errwidth", deprecated),
                capsize=kwargs.pop("capsize", 0),
            )

            p.plot_points(
                aggregator=aggregator,
                markers=markers,
                linestyles=linestyles,
                dodge=dodge,
                color=color,
                capsize=capsize,
                err_kws=err_kws,
                plot_kws=kwargs,
            )

        elif kind == "bar":

            aggregator = EstimateAggregator(
                estimator, errorbar, n_boot=n_boot, seed=seed
            )

            err_kws, capsize = p._err_kws_backcompat(
                _normalize_kwargs(kwargs.pop("err_kws", {}), mpl.lines.Line2D),
                errcolor=kwargs.pop("errcolor", deprecated),
                errwidth=kwargs.pop("errwidth", deprecated),
                capsize=kwargs.pop("capsize", 0),
            )

            p.plot_bars(
                aggregator=aggregator,
                dodge=dodge,
                width=width,
                color=color,
                saturation=saturation,
                capsize=capsize,
                err_kws=err_kws,
                plot_kws=kwargs,
            )

        elif kind == "count":

            aggregator = EstimateAggregator("sum", errorbar=None)

            count_axis = {"x": "y", "y": "x"}[p.orient]
            p.plot_data[count_axis] = 1

            stat_options = ["count", "percent", "probability", "proportion"]
            stat = _check_argument("stat", stat_options, kwargs.pop("stat", "count"))
            p.variables[count_axis] = stat
            if stat != "count":
                denom = 100 if stat == "percent" else 1
                p.plot_data[count_axis] /= len(p.plot_data) / denom

            p.plot_bars(
                aggregator=aggregator,
                dodge=dodge,
                width=width,
                color=color,
                saturation=saturation,
                capsize=0,
                err_kws={},
                plot_kws=kwargs,
            )

        for ax in g.axes.flat:
            p._adjust_cat_axis(ax, axis=p.orient)

        g.set_axis_labels(p.variables.get("x"), p.variables.get("y"))
        g.set_titles()
        g.tight_layout()

        for ax in g.axes.flat:
            g._update_legend_data(ax)
            ax.legend_ = None

        if legend and "hue" in p.variables:
            g.add_legend(title=p.variables.get("hue"), label_order=hue_order)

        return g

    # Don't allow usage of forthcoming functionality
    if native_scale is True:
        err = f"native_scale not yet implemented for `kind={kind}`"
        raise ValueError(err)
    if formatter is not None:
        err = f"formatter not yet implemented for `kind={kind}`"
        raise ValueError(err)

    # Alias the input variables to determine categorical order and palette
    # correctly in the case of a count plot
    if kind == "count":
        if x is None and y is not None:
            x_, y_, orient = y, y, "y"
        elif y is None and x is not None:
            x_, y_, orient = x, x, "x"
        else:
            raise ValueError("Either `x` or `y` must be None for kind='count'")
    else:
        x_, y_ = x, y

    # Determine the order for the whole dataset, which will be used in all
    # facets to ensure representation of all data in the final plot
    plotter_class = {
        "box": _BoxPlotter,
        "violin": _ViolinPlotter,
        "boxen": _LVPlotter,
    }[kind]
    p = _CategoricalPlotter()
    p.require_numeric = plotter_class.require_numeric
    p.establish_variables(x_, y_, hue, data, orient, order, hue_order)
    if (
        order is not None
        or (sharex and p.orient == "x")
        or (sharey and p.orient == "y")
    ):
        # Sync categorical axis between facets to have the same categories
        order = p.group_names
    elif color is None and hue is None:
        msg = (
            "Setting `{}=False` with `color=None` may cause different levels of the "
            "`{}` variable to share colors. This will change in a future version."
        )
        if not sharex and p.orient == "x":
            warnings.warn(msg.format("sharex", "x"), UserWarning)
        if not sharey and p.orient == "y":
            warnings.warn(msg.format("sharey", "y"), UserWarning)

    hue_order = p.hue_names

    # Determine the palette to use
    # (FacetGrid will pass a value for ``color`` to the plotting function
    # so we need to define ``palette`` to get default behavior for the
    # categorical functions
    p.establish_colors(color, palette, 1)
    if kind != "point" or hue is not None:
        palette = p.colors

    # Determine keyword arguments for the facets
    facet_kws = {} if facet_kws is None else facet_kws
    facet_kws.update(
        data=data, row=row, col=col,
        row_order=row_order, col_order=col_order,
        col_wrap=col_wrap, height=height, aspect=aspect,
        sharex=sharex, sharey=sharey,
        legend_out=legend_out, margin_titles=margin_titles,
        dropna=False,
    )

    # Determine keyword arguments for the plotting function
    plot_kws = dict(
        order=order, hue_order=hue_order,
        orient=orient, color=color, palette=palette,
    )
    plot_kws.update(kwargs)

    # Initialize the facets
    g = FacetGrid(**facet_kws)

    # Draw the plot onto the facets
    g.map_dataframe(plot_func, x=x, y=y, hue=hue, **plot_kws)

    if p.orient == "y":
        g.set_axis_labels(p.value_label, p.group_label)
    else:
        g.set_axis_labels(p.group_label, p.value_label)

    if legend and (hue is not None) and (hue not in [x, row, col]):
        hue_order = list(map(utils.to_utf8, hue_order))
        g.add_legend(title=hue, label_order=hue_order)

    return g


catplot.__doc__ = dedent("""\
    Figure-level interface for drawing categorical plots onto a FacetGrid.

    This function provides access to several axes-level functions that
    show the relationship between a numerical and one or more categorical
    variables using one of several visual representations. The `kind`
    parameter selects the underlying axes-level function to use:

    Categorical scatterplots:

    - :func:`stripplot` (with `kind="strip"`; the default)
    - :func:`swarmplot` (with `kind="swarm"`)

    Categorical distribution plots:

    - :func:`boxplot` (with `kind="box"`)
    - :func:`violinplot` (with `kind="violin"`)
    - :func:`boxenplot` (with `kind="boxen"`)

    Categorical estimate plots:

    - :func:`pointplot` (with `kind="point"`)
    - :func:`barplot` (with `kind="bar"`)
    - :func:`countplot` (with `kind="count"`)

    Extra keyword arguments are passed to the underlying function, so you
    should refer to the documentation for each to see kind-specific options.

    Note that unlike when using the axes-level functions directly, data must be
    passed in a long-form DataFrame with variables specified by passing strings
    to `x`, `y`, `hue`, etc.

    {categorical_narrative}

    After plotting, the :class:`FacetGrid` with the plot is returned and can
    be used directly to tweak supporting plot details or add other layers.

    Parameters
    ----------
    {long_form_data}
    {string_input_params}
    row, col : names of variables in `data`
        Categorical variables that will determine the faceting of the grid.
    {col_wrap}
    {stat_api_params}
    {order_vars}
    row_order, col_order : lists of strings
        Order to organize the rows and/or columns of the grid in, otherwise the
        orders are inferred from the data objects.
    {height}
    {aspect}
    kind : str
        The kind of plot to draw, corresponds to the name of a categorical
        axes-level plotting function. Options are: "strip", "swarm", "box", "violin",
        "boxen", "point", "bar", or "count".
    {native_scale}
    {formatter}
    {orient}
    {color}
    {palette}
    {hue_norm}
    legend : str or bool
        Set to `False` to disable the legend. With `strip` or `swarm` plots,
        this also accepts a string, as described in the axes-level docstrings.
    {legend_out}
    {share_xy}
    {margin_titles}
    facet_kws : dict
        Dictionary of other keyword arguments to pass to :class:`FacetGrid`.
    kwargs : key, value pairings
        Other keyword arguments are passed through to the underlying plotting
        function.

    Returns
    -------
    g : :class:`FacetGrid`
        Returns the :class:`FacetGrid` object with the plot on it for further
        tweaking.

    Examples
    --------

    .. include:: ../docstrings/catplot.rst

    """).format(**_categorical_docs)


class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""
    def __init__(self, orient="x", width=0.8, warn_thresh=.05):

        # XXX should we keep the orient parameterization or specify the swarm axis?

        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points, center):
        """Swarm `points`, a PathCollection, around the `center` position."""
        # Convert from point size (area) to diameter

        ax = points.axes
        dpi = ax.figure.dpi

        # Get the original positions of the points
        orig_xy_data = points.get_offsets()

        # Reset the categorical positions to the center line
        cat_idx = 1 if self.orient == "y" else 0
        orig_xy_data[:, cat_idx] = center

        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)

        # Order the variables so that x is the categorical axis
        if self.orient == "y":
            orig_xy = orig_xy[:, [1, 0]]

        # Add a column with each point's radius
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]

        # Sort along the value axis to facilitate the beeswarm
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]

        # Adjust points along the categorical axis to prevent overlaps
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)

        # Transform the point coordinates back to data coordinates
        if self.orient == "y":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T

        log_scale = getattr(ax, f"get_{self.orient}scale")() == "log"

        # Add gutters
        if self.orient == "y":
            self.add_gutters(new_y_data, center, log_scale=log_scale)
        else:
            self.add_gutters(new_x_data, center, log_scale=log_scale)

        # Reposition the points so they do not overlap
        if self.orient == "y":
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        # In this method, `x` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xyr[0, 0]

        # Start the swarm with the first point
        swarm = np.atleast_2d(orig_xyr[0])

        # Loop over the remaining points
        for xyr_i in orig_xyr[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xyr_i, swarm)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xyr_i, neighbors)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbors
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # Place it into the swarm
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        # Because we work backwards through the swarm and can short-circuit,
        # the for-loop is faster than vectorization
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""

        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:

            x_i, y_i, r_i = xyr_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)

            sep_needed = np.square(neighbors_r + r_i)

            # Good candidate does not overlap any of neighbors which means that
            # squared distance between candidate and any of the neighbors has
            # to be at least square of the summed radii
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )

    def add_gutters(self, points, center, log_scale=False):
        """Stop points from extending beyond their territory."""
        half_width = self.width / 2
        if log_scale:
            low_gutter = 10 ** (np.log10(center) - half_width)
        else:
            low_gutter = center - half_width
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        if log_scale:
            high_gutter = 10 ** (np.log10(center) + half_width)
        else:
            high_gutter = center + half_width
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter

        gutter_prop = (off_high + off_low).sum() / len(points)
        if gutter_prop > self.warn_thresh:
            msg = (
                "{:.1%} of the points cannot be placed; you may want "
                "to decrease the size of the markers or use stripplot."
            ).format(gutter_prop)
            warnings.warn(msg, UserWarning)

        return points


class BoxPlotContainer:

    def __init__(self, artist_dict):

        self.boxes = artist_dict["boxes"]
        self.medians = artist_dict["medians"]
        self.whiskers = artist_dict["whiskers"]
        self.caps = artist_dict["caps"]
        self.fliers = artist_dict["fliers"]
        self.means = artist_dict["means"]

        self._label = None
        self._children = [
            *self.boxes,
            *self.medians,
            *self.whiskers,
            *self.caps,
            *self.fliers,
            *self.means,
        ]

    def __repr__(self):
        return f"<BoxPlotContainer object with {len(self.boxes)} boxes>"

    def __getitem__(self, idx):
        pair_slice = slice(2 * idx, 2 * idx + 2)
        return {
            "box": self.boxes[idx] if self.boxes else [],
            "median": self.medians[idx] if self.medians else [],
            "whiskers": self.whiskers[pair_slice] if self.whiskers else [],
            "caps": self.caps[pair_slice] if self.caps else [],
            "fliers": self.fliers[idx] if self.fliers else [],
            "means": self.means[idx]if self.means else [],
        }

    def __iter__(self):
        yield from [self[i] for i in range(self.boxes)]

    def get_label(self):
        return self._label

    def set_label(self, value):
        self._label = value

    def get_children(self):
        return self._children

    def remove(self):
        for child in self._children:
            child.remove()
