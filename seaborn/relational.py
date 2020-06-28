from textwrap import dedent
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from ._core import (
    VectorPlotter,
)
from .utils import (
    ci_to_errsize,
    locator_to_legend_entries,
    ci as ci_func
)
from .algorithms import bootstrap
from .axisgrid import FacetGrid, _facet_docs
from ._decorators import _deprecate_positional_args


__all__ = ["relplot", "scatterplot", "lineplot"]


class _RelationalPlotter(VectorPlotter):

    wide_structure = {
        "x": "index", "y": "values", "hue": "columns", "style": "columns",
    }

    # TODO where best to define default parameters?
    sort = True

    def label_axes(self, ax):
        """Set x and y labels with visibility that matches the ticklabels."""
        if "x" in self.variables and self.variables["x"] is not None:
            x_visible = any(t.get_visible() for t in ax.get_xticklabels())
            ax.set_xlabel(self.variables["x"], visible=x_visible)
        if "y" in self.variables and self.variables["y"] is not None:
            y_visible = any(t.get_visible() for t in ax.get_yticklabels())
            ax.set_ylabel(self.variables["y"], visible=y_visible)

    def add_legend_data(self, ax):
        """Add labeled artists to represent the different plot semantics."""
        verbosity = self.legend
        if verbosity not in ["brief", "full"]:
            err = "`legend` must be 'brief', 'full', or False"
            raise ValueError(err)

        legend_kwargs = {}
        keys = []

        title_kws = dict(color="w", s=0, linewidth=0, marker="", dashes="")

        def update(var_name, val_name, **kws):

            key = var_name, val_name
            if key in legend_kwargs:
                legend_kwargs[key].update(**kws)
            else:
                keys.append(key)

                legend_kwargs[key] = dict(**kws)

        # -- Add a legend for hue semantics
        if verbosity == "brief" and self._hue_map.map_type == "numeric":
            if isinstance(self._hue_map.norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=3)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=3)
            limits = min(self._hue_map.levels), max(self._hue_map.levels)
            hue_levels, hue_formatted_levels = locator_to_legend_entries(
                locator, limits, self.plot_data["hue"].dtype
            )
        elif self._hue_map.levels is None:
            hue_levels = hue_formatted_levels = []
        else:
            hue_levels = hue_formatted_levels = self._hue_map.levels

        # Add the hue semantic subtitle
        if "hue" in self.variables and self.variables["hue"] is not None:
            update((self.variables["hue"], "title"),
                   self.variables["hue"], **title_kws)

        # Add the hue semantic labels
        for level, formatted_level in zip(hue_levels, hue_formatted_levels):
            if level is not None:
                color = self._hue_map(level)
                update(self.variables["hue"], formatted_level, color=color)

        # -- Add a legend for size semantics

        if verbosity == "brief" and self._size_map.map_type == "numeric":
            # Define how ticks will interpolate between the min/max data values
            if isinstance(self._size_map.norm, mpl.colors.LogNorm):
                locator = mpl.ticker.LogLocator(numticks=3)
            else:
                locator = mpl.ticker.MaxNLocator(nbins=3)
            # Define the min/max data values
            limits = min(self._size_map.levels), max(self._size_map.levels)
            size_levels, size_formatted_levels = locator_to_legend_entries(
                locator, limits, self.plot_data["size"].dtype
            )
        elif self._size_map.levels is None:
            size_levels = size_formatted_levels = []
        else:
            size_levels = size_formatted_levels = self._size_map.levels

        # Add the size semantic subtitle
        if "size" in self.variables and self.variables["size"] is not None:
            update((self.variables["size"], "title"),
                   self.variables["size"], **title_kws)

        # Add the size semantic labels
        for level, formatted_level in zip(size_levels, size_formatted_levels):
            if level is not None:
                size = self._size_map(level)
                update(
                    self.variables["size"],
                    formatted_level,
                    linewidth=size,
                    s=size,
                )

        # -- Add a legend for style semantics

        # Add the style semantic title
        if "style" in self.variables and self.variables["style"] is not None:
            update((self.variables["style"], "title"),
                   self.variables["style"], **title_kws)

        # Add the style semantic labels
        if self._style_map.levels is not None:
            for level in self._style_map.levels:
                if level is not None:
                    attrs = self._style_map(level)
                    update(
                        self.variables["style"],
                        level,
                        marker=attrs.get("marker", ""),
                        dashes=attrs.get("dashes", ""),
                    )

        func = getattr(ax, self._legend_func)

        legend_data = {}
        legend_order = []

        for key in keys:

            _, label = key
            kws = legend_kwargs[key]
            kws.setdefault("color", ".2")
            use_kws = {}
            for attr in self._legend_attributes + ["visible"]:
                if attr in kws:
                    use_kws[attr] = kws[attr]
            artist = func([], [], label=label, **use_kws)
            if self._legend_func == "plot":
                artist = artist[0]
            legend_data[key] = artist
            legend_order.append(key)

        self.legend_data = legend_data
        self.legend_order = legend_order


class _LinePlotter(_RelationalPlotter):

    _legend_attributes = ["color", "linewidth", "marker", "dashes"]
    _legend_func = "plot"

    def __init__(
        self, *,
        data=None, variables={},
        estimator=None, ci=None, n_boot=None, seed=None,
        sort=True, err_style=None, err_kws=None, legend=None
    ):

        # TODO this is messy, we want the mapping to be agnoistic about
        # the kind of plot to draw, but for the time being we need to set
        # this information so the SizeMapping can use it
        self._default_size_range = (
            np.r_[.5, 2] * mpl.rcParams["lines.linewidth"]
        )

        super().__init__(data=data, variables=variables)

        self.estimator = estimator
        self.ci = ci
        self.n_boot = n_boot
        self.seed = seed
        self.sort = sort
        self.err_style = err_style
        self.err_kws = {} if err_kws is None else err_kws

        self.legend = legend

    def aggregate(self, vals, grouper, units=None):
        """Compute an estimate and confidence interval using grouper."""
        func = self.estimator
        ci = self.ci
        n_boot = self.n_boot
        seed = self.seed

        # Define a "null" CI for when we only have one value
        null_ci = pd.Series(index=["low", "high"], dtype=np.float)

        # Function to bootstrap in the context of a pandas group by
        def bootstrapped_cis(vals):

            if len(vals) <= 1:
                return null_ci

            boots = bootstrap(vals, func=func, n_boot=n_boot, seed=seed)
            cis = ci_func(boots, ci)
            return pd.Series(cis, ["low", "high"])

        # Group and get the aggregation estimate
        grouped = vals.groupby(grouper, sort=self.sort)
        est = grouped.agg(func)

        # Exit early if we don't want a confidence interval
        if ci is None:
            return est.index, est, None

        # Compute the error bar extents
        if ci == "sd":
            sd = grouped.std()
            cis = pd.DataFrame(np.c_[est - sd, est + sd],
                               index=est.index,
                               columns=["low", "high"]).stack()
        else:
            cis = grouped.apply(bootstrapped_cis)

        # Unpack the CIs into "wide" format for plotting
        if cis.notnull().any():
            cis = cis.unstack().reindex(est.index)
        else:
            cis = None

        return est.index, est, cis

    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        scout, = ax.plot([], [], **kws)

        orig_color = kws.pop("color", scout.get_color())
        orig_marker = kws.pop("marker", scout.get_marker())
        orig_linewidth = kws.pop("linewidth",
                                 kws.pop("lw", scout.get_linewidth()))

        # Note that scout.get_linestyle() is` not correct as of mpl 3.2
        orig_linestyle = kws.pop("linestyle", kws.pop("ls", None))

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        scout.remove()

        # Set default error kwargs
        err_kws = self.err_kws.copy()
        if self.err_style == "band":
            err_kws.setdefault("alpha", .2)
        elif self.err_style == "bars":
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))

        # Set the default artist keywords
        kws.update(dict(
            color=orig_color,
            marker=orig_marker,
            linewidth=orig_linewidth,
            linestyle=orig_linestyle,
        ))

        # Loop over the semantic subsets and add to the plot
        grouping_semantics = "hue", "size", "style"
        for sub_vars, sub_data in self._semantic_subsets(
            grouping_semantics, from_comp_data=True
        ):

            if self.sort:
                sort_vars = ["units", "x", "y"]
                sort_cols = [var for var in sort_vars if var in self.variables]
                sub_data = sub_data.sort_values(sort_cols)

            # Due to the original design, code below was written assuming that
            # sub_data always has x, y, and units columns, which may be empty.
            # Adding this here to avoid otherwise disruptive changes, but it
            # could get removed if the rest of the logic is sorted out
            null = pd.Series(index=sub_data.index, dtype=float)

            x = sub_data.get("x", null)
            y = sub_data.get("y", null)
            u = sub_data.get("units", null)

            if self.estimator is not None:
                if "units" in self.variables:
                    err = "estimator must be None when specifying units"
                    raise ValueError(err)
                x, y, y_ci = self.aggregate(y, x, u)
            else:
                y_ci = None

            if "hue" in sub_vars:
                kws["color"] = self._hue_map(sub_vars["hue"])
            if "size" in sub_vars:
                kws["linewidth"] = self._size_map(sub_vars["size"])
            if "style" in sub_vars:
                attributes = self._style_map(sub_vars["style"])
                if "dashes" in attributes:
                    kws["dashes"] = attributes["dashes"]
                if "marker" in attributes:
                    kws["marker"] = attributes["marker"]

            line, = ax.plot([], [], **kws)
            line_color = line.get_color()
            line_alpha = line.get_alpha()
            line_capstyle = line.get_solid_capstyle()
            line.remove()

            # --- Draw the main line

            x, y = np.asarray(x), np.asarray(y)

            if "units" in self.variables:
                for u_i in u.unique():
                    rows = np.asarray(u == u_i)
                    ax.plot(x[rows], y[rows], **kws)
            else:
                line, = ax.plot(x, y, **kws)

            # --- Draw the confidence intervals

            if y_ci is not None:

                low, high = np.asarray(y_ci["low"]), np.asarray(y_ci["high"])

                if self.err_style == "band":

                    ax.fill_between(x, low, high, color=line_color, **err_kws)

                elif self.err_style == "bars":

                    y_err = ci_to_errsize((low, high), y)
                    ebars = ax.errorbar(x, y, y_err, linestyle="",
                                        color=line_color, alpha=line_alpha,
                                        **err_kws)

                    # Set the capstyle properly on the error bars
                    for obj in ebars.get_children():
                        try:
                            obj.set_capstyle(line_capstyle)
                        except AttributeError:
                            # Does not exist on mpl < 2.2
                            pass

        # Finalize the axes details
        self.label_axes(ax)
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()


class _ScatterPlotter(_RelationalPlotter):

    _legend_attributes = ["color", "s", "marker"]
    _legend_func = "scatter"

    def __init__(
        self, *,
        data=None, variables={},
        x_bins=None, y_bins=None,
        estimator=None, ci=None, n_boot=None,
        alpha=None, x_jitter=None, y_jitter=None,
        legend=None
    ):

        # TODO this is messy, we want the mapping to be agnoistic about
        # the kind of plot to draw, but for the time being we need to set
        # this information so the SizeMapping can use it
        self._default_size_range = (
            np.r_[.5, 2] * np.square(mpl.rcParams["lines.markersize"])
        )

        super().__init__(data=data, variables=variables)

        self.alpha = alpha
        self.legend = legend

    def plot(self, ax, kws):

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        scout_size = max(
            np.atleast_1d(kws.get("s", [])).shape[0],
            np.atleast_1d(kws.get("c", [])).shape[0],
        )
        scout_x = scout_y = np.full(scout_size, np.nan)
        scout = ax.scatter(scout_x, scout_y, **kws)
        s = kws.pop("s", scout.get_sizes())
        c = kws.pop("c", scout.get_facecolors())
        scout.remove()

        kws.pop("color", None)  # TODO is this optimal?

        # --- Determine the visual attributes of the plot

        data = self.plot_data[list(self.variables)].dropna()
        if not data.size:
            return

        # Define the vectors of x and y positions
        empty = np.full(len(data), np.nan)
        x = data.get("x", empty)
        y = data.get("y", empty)

        # Apply the mapping from semantic varibles to artist attributes
        if "hue" in self.variables:
            c = self._hue_map(data["hue"])

        if "size" in self.variables:
            s = self._size_map(data["size"])

        # Set defaults for other visual attributres
        kws.setdefault("linewidth", .08 * np.sqrt(np.percentile(s, 10)))
        kws.setdefault("edgecolor", "w")

        if "style" in self.variables:
            # Use a representative marker so scatter sets the edgecolor
            # properly for line art markers. We currently enforce either
            # all or none line art so this works.
            example_level = self._style_map.levels[0]
            example_marker = self._style_map(example_level, "marker")
            kws.setdefault("marker", example_marker)

        # TODO this makes it impossible to vary alpha with hue which might
        # otherwise be useful? Should we just pass None?
        kws["alpha"] = 1 if self.alpha == "auto" else self.alpha

        # Draw the scatter plot
        args = np.asarray(x), np.asarray(y), np.asarray(s), np.asarray(c)
        points = ax.scatter(*args, **kws)

        # Update the paths to get different marker shapes.
        # This has to be done here because ax.scatter allows varying sizes
        # and colors but only a single marker shape per call.
        if "style" in self.variables:
            p = [self._style_map(val, "path") for val in data["style"]]
            points.set_paths(p)

        # Finalize the axes details
        self.label_axes(ax)
        if self.legend:
            self.add_legend_data(ax)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend()


_relational_docs = dict(

    # ---  Introductory prose
    main_api_narrative=dedent("""\
    The relationship between ``x`` and ``y`` can be shown for different subsets
    of the data using the ``hue``, ``size``, and ``style`` parameters. These
    parameters control what visual semantics are used to identify the different
    subsets. It is possible to show up to three dimensions independently by
    using all three semantic types, but this style of plot can be hard to
    interpret and is often ineffective. Using redundant semantics (i.e. both
    ``hue`` and ``style`` for the same variable) can be helpful for making
    graphics more accessible.

    See the :ref:`tutorial <relational_tutorial>` for more information.\
    """),

    relational_semantic_narrative=dedent("""\
    The default treatment of the ``hue`` (and to a lesser extent, ``size``)
    semantic, if present, depends on whether the variable is inferred to
    represent "numeric" or "categorical" data. In particular, numeric variables
    are represented with a sequential colormap by default, and the legend
    entries show regular "ticks" with values that may or may not exist in the
    data. This behavior can be controlled through various parameters, as
    described and illustrated below.\
    """),

    # --- Shared function parameters
    data_vars=dedent("""\
    x, y : names of variables in ``data`` or vector data, optional
        Input data variables; must be numeric. Can pass data directly or
        reference columns in ``data``.\
    """),
    data=dedent("""\
    data : DataFrame, array, or list of arrays, optional
        Input data structure. If ``x`` and ``y`` are specified as names, this
        should be a "long-form" DataFrame containing those columns. Otherwise
        it is treated as "wide-form" data and grouping variables are ignored.
        See the examples for the various ways this parameter can be specified
        and the different effects of each.\
    """),
    palette=dedent("""\
    palette : string, list, dict, or matplotlib colormap
        An object that determines how colors are chosen when ``hue`` is used.
        It can be the name of a seaborn palette or matplotlib colormap, a list
        of colors (anything matplotlib understands), a dict mapping levels
        of the ``hue`` variable to colors, or a matplotlib colormap object.\
    """),
    hue_order=dedent("""\
    hue_order : list, optional
        Specified order for the appearance of the ``hue`` variable levels,
        otherwise they are determined from the data. Not relevant when the
        ``hue`` variable is numeric.\
    """),
    hue_norm=dedent("""\
    hue_norm : tuple or Normalize object, optional
        Normalization in data units for colormap applied to the ``hue``
        variable when it is numeric. Not relevant if it is categorical.\
    """),
    sizes=dedent("""\
    sizes : list, dict, or tuple, optional
        An object that determines how sizes are chosen when ``size`` is used.
        It can always be a list of size values or a dict mapping levels of the
        ``size`` variable to sizes. When ``size``  is numeric, it can also be
        a tuple specifying the minimum and maximum size to use such that other
        values are normalized within this range.\
    """),
    size_order=dedent("""\
    size_order : list, optional
        Specified order for appearance of the ``size`` variable levels,
        otherwise they are determined from the data. Not relevant when the
        ``size`` variable is numeric.\
    """),
    size_norm=dedent("""\
    size_norm : tuple or Normalize object, optional
        Normalization in data units for scaling plot objects when the
        ``size`` variable is numeric.\
    """),
    markers=dedent("""\
    markers : boolean, list, or dictionary, optional
        Object determining how to draw the markers for different levels of the
        ``style`` variable. Setting to ``True`` will use default markers, or
        you can pass a list of markers or a dictionary mapping levels of the
        ``style`` variable to markers. Setting to ``False`` will draw
        marker-less lines.  Markers are specified as in matplotlib.\
    """),
    style_order=dedent("""\
    style_order : list, optional
        Specified order for appearance of the ``style`` variable levels
        otherwise they are determined from the data. Not relevant when the
        ``style`` variable is numeric.\
    """),
    units=dedent("""\
    units : {long_form_var}
        Grouping variable identifying sampling units. When used, a separate
        line will be drawn for each unit with appropriate semantics, but no
        legend entry will be added. Useful for showing distribution of
        experimental replicates when exact identities are not needed.
    """),
    estimator=dedent("""\
    estimator : name of pandas method or callable or None, optional
        Method for aggregating across multiple observations of the ``y``
        variable at the same ``x`` level. If ``None``, all observations will
        be drawn.\
    """),
    ci=dedent("""\
    ci : int or "sd" or None, optional
        Size of the confidence interval to draw when aggregating with an
        estimator. "sd" means to draw the standard deviation of the data.
        Setting to ``None`` will skip bootstrapping.\
    """),
    n_boot=dedent("""\
    n_boot : int, optional
        Number of bootstraps to use for computing the confidence interval.\
    """),
    seed=dedent("""\
    seed : int, numpy.random.Generator, or numpy.random.RandomState, optional
        Seed or random number generator for reproducible bootstrapping.\
    """),
    legend=dedent("""\
    legend : "brief", "full", or False, optional
        How to draw the legend. If "brief", numeric ``hue`` and ``size``
        variables will be represented with a sample of evenly spaced values.
        If "full", every group will get an entry in the legend. If ``False``,
        no legend data is added and no legend is drawn.\
    """),
    ax_in=dedent("""\
    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.\
    """),
    ax_out=dedent("""\
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.\
    """),

    # --- Repeated phrases
    long_form_var="name of variables in ``data`` or vector data, optional",


)

_relational_docs.update(_facet_docs)


@_deprecate_positional_args
def lineplot(
    *,
    x=None, y=None,
    hue=None, size=None, style=None,
    data=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    dashes=True, markers=None, style_order=None,
    units=None, estimator="mean", ci=95, n_boot=1000, seed=None,
    sort=True, err_style="band", err_kws=None,
    legend="brief", ax=None, **kwargs
):

    variables = _LinePlotter.get_semantics(locals())
    p = _LinePlotter(
        data=data, variables=variables,
        estimator=estimator, ci=ci, n_boot=n_boot, seed=seed,
        sort=sort, err_style=err_style, err_kws=err_kws, legend=legend,
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    if ax is None:
        ax = plt.gca()

    if not p.has_xy_data:
        return ax

    p._attach(ax)

    p.plot(ax, kwargs)
    return ax


lineplot.__doc__ = dedent("""\
    Draw a line plot with possibility of several semantic groupings.

    {main_api_narrative}

    {relational_semantic_narrative}

    By default, the plot aggregates over multiple ``y`` values at each value of
    ``x`` and shows an estimate of the central tendency and a confidence
    interval for that estimate.

    Parameters
    ----------
    {data_vars}
    hue : {long_form_var}
        Grouping variable that will produce lines with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : {long_form_var}
        Grouping variable that will produce lines with different widths.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : {long_form_var}
        Grouping variable that will produce lines with different dashes
        and/or markers. Can have a numeric dtype but will always be treated
        as categorical.
    {data}
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    dashes : boolean, list, or dictionary, optional
        Object determining how to draw the lines for different levels of the
        ``style`` variable. Setting to ``True`` will use default dash codes, or
        you can pass a list of dash codes or a dictionary mapping levels of the
        ``style`` variable to dash codes. Setting to ``False`` will use solid
        lines for all subsets. Dashes are specified as in matplotlib: a tuple
        of ``(segment, gap)`` lengths, or an empty string to draw a solid line.
    {markers}
    {style_order}
    {units}
    {estimator}
    {ci}
    {n_boot}
    {seed}
    sort : boolean, optional
        If True, the data will be sorted by the x and y variables, otherwise
        lines will connect points in the order they appear in the dataset.
    err_style : "band" or "bars", optional
        Whether to draw the confidence intervals with translucent error bands
        or discrete error bars.
    err_kws : dict of keyword arguments
        Additional paramters to control the aesthetics of the error bars. The
        kwargs are passed either to :meth:`matplotlib.axes.Axes.fill_between`
        or :meth:`matplotlib.axes.Axes.errorbar`, depending on ``err_style``.
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed down to
        :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    scatterplot : Show the relationship between two variables without
                  emphasizing continuity of the ``x`` variable.
    pointplot : Show the relationship between two variables when one is
                categorical.

    Examples
    --------

    Draw a single line plot with error bands showing a confidence interval:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set()
        >>> import matplotlib.pyplot as plt
        >>> fmri = sns.load_dataset("fmri")
        >>> ax = sns.lineplot(x="timepoint", y="signal", data=fmri)

    Group by another variable and show the groups with different colors:


    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   data=fmri)

    Show the grouping variable with both color and line dashing:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="event", style="event", data=fmri)

    Use color and line dashing to represent two different grouping variables:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="region", style="event", data=fmri)

    Use markers instead of the dashes to identify groups:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal",
        ...                   hue="event", style="event",
        ...                   markers=True, dashes=False, data=fmri)

    Show error bars instead of error bands and plot the standard error:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   err_style="bars", ci=68, data=fmri)

    Show experimental replicates instead of aggregating:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="timepoint", y="signal", hue="event",
        ...                   units="subject", estimator=None, lw=1,
        ...                   data=fmri.query("region == 'frontal'"))

    Use a quantitative color mapping:

    .. plot::
        :context: close-figs

        >>> dots = sns.load_dataset("dots").query("align == 'dots'")
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   data=dots)

    Use a different normalization for the colormap:

    .. plot::
        :context: close-figs

        >>> from matplotlib.colors import LogNorm
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   hue_norm=LogNorm(),
        ...                   data=dots.query("coherence > 0"))

    Use a different color palette:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   palette="ch:2.5,.25", data=dots)

    Use specific color values, treating the hue variable as categorical:

    .. plot::
        :context: close-figs

        >>> palette = sns.color_palette("mako_r", 6)
        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   hue="coherence", style="choice",
        ...                   palette=palette, data=dots)

    Change the width of the lines with a quantitative variable:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   size="coherence", hue="choice",
        ...                   legend="full", data=dots)

    Change the range of line widths used to normalize the size variable:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(x="time", y="firing_rate",
        ...                   size="coherence", hue="choice",
        ...                   sizes=(.25, 2.5), data=dots)

    Plot from a wide-form DataFrame:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; plt.close("all")
        >>> index = pd.date_range("1 1 2000", periods=100,
        ...                       freq="m", name="date")
        >>> data = np.random.randn(100, 4).cumsum(axis=0)
        >>> wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
        >>> ax = sns.lineplot(data=wide_df)

    Plot from a list of Series:

    .. plot::
        :context: close-figs

        >>> list_data = [wide_df.loc[:"2005", "a"], wide_df.loc["2003":, "b"]]
        >>> ax = sns.lineplot(data=list_data)

    Plot a single Series, pass kwargs to :meth:`matplotlib.axes.Axes.plot`:

    .. plot::
        :context: close-figs

        >>> ax = sns.lineplot(data=wide_df["a"], color="coral", label="line")

    Draw lines at points as they appear in the dataset:

    .. plot::
        :context: close-figs

        >>> x, y = np.random.randn(2, 5000).cumsum(axis=1)
        >>> ax = sns.lineplot(x=x, y=y, sort=False, lw=1)

    Use :func:`relplot` to combine :func:`lineplot` and :class:`FacetGrid`:
    This allows grouping within additional categorical variables. Using
    :func:`relplot` is safer than using :class:`FacetGrid` directly, as it
    ensures synchronization of the semantic mappings across facets.

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                  col="region", hue="event", style="event",
        ...                  kind="line", data=fmri)

    """).format(**_relational_docs)


@_deprecate_positional_args
def scatterplot(
    *,
    x=None, y=None,
    hue=None, style=None, size=None, data=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=True, style_order=None,
    x_bins=None, y_bins=None,
    units=None, estimator=None, ci=95, n_boot=1000,
    alpha=None, x_jitter=None, y_jitter=None,
    legend="brief", ax=None, **kwargs
):

    variables = _ScatterPlotter.get_semantics(locals())
    p = _ScatterPlotter(
        data=data, variables=variables,
        x_bins=x_bins, y_bins=y_bins,
        estimator=estimator, ci=ci, n_boot=n_boot,
        alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend,
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, order=style_order)

    if ax is None:
        ax = plt.gca()

    if not p.has_xy_data:
        return ax

    p._attach(ax)

    p.plot(ax, kwargs)

    return ax


scatterplot.__doc__ = dedent("""\
    Draw a scatter plot with possibility of several semantic groupings.

    {main_api_narrative}

    {relational_semantic_narrative}

    Parameters
    ----------
    {data_vars}
    hue : {long_form_var}
        Grouping variable that will produce points with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : {long_form_var}
        Grouping variable that will produce points with different sizes.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : {long_form_var}
        Grouping variable that will produce points with different markers.
        Can have a numeric dtype but will always be treated as categorical.
    {data}
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    {markers}
    {style_order}
    {{x,y}}_bins : lists or arrays or functions
        *Currently non-functional.*
    {units}
        *Currently non-functional.*
    {estimator}
        *Currently non-functional.*
    {ci}
        *Currently non-functional.*
    {n_boot}
        *Currently non-functional.*
    alpha : float
        Proportional opacity of the points.
    {{x,y}}_jitter : booleans or floats
        *Currently non-functional.*
    {legend}
    {ax_in}
    kwargs : key, value mappings
        Other keyword arguments are passed down to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    {ax_out}

    See Also
    --------
    lineplot : Show the relationship between two variables connected with
               lines to emphasize continuity.
    swarmplot : Draw a scatter plot with one categorical variable, arranging
                the points to show the distribution of values.

    Examples
    --------

    Draw a simple scatter plot between two variables:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns; sns.set()
        >>> import matplotlib.pyplot as plt
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.scatterplot(x="total_bill", y="tip", data=tips)

    Group by another variable and show the groups with different colors:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip", hue="time",
        ...                      data=tips)

    Show the grouping variable by varying both color and marker:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="time", style="time", data=tips)

    Vary colors and markers to show two different grouping variables:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="day", style="time", data=tips)

    Show a quantitative variable by varying the size of the points:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip", size="size",
        ...                      data=tips)

    Also show the quantitative variable by also using continuous colors:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      data=tips)

    Use a different continuous color map:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      palette=cmap,
        ...                      data=tips)

    Change the minimum and maximum point size and show all sizes in legend:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      sizes=(20, 200), palette=cmap,
        ...                      legend="full", data=tips)

    Use a narrower range of color map intensities:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="size", size="size",
        ...                      sizes=(20, 200), hue_norm=(0, 7),
        ...                      legend="full", data=tips)

    Vary the size with a categorical variable, and use a different palette:

    .. plot::
        :context: close-figs

        >>> cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      hue="day", size="smoker",
        ...                      palette="Set2",
        ...                      data=tips)

    Use a specific set of markers:

    .. plot::
        :context: close-figs

        >>> markers = {{"Lunch": "s", "Dinner": "X"}}
        >>> ax = sns.scatterplot(x="total_bill", y="tip", style="time",
        ...                      markers=markers,
        ...                      data=tips)

    Control plot attributes using matplotlib parameters:

    .. plot::
        :context: close-figs

        >>> ax = sns.scatterplot(x="total_bill", y="tip",
        ...                      s=100, color=".2", marker="+",
        ...                      data=tips)

    Pass data vectors instead of names in a data frame:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> ax = sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width,
        ...                      hue=iris.species, style=iris.species)

    Pass a wide-form dataset and plot against its index:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; plt.close("all")
        >>> index = pd.date_range("1 1 2000", periods=100,
        ...                       freq="m", name="date")
        >>> data = np.random.randn(100, 4).cumsum(axis=0)
        >>> wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
        >>> ax = sns.scatterplot(data=wide_df)

    Use :func:`relplot` to combine :func:`scatterplot` and :class:`FacetGrid`:
    This allows grouping within additional categorical variables. Using
    :func:`relplot` is safer than using :class:`FacetGrid` directly, as it
    ensures synchronization of the semantic mappings across facets.

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip",
        ...                  col="time", hue="day", style="day",
        ...                  kind="scatter", data=tips)


    """).format(**_relational_docs)


@_deprecate_positional_args
def relplot(
    *,
    x=None, y=None,
    hue=None, size=None, style=None, data=None,
    row=None, col=None,
    col_wrap=None, row_order=None, col_order=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=None, dashes=None, style_order=None,
    legend="brief", kind="scatter",
    height=5, aspect=1, facet_kws=None,
    units=None,
    **kwargs
):

    if kind == "scatter":

        plotter = _ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers

    elif kind == "line":

        plotter = _LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes

    else:
        err = "Plot kind {} not recognized".format(kind)
        raise ValueError(err)

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = (
            "relplot is a figure-level function and does not accept "
            "the ax= paramter. You may wish to try {}".format(kind + "plot")
        )
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # Use the full dataset to map the semantics
    p = plotter(
        data=data,
        variables=plotter.get_semantics(locals()),
        legend=legend,
    )
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    # Extract the semantic mappings
    if "hue" in p.variables:
        palette = p._hue_map.lookup_table
        hue_order = p._hue_map.levels
        hue_norm = p._hue_map.norm
    else:
        palette = hue_order = hue_norm = None

    if "size" in p.variables:
        sizes = p._size_map.lookup_table
        size_order = p._size_map.levels
        size_norm = p._size_map.norm

    if "style" in p.variables:
        style_order = p._style_map.levels
        if markers:
            markers = {k: p._style_map(k, "marker") for k in style_order}
        else:
            markers = None
        if dashes:
            dashes = {k: p._style_map(k, "dashes") for k in style_order}
        else:
            dashes = None
    else:
        markers = dashes = style_order = None

    # Now extract the data that would be used to draw a single plot
    variables = p.variables
    plot_data = p.plot_data
    plot_semantics = p.semantics

    # Define the common plotting parameters
    plot_kws = dict(
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        markers=markers, dashes=dashes, style_order=style_order,
        legend=False,
    )
    plot_kws.update(kwargs)
    if kind == "scatter":
        plot_kws.pop("dashes")

    # Define the named variables for plotting on each facet
    plot_variables = {key: key for key in p.variables}
    plot_kws.update(plot_variables)

    # Add the grid semantics onto the plotter
    grid_semantics = "row", "col"
    p.semantics = plot_semantics + grid_semantics
    p.assign_variables(
        data=data,
        variables=dict(
            x=x, y=y,
            hue=hue, size=size, style=style, units=units,
            row=row, col=col,
        ),
    )

    # Pass the row/col variables to FacetGrid with their original
    # names so that the axes titles render correctly
    grid_kws = {v: p.variables.get(v, None) for v in grid_semantics}
    full_data = p.plot_data.rename(columns=grid_kws)

    # Set up the FacetGrid object
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    facet_kws.update(grid_kws)
    g = FacetGrid(
        data=full_data,
        col_wrap=col_wrap, row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet_kws
    )

    # Draw the plot
    g.map_dataframe(func, **plot_kws)

    # Label the axes
    g.set_axis_labels(
        variables.get("x", None), variables.get("y", None)
    )

    # Show the legend
    if legend:
        # Replace the original plot data so the legend uses
        # numeric data with the correct type
        p.plot_data = plot_data
        p.add_legend_data(g.axes.flat[0])
        if p.legend_data:
            g.add_legend(legend_data=p.legend_data,
                         label_order=p.legend_order)

    return g


relplot.__doc__ = dedent("""\
    Figure-level interface for drawing relational plots onto a FacetGrid.

    This function provides access to several different axes-level functions
    that show the relationship between two variables with semantic mappings
    of subsets. The ``kind`` parameter selects the underlying axes-level
    function to use:

    - :func:`scatterplot` (with ``kind="scatter"``; the default)
    - :func:`lineplot` (with ``kind="line"``)

    Extra keyword arguments are passed to the underlying function, so you
    should refer to the documentation for each to see kind-specific options.

    {main_api_narrative}

    {relational_semantic_narrative}

    After plotting, the :class:`FacetGrid` with the plot is returned and can
    be used directly to tweak supporting plot details or add other layers.

    Note that, unlike when using the underlying plotting functions directly,
    data must be passed in a long-form DataFrame with variables specified by
    passing strings to ``x``, ``y``, and other parameters.

    Parameters
    ----------
    x, y : names of variables in ``data``
        Input data variables; must be numeric.
    hue : name in ``data``, optional
        Grouping variable that will produce elements with different colors.
        Can be either categorical or numeric, although color mapping will
        behave differently in latter case.
    size : name in ``data``, optional
        Grouping variable that will produce elements with different sizes.
        Can be either categorical or numeric, although size mapping will
        behave differently in latter case.
    style : name in ``data``, optional
        Grouping variable that will produce elements with different styles.
        Can have a numeric dtype but will always be treated as categorical.
    {data}
    row, col : names of variables in ``data``, optional
        Categorical variables that will determine the faceting of the grid.
    {col_wrap}
    row_order, col_order : lists of strings, optional
        Order to organize the rows and/or columns of the grid in, otherwise the
        orders are inferred from the data objects.
    {palette}
    {hue_order}
    {hue_norm}
    {sizes}
    {size_order}
    {size_norm}
    {legend}
    kind : string, optional
        Kind of plot to draw, corresponding to a seaborn relational plot.
        Options are {{``scatter`` and ``line``}}.
    {height}
    {aspect}
    facet_kws : dict, optional
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

    Draw a single facet to use the :class:`FacetGrid` legend placement:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns
        >>> sns.set(style="ticks")
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.relplot(x="total_bill", y="tip", hue="day", data=tips)

    Facet on the columns with another variable:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip",
        ...                 hue="day", col="time", data=tips)

    Facet on the columns and rows:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="day",
        ...                 col="time", row="sex", data=tips)

    "Wrap" many column facets into multiple rows:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="time",
        ...                 col="day", col_wrap=2, data=tips)

    Use multiple semantic variables on each facet with specified attributes:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="total_bill", y="tip", hue="time", size="size",
        ...                 palette=["b", "r"], sizes=(10, 100),
        ...                 col="time", data=tips)

    Use a different kind of plot:

    .. plot::
        :context: close-figs

        >>> fmri = sns.load_dataset("fmri")
        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                 hue="event", style="event", col="region",
        ...                 kind="line", data=fmri)

    Change the size of each facet:

    .. plot::
        :context: close-figs

        >>> g = sns.relplot(x="timepoint", y="signal",
        ...                 hue="event", style="event", col="region",
        ...                 height=5, aspect=.7, kind="line", data=fmri)

    """).format(**_relational_docs)
