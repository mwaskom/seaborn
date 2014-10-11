"""Functions to visualize matrices of data."""
import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .palettes import cubehelix_palette
from .utils import despine, axis_ticklabels_overlap


class _HeatMapper(object):
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""
    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cbar, cbar_kws):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # Reverse the rows so the plot looks like the matrix
        plot_data = plot_data[::-1]
        data = data.ix[::-1]

        # Get good names for the rows and columns
        if isinstance(data.columns, pd.MultiIndex):
            xtl = ["-".join(map(str, i)) for i in data.columns.values]
            self.xticklabels = xtl
            xlabel = "-".join(map(str, data.columns.names))
        else:
            self.xticklabels = data.columns
            xlabel = data.columns.name
        if isinstance(data.index, pd.MultiIndex):
            ytl = ["-".join(map(str, i)) for i in data.index.values]
            self.yticklabels = ytl
            ylabel = "-".join(map(str, data.index.names))
        else:
            self.yticklabels = data.index
            ylabel = data.index.name

        # Get good names for the axis labels
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data
        self.annot = annot
        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws

    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        if vmin is None:
            vmin = np.percentile(plot_data, 2) if robust else plot_data.min()
        if vmax is None:
            vmax = np.percentile(plot_data, 98) if robust else plot_data.max()

        # Simple heuristics for whether these data should  have a divergent map
        divergent = ((vmin < 0) and (vmax > 0)) or center is not None

        # Now set center to 0 so math below makes sense
        if center is None:
            center = 0

        # A divergent map should be symmetric around the center value
        if divergent:
            vlim = max(abs(vmin - center), abs(vmax - center))
            vmin, vmax = -vlim, vlim
        self.divergent = divergent

        # Now add in the centering value and set the limits
        vmin += center
        vmax += center
        self.vmin = vmin
        self.vmax = vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if divergent:
                self.cmap = "RdBu_r"
            else:
                self.cmap = cubehelix_palette(light=.95, as_cmap=True)
        else:
            self.cmap = cmap

    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        xpos, ypos = np.meshgrid(ax.get_xticks(), ax.get_yticks())
        for x, y, val, color in zip(xpos.flat, ypos.flat,
                                    mesh.get_array(),  mesh.get_facecolors()):
            _, l, _ = colorsys.rgb_to_hls(*color[:3])
            text_color = ".15" if l > .5 else "w"
            val = ("{:" + self.fmt + "}").format(val)
            ax.text(x, y, val, color=text_color,
                    ha="center", va="center", **self.annot_kws)

    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap
        mesh = ax.pcolormesh(self.plot_data, vmin=self.vmin, vmax=self.vmax,
                             cmap=self.cmap, **kws)

        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Add row and column labels
        nx, ny = self.data.T.shape
        ax.set(xticks=np.arange(nx) + .5, yticks=np.arange(ny) + .5)
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation="vertical")

        # Possibly rotate them if they overlap
        plt.draw()
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Annotate the cells with the formatted values
        if self.annot:
            self._annotate_heatmap(ax, mesh)

        # Possibly add a colorbar
        if self.cbar:
            ticker = mpl.ticker.MaxNLocator(6)
            cb = ax.figure.colorbar(mesh, cax, ax,
                                    ticks=ticker, **self.cbar_kws)
            cb.outline.set_linewidth(0)


def heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=False, fmt=".2g", annot_kws=None,
            linewidths=.5, linecolor="white",
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=False, ax=None, **kwargs):
    """Plot rectangular data as a color-encoded matrix.

    This function tries to infer a good colormap to use from the data, but
    this is not guaranteed to work, so take care to make sure the kind of
    colormap (sequential or diverging) and its limits are appropriate.

    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        one of these values may be ignored.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space. If not provided, this
        will be either a cubehelix map (if the function infers a sequential
        dataset) or ``RdBu_r`` (if the function infers a diverging dataset).
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool, optional
        If True, write the data value in each cell.
    fmt : string, optional
        String formatting code to use when ``annot`` is True.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for ``ax.text`` when ``annot`` is True.
    linewidths : float, optional
        Width of the lines that divide each cell.
    linecolor : color, optional
        Color of the lines that divide each cell.
    cbar : boolean, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for `fig.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : boolean, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to ``ax.pcolormesh``.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    """
    # Initialize the plotter object
    plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws)

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax
