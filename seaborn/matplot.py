import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import despine


def heatmap(data, xticklabels=True, yticklabels=True, vmin=None, vmax=None,
            cmap=None, center_value=0, yticklabels_rotation='horizontal',
            xticklabels_rotation='vertical', colorbar=True,
            colorbar_ax=None, ax=None,
            fig=None, colorbar_orientation='vertical', colorbar_label='',
            colorbar_ticklabel_fontsize=8,
            edgecolor='white', linewidth=.5, **kwargs):
    """
    Use for large datasets

    Parameters
    ----------
    data : pandas.DataFrame
        The data to heatmap-ize
    vmin : float
        Minimum value to plot. Default the minimum of the data.
    vmax : float
        Maximum value to plot. Default the maximum of the data.
    cmap : matplotlib.cm
        Colormap to use. Default "YlGnBu" (yellow-green-blue) for
        non-divergent data, data that is only above or below 0, and "RdBu_r"
        (red-blue, reversed) for data that has values both above and below 0.
    center_value : float
        For divergent data, which value should be the center, exactly in the
        middle between the two extreme colors. For example if you have data
        above and below zero, but you want the white part of the colormap to
        be equal to 10 rather than 0, then specify 'center_value=10'.
    ax : matplotlib.Axes
        Where to plot the heatmap
    fig : matplotlib.Figure
        Where to place the heatmap and colorbar axes
    xticklabels : bool or list-like
        If True, use the column names of "data" to plot. Otherwise use
        the labels provided. Default True.
    yticklabels : bool or list-like
        If True, use the index (row names) of "data" to plot. Otherwise use
        the labels provided. Default True
    xticklabels_rotation : 'vertical' | 'horizontal'
        How to rotate the yticklabels. Default "vertical"
    yticklabels_rotation : 'horizontal' | 'vertical'
        How to rotate the yticklabels. Default "horizontal"
    colorbar : bool
        Whether or not to plot the colorbar
    colorbar_ax : matplotlib.Axes
        Where to place the colorbar ax. Default None.
    colorbar_orientation : 'vertical' | 'horizontal'
        How to rotate the colorbar. Default 'horizontal'
    edgecolor : matplotlib color
        Color of lines between matrix cells
    linewidth : float
        Width of lines between matrix cells. Default 0.5

    Returns
    -------
    p : matplotlib.pyplot.pcolormesh instance
        Returns a pcolormesh instance
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        array = data.values
    else:
        array = data
    vmin = array.min() if vmin is None else vmin
    vmax = array.max() if vmax is None else vmax

    # If this data has both negative and positive values, call it divergent
    divergent_data = False
    if vmax > 0 and vmin < 0:
        divergent_data = True
        vmax += center_value
        vmin += center_value

    # If we have both negative and positive values, use a divergent colormap
    if cmap is None:
        # Check if this is divergent
        if divergent_data:
            cmap = mpl.cm.RdBu_r
        else:
            cmap = mpl.cm.YlGnBu

    p = ax.pcolormesh(array, cmap=cmap, vmin=vmin, vmax=vmax,
                      edgecolor=edgecolor, linewidth=linewidth, **kwargs)

    # Get rid of ALL axes
    despine(bottom=True, left=True)

    # If no ticklabels provided, use the dataframe labels
    if isinstance(xticklabels, bool):
        if xticklabels:
            xticklabels = data.columns
        else:
            xticklabels = []
    if isinstance(yticklabels, bool):
        if yticklabels:
            yticklabels = data.index
        else:
            yticklabels = []

    if any(xticklabels):
        xticks = np.arange(0.5, data.shape[1] + 0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation)

    if any(yticklabels):
        yticks = np.arange(0.5, data.shape[0] + 0.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=yticklabels_rotation)

    # Show the scale of the colorbar
    if colorbar:
        cb = fig.colorbar(p, cax=colorbar_ax, use_gridspec=True,
                          orientation=colorbar_orientation,
                          label=colorbar_label)
        tick_locator = mpl.ticker.MaxNLocator(nbins=2,
                                              symmetric=divergent_data,
                                              prune=None, trim=False)
        if 'horizontal'.startswith(colorbar_orientation):
            cb.ax.set_xticklabels(tick_locator.bin_boundaries(vmin, vmax))
            cb.ax.xaxis.set_major_locator(tick_locator)
        else:
            cb.ax.set_yticklabels(tick_locator.bin_boundaries(vmin, vmax))
            cb.ax.yaxis.set_major_locator(tick_locator)
            cb.ax.yaxis.set_ticks_position('right')

        # move ticks to left side of colorbar to avoid problems with
        # tight_layout
        # cb.ax.yaxis.set_ticks_position('right')
        if colorbar_ticklabel_fontsize is not None:
            cb.ax.tick_params(labelsize=colorbar_ticklabel_fontsize)
        cb.outline.set_linewidth(0)
    return p
