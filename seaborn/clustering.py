import numpy as np

from seaborn.utils import despine
import warnings
import pdb

def _get_width_ratios(shape, side_colors,
                     colorbar_loc, dimension, side_colors_ratio=0.05):

    """
    Figures out the ratio of each subfigure within the larger figure.
    The dendrograms currently are 2*half_dendrogram, which is a proportion of
    the dataframe shape. Right now, this only supports the colormap in
    the upper left. The full figure map looks like:

    0.1  0.1  0.05    1.0
    0.1  cb              column
    0.1                  dendrogram
    0.05                 col colors
    | r   d     r
    | o   e     o
    | w   n     w
    |     d
    1.0|     r     c     heatmap
    |     o     o
    |     g     l
    |     r     o
    |     a     r
    |     m     s

    The colorbar is half_dendrogram of the whitespace in the corner between
    the row and column dendrogram. Otherwise, it's too big and its
    corners touch the heatmap, which I didn't like.

    For example, if there are side_colors, need to provide an extra value
    in the ratio tuples, with the width side_colors_ratio. But if there
    aren't any side colors, then the tuple is of size 3 (half_dendrogram,
    half_dendrogram, 1.0), and if there are then the tuple is of size 4 (
    half_dendrogram, half_dendrogram, 0.05, 1.0)

    TODO: Add option for lower right corner.

    side_colors: list of colors, or empty list
    colorbar_loc: string
    Where the colorbar will be. Valid locations: 'upper left', 'right',
    'bottom'
    dimension: string
    Which dimension are we trying to find the axes locations for.
    Valid strings: 'height', 'width'
    side_colors_ratio: float
    How much space the side colors labeling the rows or columns
    should take up. Default 0.05

    Returns
    -------
    ratios: list of ratios
    Ratios of axes on the figure at this dimension
    """
    i = 0 if dimension == 'height' else 1
    half_dendrogram = shape[i] * 0.1 / shape[i]
    if colorbar_loc not in ('upper left', 'right'):
        raise AssertionError("{} is not a valid 'colorbar_loc' (valid: "
                             "'upper left', 'right', 'bottom')".format(
            colorbar_loc))
    if dimension not in ('height', 'width'):
        raise AssertionError("{} is not a valid 'dimension' (valid: "
                             "'height', 'width')".format(
            dimension))

    ratios = [half_dendrogram, half_dendrogram]
    if side_colors:
        ratios += [side_colors_ratio]

    if (colorbar_loc == 'right' and dimension == 'width') or\
            (dimension == 'height'):
        return ratios + [1, 0.05]
    else:
        return ratios + [1]

def _color_list_to_matrix_and_cmap(colors, ind, row=True):
    """Turns a list of colors into a numpy matrix and matplotlib colormap
    For 'heatmap()'
    This only works for 1-column color lists..
    TODO: Support multiple color labels on an element in the heatmap

    These arguments can now be plotted using matplotlib.pcolormesh(matrix,
    cmap) and the provided colors will be plotted.

    Parameters
    ----------
    colors : list of matplotlib colors
        Colors to label the rows or columns of a dataframe.
    ind : list of ints
        Ordering of the rows or columns, to reorder the original colors
    row : bool
        Is this to label the rows or columns? Default True.

    Returns
    -------
    matrix : numpy.array
        A numpy array of integer values, where each corresponds to a color
        from the originally provided list of colors
    cmap : matplotlib.colors.ListedColormap

    """
    import matplotlib as mpl

    colors = set(colors)
    col_to_value = dict((col, i) for i, col in enumerate(colors))
    matrix = np.array([col_to_value[col] for col in colors])[ind]
    # Is this row-side or column side?
    if row:
        new_shape = (len(colors), 1)
    else:
        new_shape = (1, len(colors))
    matrix = matrix.reshape(new_shape)

    cmap = mpl.colors.ListedColormap(colors)
    return matrix, cmap

def _get_linkage_function(shape, use_fastcluster):
    """
    Parameters
    ----------
    shape : tuple
        (nrow, ncol) tuple of the shape of the dataframe
    use_fastcluster : bool
        Whether to use fastcluster (3rd party module) for clustering,
        which is faster than the default scipy.cluster.hierarchy.linkage module

    Returns
    -------
    linkage_function : function
        Linkage function to use for clustering

    .. warning:: If either the number of columns or rows exceeds 1000,
    this wil try to import fastcluster, and raise a warning if it does not
    exist. Vanilla scipy.cluster.hierarchy.linkage will take a long time on
    these matrices.
    """
    import scipy.cluster.hierarchy as sch
    if (shape[0] > 1000 or shape[1] > 1000) or use_fastcluster:
        try:
            import fastcluster

            linkage_function = fastcluster.linkage
        except ImportError:
            raise warnings.warn('Module "fastcluster" not found. The '
                                'dataframe '
                                'provided has '
                                'shape {}, and one '
                                'of the dimensions has greater than 1000 '
                                'variables. Calculating linkage on such a '
                                'matrix will take a long time with vanilla '
                                '"scipy.cluster.hierarchy.linkage", and we '
                                'suggest fastcluster for such large datasets' \
                                    .format(shape), RuntimeWarning)
    else:
        linkage_function = sch.linkage
    return linkage_function


def _plot_dendrogram(fig, kws, gridspec, linkage, shape, orientation='top'):
    """
    Parameters
    ----------


    Returns
    -------


    Raises
    ------
    """
    import scipy.cluster.hierarchy as sch
    almost_black = '#262626'
    ax = fig.add_subplot(gridspec)
    if kws['cluster']:
        dendrogram = sch.dendrogram(linkage,
                                        color_threshold=np.inf,
                                        color_list=[almost_black])
    else:
        dendrogram = {'leaves': list(range(shape))}

    # Can this hackery be avoided?
    despine(ax=ax, bottom=True, left=True)
    ax.set_axis_bgcolor('white')
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xticks([])
    return dendrogram

def _plot_sidecolors():
    """
    Parameters
    ----------


    Returns
    -------


    Raises
    ------
    """
    pass

def clusterplot(df,
                pivot_kws=None,
            title=None,
            title_fontsize=12,
            color_scale='linear',
            linkage_method='average',
            metric='euclidean',
            figsize=None,
            pcolormesh_kws=None,
            row_kws=None,
            col_kws=None,
            plot_df=None,
            colorbar_kws=None,
            use_fastcluster=False):
    """Plot a clustered heatmap of a pandas DataFrame
    @author Olga Botvinnik olga.botvinnik@gmail.com

    This is liberally borrowed (with permission) from http://bit.ly/1eWcYWc
    Many thanks to Christopher DeBoever and Mike Lovci for providing heatmap
    guidance.

    Parameters
    ----------
    df: DataFrame
        Data for clustering. Should be a dataframe with no NAs. If you
        still want to plot a dataframe with NAs, provide a non-NA dataframe
        to df (with NAs replaced by 0 or something by your choice) and your
        NA-full dataframe to plot_df
    pivot_kws : dict
        If the data is in "tidy" format, reshape the data with these pivot
        keyword arguments
    title: string, optional
        Title of the figure. Default None
    title_fontsize: int, optional
        Size of the plot title. Default 12
    colorbar_label: string, optional
        Label the colorbar, e.g. "density" or "gene expression". Default
        "values"
    col_side_colors: list of matplotlib colors, optional
        Label the columns with the group they are in with a color. Useful for
        evaluating whether samples within a group are clustered together.
    row_side_colors: list of matplotlib colors, optional
        Label the rows (index of a dataframe) with the group they are in with a
        color. Useful for evaluating whether samples within a group are
        clustered together.
    color_scale: string, 'log' or 'linear'
        How to scale the colors plotted in the heatmap. Default "linear"
    cmap: matplotlib colormap
        How to color the data values. Default is YlGnBu ('yellow-green-blue')
        for positive- or negative-only data, and RdBu_r ('red-blue-reversed'
        so blue is "cold," negative numbers and red is "hot," positive
        numbers) for divergent data, i.e. if your data has both positive and
        negative values.
    linkage_method: string
        Which linkage method to use for calculating clusters.
        See scipy.cluster.hierarchy.linkage documentation for more information:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        Default "average"
    figsize: tuple of two ints
        Size of the figure to create. Default is a function of the dataframe
        size.
    label_rows: bool or list of strings, optional
        Label the rows with the index labels of the dataframe. Optionally
        provide another list of strings of the same length as the number of
        rows. Can also be False. Default True.
    label_cols: bool or list of strings, optional
        Label the columns with the column labels of the dataframe. Optionally
        provide another list of strings of the same length as the number of
        column. Can also be False. Default True.
    pcolormesh_kws['vmin']: int or float
        Minimum value to plot on the heatmap. Default the smallest value in
        the dataframe.
    vmax: int or float
        Maximum value to plot on the heatmap. Default the largest value in
        the dataframe.
    xlabel_fontsize: int
        Size of the x tick labels on the heatmap. Default 12
    ylabel_fontsize: int
        Size of the y tick labels on the heatmap. Default 10
    cluster_cols: bool or dendrogram in the dict format returned by
    scipy.cluster.hierarchy.dendrogram
        Whether or not to cluster the columns of the data. Default True. Can
        provide your own dendrogram if you have separately performed
        clustering.
    cluster_rows: bool or dendrogram in the dict format returned by
    scipy.cluster.hierarchy.dendrogram
        Whether or not to cluster the columns of the data. Default True. Can
        provide your own dendrogram if you have separately performed
        clustering.
    linewidth: int
        Width of lines to draw on top of the heatmap to separate the cells
        from one another. Default 0 (no lines)
    edgecolor: matplotlib color
        Color of the lines to draw on top of the heatmap to separate cells
        from one another. Default "white"
    plot_df: Dataframe
        The dataframe to plot. May have NAs, unlike the "df" argument.
    colorbar_ticklabels_fontsize: int
        Size of the tick labels on the colorbar
    use_fastcluster: bool
        Whether or not to use the "fastcluster" module in Python,
        which calculates linkage several times faster than scipy.cluster.hierachy
        Default False except for datasets with more than 1000 rows or columns.
    metric: string
        Distance metric to use for the data. Default is "euclidean." See
        scipy.spatial.distance.pdist documentation for more options
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    pcolormesh_kws: dict, arguments to pcolormesh (like imshow and pcolor but faster)
        For example, dict(linewidth=0, edgecolor='white', cmap=None, pcolormesh_kws['vmin']=None,

        vmax=None)
    Returns
    -------
    fig: matplotlib Figure
        Figure containing multiple axes where the clustered heatmap is plotted.
    row_dendrogram: dict
        dict with keys 'leaves', 'icoords' (coordinates of the cluster nodes
        along the data, here the y-axis coords), 'dcoords' (coordinates of the
        cluster nodes along the dendrogram height, here the x-axis coords)
    col_dendrogram: dict
        dict with keys 'leaves', 'icoords' (coordinates of the cluster nodes
        along the data, here the x-axis coords), 'dcoords' (coordinates of the
        cluster nodes along the dendrogram height, here the y-axis coords)
    """
    #@return: fig, row_dendrogram, col_dendrogram
    #@rtype: matplotlib.figure.Figure, dict, dict
    #@raise TypeError:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scipy.spatial.distance as distance
    import scipy.cluster.hierarchy as sch
    import matplotlib as mpl
    from collections import Iterable

    if pivot_kws is not None:
        df = df.pivot(pivot_kws)

    almost_black = '#262626'
    sch.set_link_color_palette([almost_black])
    if plot_df is None:
        plot_df = df

    if (plot_df.index != df.index).any():
        raise ValueError('plot_df must have the exact same indices as df')
    if (plot_df.columns != df.columns).any():
        raise ValueError('plot_df must have the exact same columns as df')
        # make norm

    if row_kws is None:
        row_kws = dict(cluster=True,
             side_colors=None,
             label=True,
             fontsize=12,
             linkage_matrix=None)
    if col_kws is None:
        col_kws = dict(cluster=True,
             side_colors=None,
             label=True,
             fontsize=12,
             linkage_matrix=None)
    if colorbar_kws is None:
        colorbar_kws = dict(ticklabels_fontsize=10,
             loc='upper left',
             label='values')
    if pcolormesh_kws is None:
        pcolormesh_kws = {}
    vmin = pcolormesh_kws.setdefault('vmin', None)
    vmax = pcolormesh_kws.setdefault('vmax', None)

    # Check if the matrix has values both above and below zero, or only above
    # or only below zero. If both above and below, then the data is
    # "divergent" and we will use a colormap with 0 centered at white,
    # negative values blue, and positive values red. Otherwise, we will use
    # the YlGnBu colormap.
    divergent = (df.max().max() > 0 and df.min().min() < 0) and not \
        color_scale == 'log'

    if color_scale == 'log':
       if pcolormesh_kws['vmin'] is None:
           pcolormesh_kws['vmin'] = max(df.dropna(how='all').min().dropna().min(), 1e-10)
       if pcolormesh_kws['vmax'] is None:
           pcolormesh_kws['vmax'] = np.ceil(df.dropna(how='all').max().dropna()
                                         .max())
       pcolormesh_kws['norm'] = mpl.colors.LogNorm(pcolormesh_kws['vmin'],
                                                   vmax)

    print "pcolormesh_kws['vmin']", pcolormesh_kws['vmin']

    if 'cmap' not in pcolormesh_kws:
        cmap = mpl.cm.RdBu_r if divergent else mpl.cm.YlGnBu
        cmap.set_bad('white')
        pcolormesh_kws['cmap'] = cmap

    if divergent:
        abs_max = abs(df.max().max())
        abs_min = abs(df.min().min())
        vmaxx = max(abs_max, abs_min)
        pcolormesh_kws['vmin'] = -vmaxx
        pcolormesh_kws['vmax'] = vmaxx
        norm = mpl.colors.Normalize(vmin=-vmaxx, vmax=vmaxx)



    # TODO: Add optimal leaf ordering for clusters
    # TODO: if color_scale is 'log', should distance also be on np.log(df)?
    # calculate pairwise distances for rows
    # if color_scale == 'log':
    #     df = np.log10(df)

    if row_kws['linkage_matrix'] is None:
        linkage_function = _get_linkage_function(df.shape, use_fastcluster)
        row_pairwise_dists = distance.squareform(distance.pdist(df,
                                                                metric=metric))
        row_linkage = linkage_function(row_pairwise_dists,
                                       method=linkage_method)
    else:
        row_linkage = row_kws['linkage_matrix']

    # calculate pairwise distances for columns
    if col_kws['linkage_matrix'] is None:
        linkage_function = _get_linkage_function(df.shape, use_fastcluster)
        col_pairwise_dists = distance.squareform(distance.pdist(df.T,
                                                                metric=metric))
        # cluster
        col_linkage = linkage_function(col_pairwise_dists,
                                       method=linkage_method)
    else:
        col_linkage = col_kws['linkage_matrix']

    # heatmap with row names
    width_ratios = _get_width_ratios(df.shape,
                                    row_kws['side_colors'],
                                    colorbar_kws['loc'], dimension='width')
    height_ratios = _get_width_ratios(df.shape,
                                     col_kws['side_colors'],
                                     colorbar_kws['loc'], dimension='height')
    nrows = 3 if col_kws['side_colors'] is None else 4
    ncols = 3 if row_kws['side_colors'] is None else 4


    if figsize is None:
        width = df.shape[1] * 0.25
        height = min(df.shape[0] * .75, 40)
        figsize = (width, height)

    edgecolor = pcolormesh_kws.setdefault('edgecolor', 'none')
    linewidth = pcolormesh_kws.setdefault('linewidth', 0)

    fig = plt.figure(figsize=figsize)
    heatmap_gridspec = \
        gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios)

    ### col dendrogram ###
    col_dendrogram_ax = fig.add_subplot(heatmap_gridspec[1, ncols - 1])
    if col_kws['cluster']:
        col_dendrogram = sch.dendrogram(col_linkage,
                                        color_threshold=np.inf,
                                        color_list=[almost_black])
    else:
        col_dendrogram = {'leaves': list(range(df.shape[1]))}

    # Can this hackery be avoided?
    despine(ax=col_dendrogram_ax, bottom=True, left=True)
    col_dendrogram_ax.set_axis_bgcolor('white')
    col_dendrogram_ax.grid(False)
    col_dendrogram_ax.set_yticks([])
    col_dendrogram_ax.set_xticks([])

    # TODO: Allow for array of color labels
    ### col colorbar ###
    if col_kws['side_colors'] is not None:
        column_colorbar_ax = fig.add_subplot(heatmap_gridspec[2, ncols - 1])
        col_side_matrix, col_cmap = _color_list_to_matrix_and_cmap(
            col_kws['side_colors'],
            ind=col_dendrogram['leaves'],
            row=False)
        column_colorbar_ax_pcolormesh = column_colorbar_ax.pcolormesh(
            col_side_matrix, cmap=col_cmap,
            edgecolor=edgecolor,
            linewidth=linewidth)
        column_colorbar_ax.set_xlim(0, col_side_matrix.shape[1])
        column_colorbar_ax.set_yticks([])
        column_colorbar_ax.set_xticks([])

    ### row dendrogram ##
    row_dendrogram_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, 1])
    if row_kws['cluster']:
        row_dendrogram = \
            sch.dendrogram(row_linkage,
                           color_threshold=np.inf,
                           orientation='right',
                           color_list=[almost_black])
    else:
        row_dendrogram ={'leaves': list(range(df.shape[0]))}
    despine(ax=row_dendrogram_ax, bottom=True, left=True)
    row_dendrogram_ax.set_axis_bgcolor('white')
    row_dendrogram_ax.grid(False)
    row_dendrogram_ax.set_yticks([])
    row_dendrogram_ax.set_xticks([])


    ### row colorbar ###
    if row_kws['side_colors'] is not None:
        row_colorbar_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, 2])
        row_side_matrix, row_cmap = _color_list_to_matrix_and_cmap(
            row_kws['side_colors'],
            ind=row_dendrogram['leaves'],
            row=True)
        row_colorbar_ax.pcolormesh(row_side_matrix, cmap=row_cmap,
                                   edgecolors=edgecolor,
                                   linewidth=linewidth)
        row_colorbar_ax.set_ylim(0, row_side_matrix.shape[0])
        row_colorbar_ax.set_xticks([])
        row_colorbar_ax.set_yticks([])

    ### heatmap ####
    heatmap_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, ncols - 1])
    heatmap_ax_pcolormesh = \
        heatmap_ax.pcolormesh(plot_df.ix[row_dendrogram['leaves'],
                                         col_dendrogram['leaves']].values,
                              **pcolormesh_kws)

    heatmap_ax.set_ylim(0, df.shape[0])
    heatmap_ax.set_xlim(0, df.shape[1])

    ## row labels ##
    label_rows = row_kws.setdefault('label', True)
    if isinstance(label_rows, Iterable):
        if len(row_kws['label']) == df.shape[0]:
            yticklabels = row_kws['label']
            label_rows = True
        else:
            raise AssertionError("Length of 'row_kws['label_rows']' must be "
                                 "the "
                                 "' \
                                                            'same as "
                                 "df.shape[0] (len(row_kws['label_rows'])={}, df.shape["
                                 "0]={})".format(len(row_kws['label']), df.shape[0]))
    elif label_rows:
        yticklabels = df.index
    else:
        heatmap_ax.set_yticklabels([])

    if label_rows:
        yticklabels = [yticklabels[i] for i in row_dendrogram['leaves']]
        despine(ax=heatmap_ax, bottom=True, left=True)
        heatmap_ax.set_yticks(np.arange(df.shape[0]) + 0.5)
        heatmap_ax.yaxis.set_ticks_position('right')
        heatmap_ax.set_yticklabels(yticklabels)

    # Add title if there is one:
    if title is not None:
        col_dendrogram_ax.set_title(title, fontsize=title_fontsize)

    ## col labels ##
    label_cols = col_kws.setdefault('label', True)
    if isinstance(col_kws['label'], Iterable):
        if len(col_kws['label']) == df.shape[1]:
            xticklabels = col_kws['label']
            label_cols = True
        else:
            raise AssertionError("Length of 'label_cols' must be the same as "
                                 "df.shape[1] (len(label_cols)={}, df.shape["
                                 "1]={})".format(len(col_kws['label']), df.shape[1]))
    elif col_kws['label']:
        xticklabels = df.columns
    else:
        heatmap_ax.set_xticklabels([])

    if label_cols:
        xticklabels = [xticklabels[i] for i in col_dendrogram['leaves']]
        heatmap_ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        xticklabels = heatmap_ax.set_xticklabels(xticklabels)
        # rotate labels 90 degrees
        for label in xticklabels:
            label.set_rotation(90)

    # remove the tick lines
    for l in heatmap_ax.get_xticklines() + heatmap_ax.get_yticklines():
        l.set_markersize(0)

    ### scale colorbar ###
    scale_colorbar_ax = fig.add_subplot(
        heatmap_gridspec[0:(nrows - 1),
        0]) # colorbar for scale in upper left corner

    cb = fig.colorbar(heatmap_ax_pcolormesh,
                      cax=scale_colorbar_ax)
    cb.set_label(colorbar_kws['label'])

    if color_scale == 'log':
        #TODO: get at most 3 ticklabels showing on the colorbar for log-scaled
        pass
        # tick_locator = mpl.ticker.LogLocator(numticks=3)
        # pdb.set_trace()
        # tick_locator.tick_values(pcolormesh_kws['vmin'],
        #                          pcolormesh_kws['vmax'])
    else:
        tick_locator = mpl.ticker.MaxNLocator(nbins=2,
                                              symmetric=divergent,
                                              prune=None, trim=False)
        cb.ax.set_yticklabels(tick_locator.bin_boundaries(pcolormesh_kws[
                                                                 'vmin'],
                                    pcolormesh_kws['vmax']))
        cb.ax.yaxis.set_major_locator(tick_locator)

    # move ticks to left side of colorbar to avoid problems with tight_layout
    cb.ax.yaxis.set_ticks_position('left')
    cb.outline.set_linewidth(0)
    # pdb.set_trace()

    ## Make colorbar narrower
    #xmin, xmax, ymin, ymax = cb.ax.axis()
    #cb.ax.set_xlim(xmin, xmax/0.2)

    # make colorbar labels smaller
    # yticklabels = cb.ax.yaxis.get_ticklabels()
    # for t in yticklabels:
    #     t.set_fontsize(colorbar_kws['ticklabels_fontsize'])

    fig.tight_layout()
    #despine(fig, top=True, bottom=True, left=True, right=True)
    return fig, row_dendrogram, col_dendrogram
