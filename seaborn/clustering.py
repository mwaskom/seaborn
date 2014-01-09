

def _color_list_to_matrix_and_cmap(color_list, ind, row=True):
    """
    For 'heatmap()'
    This only works for 1-column color lists..
    TODO: Support multiple color labels on an element in the heatmap
    """
    import matplotlib as mpl

    colors = set(color_list)
    col_to_value = dict((col, i) for i, col in enumerate(colors))

    #     ind = column_dendrogram_distances['leaves']
    matrix = np.array([col_to_value[col] for col in color_list])[ind]
    # Is this row-side or column side?
    if row:
        new_shape = (len(color_list), 1)
    else:
        new_shape = (1, len(color_list))
    matrix = matrix.reshape(new_shape)

    cmap = mpl.colors.ListedColormap(colors)
    return matrix, cmap


def heatmap(df,
            title=None,
            title_fontsize=12,
            colorbar_label='values',
            col_side_colors=None,
            row_side_colors=None,
            color_scale='linear',
            cmap=None,
            linkage_method='average',
            figsize=None,
            label_rows=True,
            label_cols=True,
            vmin=None,
            vmax=None,
            xlabel_fontsize=12,
            ylabel_fontsize=10,
            cluster_cols=True,
            cluster_rows=True,
            linewidth=0,
            edgecolor='white',
            plot_df=None,
            colorbar_ticklabels_fontsize=10,
            colorbar_loc="upper left",
            use_fastcluster=False,
            metric='euclidean'):
    """
    @author Olga Botvinnik olga.botvinnik@gmail.com

    This is liberally borrowed (with permission) from http://bit.ly/1eWcYWc
    Many thanks to Christopher DeBoever and Mike Lovci for providing heatmap
    guidance.




    :param title_fontsize:
    :param colorbar_ticklabels_fontsize:
    :param colorbar_loc: Can be 'upper left' (in the corner), 'right',
    or 'bottom'


    :param df: The dataframe you want to cluster on
    :param title: Title of the figure
    :param colorbar_label: What to colorbar (color scale of the heatmap)
    :param col_side_colors: Label the columns with a color
    :param row_side_colors: Label the rows with a color
    :param color_scale: Either 'linear' or 'log'
    :param cmap: A matplotlib colormap, default is mpl.cm.Blues_r if data is
    sequential, or mpl.cm.RdBu_r if data is divergent (has both positive and
    negative numbers)
    :param figsize: Size of the figure. The default is a function of the
    dataframe size.
    :param label_rows: Can be boolean or a list of strings, with exactly the
    length of the number of rows in df.
    :param label_cols: Can be boolean or a list of strings, with exactly the
    length of the number of columns in df.
    :param col_labels: If True, label with df.columns. If False, unlabeled.
    Else, this can be an iterable to relabel the columns with labels of your own
    choosing. This is helpful if you have duplicate column names and pandas
    won't let you reindex it.
    :param row_labels: If True, label with df.index. If False, unlabeled.
    Else, this can be an iterable to relabel the row names with labels of your
    own choosing. This is helpful if you have duplicate index names and pandas
    won't let you reindex it.
    :param xlabel_fontsize: Default 12pt
    :param ylabel_fontsize: Default 10pt
    :param cluster_cols: Boolean, whether or not to cluster the columns
    :param cluster_rows:
    :param plot_df: The dataframe you want to plot. This can contain NAs and
    other nasty things.
    :param row_linkage_method:
    :param col_linkage_method:
    :param vmin: Minimum value to plot on heatmap
    :param vmax: Maximum value to plot on heatmap
    :param linewidth: Linewidth of lines around heatmap box elements
    (default 0)
    :param edgecolor: Color of lines around heatmap box elements (default
    white)
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

    #if cluster

    if (df.shape[0] > 1000 or df.shape[1] > 1000) or use_fastcluster:
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
                                'suggest fastcluster for such large datasets'\
            .format(df.shape), RuntimeWarning)
    else:
        linkage_function = sch.linkage

    almost_black = '#262626'
    sch.set_link_color_palette([almost_black])
    if plot_df is None:
        plot_df = df

    if (plot_df.index != df.index).any():
        raise ValueError('plot_df must have the exact same indices as df')
    if (plot_df.columns != df.columns).any():
        raise ValueError('plot_df must have the exact same columns as df')
        # make norm

    # Check if the matrix has values both above and below zero, or only above
    # or only below zero. If both above and below, then the data is
    # "divergent" and we will use a colormap with 0 centered at white,
    # negative values blue, and positive values red. Otherwise, we will use
    # the YlGnBu colormap.
    divergent = df.max().max() > 0 and df.min().min() < 0

    if color_scale == 'log':
        if vmin is None:
            vmin = max(np.floor(df.dropna(how='all').min().dropna().min()), 1e-10)
        if vmax is None:
            vmax = np.ceil(df.dropna(how='all').max().dropna().max())
        my_norm = mpl.colors.LogNorm(vmin, vmax)
    elif divergent:
        abs_max = abs(df.max().max())
        abs_min = abs(df.min().min())
        vmaxx = max(abs_max, abs_min)
        my_norm = mpl.colors.Normalize(vmin=-vmaxx, vmax=vmaxx)
    else:
        my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if cmap is None:
        cmap = mpl.cm.RdBu_r if divergent else mpl.cm.YlGnBu
        cmap.set_bad('white')

    # TODO: Add optimal leaf ordering for clusters
    # TODO: if color_scale is 'log', should distance also be on np.log(df)?
    # calculate pairwise distances for rows
    if color_scale == 'log':
        df = np.log10(df)
    row_pairwise_dists = distance.squareform(distance.pdist(df,
                                                            metric=metric))
    row_linkage = linkage_function(row_pairwise_dists, method=linkage_method)

    # calculate pairwise distances for columns
    col_pairwise_dists = distance.squareform(distance.pdist(df.T,
                                                            metric=metric))
    # cluster
    col_linkage = linkage_function(col_pairwise_dists, method=linkage_method)

    # heatmap with row names

    def get_width_ratios(shape, side_colors,
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

        :param side_colors:
        :type side_colors:
        :param colorbar_loc:
        :type colorbar_loc:
        :param dimension:
        :type dimension:
        :param side_colors_ratio:
        :type side_colors_ratio:
        :return:
        :rtype:
        """
        i = 0 if dimension == 'height' else 1
        half_dendrogram = shape[i] * 0.1/shape[i]
        if colorbar_loc not in ('upper left', 'right', 'bottom'):
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

        if (colorbar_loc == 'right' and dimension == 'width') or (
                    colorbar_loc == 'bottom' and dimension == 'height'):
            return ratios + [1, 0.05]
        else:
            return ratios + [1]


    width_ratios = get_width_ratios(df.shape,
                                    row_side_colors,
                                    colorbar_loc, dimension='width')
    height_ratios = get_width_ratios(df.shape,
                                     col_side_colors,
                                     colorbar_loc, dimension='height')
    nrows = 3 if col_side_colors is None else 4
    ncols = 3 if row_side_colors is None else 4

    width = df.shape[1] * 0.25
    height = min(df.shape[0] * .75, 40)
    if figsize is None:
        figsize = (width, height)
    #print figsize



    fig = plt.figure(figsize=figsize)
    heatmap_gridspec = \
        gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios)
    #     print heatmap_gridspec

    ### col dendrogram ###
    col_dendrogram_ax = fig.add_subplot(heatmap_gridspec[1, ncols - 1])
    if cluster_cols:
        col_dendrogram = sch.dendrogram(col_linkage,
                                        color_threshold=np.inf,
                                        color_list=[almost_black])
    else:
        col_dendrogram = {'leaves': list(range(df.shape[1]))}
    _clean_axis(col_dendrogram_ax)

    # TODO: Allow for array of color labels
    ### col colorbar ###
    if col_side_colors is not None:
        column_colorbar_ax = fig.add_subplot(heatmap_gridspec[2, ncols - 1])
        col_side_matrix, col_cmap = _color_list_to_matrix_and_cmap(
            col_side_colors,
            ind=col_dendrogram['leaves'],
            row=False)
        column_colorbar_ax_pcolormesh = column_colorbar_ax.pcolormesh(
            col_side_matrix, cmap=col_cmap,
            edgecolor=edgecolor, linewidth=linewidth)
        column_colorbar_ax.set_xlim(0, col_side_matrix.shape[1])
        _clean_axis(column_colorbar_ax)

    ### row dendrogram ###
    row_dendrogram_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, 1])
    if cluster_rows:
        row_dendrogram = \
            sch.dendrogram(row_linkage,
                           color_threshold=np.inf,
                           orientation='right',
                           color_list=[almost_black])
    else:
        row_dendrogram = {'leaves': list(range(df.shape[0]))}
    _clean_axis(row_dendrogram_ax)

    ### row colorbar ###
    if row_side_colors is not None:
        row_colorbar_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, 2])
        row_side_matrix, row_cmap = _color_list_to_matrix_and_cmap(
            row_side_colors,
            ind=row_dendrogram['leaves'],
            row=True)
        row_colorbar_ax.pcolormesh(row_side_matrix, cmap=row_cmap,
                                   edgecolors=edgecolor, linewidth=linewidth)
        row_colorbar_ax.set_ylim(0, row_side_matrix.shape[0])
        _clean_axis(row_colorbar_ax)

    ### heatmap ####
    heatmap_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, ncols - 1])
    heatmap_ax_pcolormesh = \
        heatmap_ax.pcolormesh(plot_df.ix[row_dendrogram['leaves'],
                                         col_dendrogram['leaves']].values,
                              norm=my_norm, cmap=cmap,
                              edgecolor=edgecolor,
                              lw=linewidth)

    heatmap_ax.set_ylim(0, df.shape[0])
    heatmap_ax.set_xlim(0, df.shape[1])
    _clean_axis(heatmap_ax)

    ## row labels ##
    if isinstance(label_rows, Iterable):
        if len(label_rows) == df.shape[0]:
            yticklabels = label_rows
            label_rows = True
        else:
            raise AssertionError("Length of 'label_rows' must be the same as "
                                 "df.shape[0] (len(label_rows)={}, df.shape["
                                 "0]={})".format(len(label_rows), df.shape[0]))
    elif label_rows:
        yticklabels = df.index

    if label_rows:
        yticklabels = [yticklabels[i] for i in row_dendrogram['leaves']]
        heatmap_ax.set_yticks(np.arange(df.shape[0]) + 0.5)
        heatmap_ax.yaxis.set_ticks_position('right')
        heatmap_ax.set_yticklabels(yticklabels, fontsize=ylabel_fontsize)

    # Add title if there is one:
    if title is not None:
        col_dendrogram_ax.set_title(title, fontsize=title_fontsize)

    ## col labels ##
    if isinstance(label_cols, Iterable):
        if len(label_cols) == df.shape[1]:
            xticklabels = label_cols
            label_cols = True
        else:
            raise AssertionError("Length of 'label_cols' must be the same as "
                                 "df.shape[1] (len(label_cols)={}, df.shape["
                                 "1]={})".format(len(label_cols), df.shape[1]))
    elif label_cols:
        xticklabels = df.columns

    if label_cols:
        xticklabels = [xticklabels[i] for i in col_dendrogram['leaves']]
        heatmap_ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        xticklabels = heatmap_ax.set_xticklabels(xticklabels,
                                                 fontsize=xlabel_fontsize)
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

    # note that we could pass the norm explicitly with norm=my_norm
    cb = fig.colorbar(heatmap_ax_pcolormesh,
                      cax=scale_colorbar_ax)
    cb.set_label(colorbar_label)

    # move ticks to left side of colorbar to avoid problems with tight_layout
    cb.ax.yaxis.set_ticks_position('left')
    cb.outline.set_linewidth(0)

    ## Make colorbar narrower
    #xmin, xmax, ymin, ymax = cb.ax.axis()
    #cb.ax.set_xlim(xmin, xmax/0.2)

    # make colorbar labels smaller
    yticklabels = cb.ax.yaxis.get_ticklabels()
    for t in yticklabels:
        t.set_fontsize(colorbar_ticklabels_fontsize)

    fig.tight_layout()
    return fig, row_dendrogram, col_dendrogram
