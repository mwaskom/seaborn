from collections import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings


import pdb

from . import utils

class _MatrixPlotter(object):
    """Plotter for 2D matrix data

    This will be used by the `clusteredheatplot`
    """
    def establish_variables(self, data, **kws):
        """Extract variables from data or use directly."""
        self.data = data

        # Either the data is already in 2d matrix format, or need to do a pivot
        if kws['pivot_kws'] is not None:
            self.data2d = self.data.pivot(**kws['pivot_kws'])
        else:
            self.data2d = self.data

    def plot(self, ax):
        raise NotImplementedError

class _ClusteredHeatmapPlotter(_MatrixPlotter):
    """Plotter of 2d matrix data, hierarchically clustered with dendrograms

    """

    def __init__(self, data, pivot_kws=None,
                 color_scale='linear', linkage_method='average',
                 metric='euclidean', pcolormesh_kws=None,
                 row_kws=None, col_kws=None,
                 colorbar_kws=None,
                 use_fastcluster=False):
        self.data = data
        self.pivot_kws = pivot_kws
        self.color_scale = color_scale
        self.linkage_method = linkage_method
        self.metric = metric
        # self.pcolormesh_kws = pcolormesh_kws
        # self.row_kws = row_kws
        # self.col_kws = col_kws
        # self.colorbar_kws = colorbar_kws
        self.use_fastcluster = use_fastcluster

        self.establish_variables(data, pivot_kws=pivot_kws)
        self.interpret_kws(row_kws, col_kws, pcolormesh_kws, colorbar_kws)
        self.calculate_linkage()

    def interpret_kws(self, row_kws, col_kws, pcolormesh_kws, colorbar_kws):
        """Set defaults for keyword arguments
        """
        # Interpret keyword arguments
        self.row_kws = {} if row_kws is None else row_kws
        self.col_kws = {} if col_kws is None else col_kws

        for kws in [self.row_kws, self.col_kws]:
            kws.setdefault('linkage_matrix', None)
            kws.setdefault('cluster', True)
            kws.setdefault('label_loc', 'dendrogram')
            kws.setdefault('label', True)
            kws.setdefault('fontsize', None)
            kws.setdefault('side_colors', None)

        self.colorbar_kws = {} if colorbar_kws is None else colorbar_kws
        self.colorbar_kws.setdefault('fontsize', None)
        self.colorbar_kws.setdefault('label', 'values')

        # Pcolormesh keyword arguments take more work
        self.pcolormesh_kws = {} if pcolormesh_kws is None else pcolormesh_kws
        self.pcolormesh_kws.setdefault('vmin', None)
        self.pcolormesh_kws.setdefault('vmax', None)
        self.pcolormesh_kws.setdefault('edgecolor', 'white')
        self.pcolormesh_kws.setdefault('linewidth', 0)

        self.norm = None if 'norm' not in pcolormesh_kws else pcolormesh_kws[
            'norm']
        self.cmap = None if 'cmap' not in pcolormesh_kws else pcolormesh_kws[
            'cmap']
        self.edgecolor = self.pcolormesh_kws['edgecolor']
        self.linewidth = self.pcolormesh_kws['linewidth']

        # Check if the matrix has values both above and below zero, or only above
        # or only below zero. If both above and below, then the data is
        # "divergent" and we will use a colormap with 0 centered at white,
        # negative values blue, and positive values red. Otherwise, we will use
        # the YlGnBu colormap.
        self.divergent = (self.data2d.max().max() > 0 and
                     self.data2d.min().min() < 0) and \
                    not self.color_scale == 'log'
        if self.color_scale == 'log':
            if self.pcolormesh_kws['vmin'] is None:
                self.pcolormesh_kws['vmin'] = self.data2d.replace(0, np.nan)\
                    .dropna(how='all').min().dropna().min()
            if self.pcolormesh_kws['vmax'] is None:
                self.pcolormesh_kws['vmax'] = self.data2d.dropna(how='all')\
                    .max().dropna().max()
            if self.norm is None:
                self.norm = mpl.colors.LogNorm(
                    self.pcolormesh_kws['vmin'], self.pcolormesh_kws['vmax'])
        elif self.divergent:
            abs_max = abs(self.data2d.max().max())
            abs_min = abs(self.data2d.min().min())
            vmaxx = max(abs_max, abs_min)
            self.pcolormesh_kws['vmin'] = -vmaxx
            self.pcolormesh_kws['vmax'] = vmaxx
            self.pcolormesh_kws['norm'] = mpl.colors.Normalize(vmin=-vmaxx,
                                                               vmax=vmaxx)
        else:
            self.pcolormesh_kws.setdefault('vmin', self.data2d.min().min())
            self.pcolormesh_kws.setdefault('vmax', self.data2d.max().max())

        if self.cmap is None:
            self.cmap = mpl.cm.RdBu_r if self.divergent else mpl.cm.YlGnBu
            self.cmap.set_bad('white')


    def calculate_linkage(self):
        """Calculate linkage matrices

        These are then passed to the dendrogram functions to plot pairwise
        similarity of samples
        """

        if self.color_scale == 'log':
            values = np.log10(self.data2d.values)
        else:
            values = self.data2d.values

        if self.row_kws['linkage_matrix'] is None:
            import scipy.spatial.distance as distance
            linkage_function = self.get_linkage_function(values.shape,
                                                     self.use_fastcluster)
            row_pairwise_dists = distance.squareform(
                distance.pdist(values, metric=self.metric))
            self.row_linkage = linkage_function(row_pairwise_dists,
                                           method=self.linkage_method)
        else:
            self.row_linkage = self.row_kws['linkage_matrix']

        # calculate pairwise distances for columns
        if self.col_kws['linkage_matrix'] is None:
            linkage_function = self.get_linkage_function(values.shape,
                                                     self.use_fastcluster)
            col_pairwise_dists = distance.squareform(
                distance.pdist(values.T, metric=self.metric))
            # cluster
            self.col_linkage = linkage_function(col_pairwise_dists,
                                           method=self.linkage_method)
        else:
            self.col_linkage = self.col_kws['linkage_matrix']

    def get_width_ratios(self, shape, side_colors,
                                  dimension, side_colors_ratio=0.05):

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
        if dimension not in ('height', 'width'):
            raise AssertionError("{} is not a valid 'dimension' (valid: "
                                 "'height', 'width')".format(
                dimension))

        ratios = [half_dendrogram, half_dendrogram]
        if side_colors:
            ratios += [side_colors_ratio]

        if (dimension == 'height'):
            return ratios + [1, 0.05]
        else:
            return ratios + [1]

    def color_list_to_matrix_and_cmap(self, colors, ind, row=True):
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

        colors_original = colors
        colors = set(colors)
        col_to_value = dict((col, i) for i, col in enumerate(colors))
        matrix = np.array([col_to_value[col] for col in colors_original])[
            ind]
        # Is this row-side or column side?
        if row:
            new_shape = (len(colors_original), 1)
        else:
            new_shape = (1, len(colors_original))
        matrix = matrix.reshape(new_shape)

        cmap = mpl.colors.ListedColormap(colors)
        return matrix, cmap

    def get_linkage_function(self, shape, use_fastcluster):
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
            import scipy.cluster.hierarchy as sch
            linkage_function = sch.linkage
        return linkage_function


    def plot_dendrogram(self, fig, kws, gridspec, linkage, orientation='top',
                         dendrogram_kws=None):
        """Plots a dendrogram on the figure at the gridspec location using
        the linkage matrix

        Both the computation and plotting must be in this same function because
        scipy.cluster.hierarchy.dendrogram does ax = plt.gca() and cannot be
        specified its own ax object.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure instance to plot onto
        kws : dict
            Keyword arguments for column or row plotting passed to clusterplot
        gridspec : matplotlib.gridspec.gridspec
            Indexed gridspec object for where to put the dendrogram plot
        linkage : numpy.array
            Linkage matrix, usually created by scipy.cluster.hierarchy.linkage
        orientation : str
            Specify the orientation of the dendrogram
        dendrogram_kws : dict
            Any additional keyword arguments for scipy.cluster.hierarchy.dendrogram

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes upon which the dendrogram was plotted
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is "leaves" which tells
            the ordering of the matrix
        """
        import scipy.cluster.hierarchy as sch

        if dendrogram_kws is None:
            dendrogram_kws = {}

        almost_black = '#262626'
        ax = fig.add_subplot(gridspec)
        if kws['cluster']:
            dendrogram = sch.dendrogram(linkage,
                                        color_threshold=np.inf,
                                        color_list=[almost_black],
                                        orientation=orientation,
                                        **dendrogram_kws)
        else:
            dendrogram = {'leaves': list(range(linkage.shape[0]))}

        utils.despine(ax=ax, bottom=True, left=True)
        ax.set_axis_bgcolor('white')
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])
        return ax, dendrogram

    def plot_sidecolors(self, fig, kws, gridspec, dendrogram, edgecolor,
                         linewidth):
        """Plots color labels between the dendrogram and the heatmap
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure instance to plot onto
        kws : dict
            Keyword arguments for column or row plotting passed to clusterplot
        gridspec : matplotlib.gridspec.gridspec
            Indexed gridspec object for where to put the dendrogram plot
        dendrogram : dict
            Dendrogram with key-value 'leaves' as a list of indices in the
            clustered order
        edgecolor : matplotlib color
            Color of the lines outlining each box of the heatmap
        linewidth : float
            Width of the lines outlining each box of the heatmap

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object, if plotted
        """
        if kws['side_colors'] is not None:
            ax = fig.add_subplot(gridspec)
            side_matrix, cmap = self.color_list_to_matrix_and_cmap(
                kws['side_colors'],
                ind=dendrogram['leaves'],
                row=False)
            ax.pcolormesh(side_matrix, cmap=cmap, edgecolor=edgecolor,
                          linewidth=linewidth)
            ax.set_xlim(0, side_matrix.shape[1])
            ax.set_yticks([])
            ax.set_xticks([])
            utils.despine(ax=ax, left=True, bottom=True)
            return ax

    def label_dimension(self, dimension, kws, heatmap_ax, dendrogram_ax,
                         dendrogram,
                         df):
        """Label either the rows or columns of a heatmap
        Parameters
        ----------
        dimension : str
            either "row" or "column", which dimension we are labeling
        kws : dict
            Keyword arguments for the dimension, either row_kws or col_kws from
            clusterplot()
        heatmap_ax : matplotlib.axes.Axes
            Axes object where the heatmap is plotted
        dendrogram_ax : matplotlib.axes.Axes
            Axes object where this dimensions's dendrogram is plotted
        dendrogram : dict
            Dendrogram dictionary with key 'leaves' containing the reordered
            columns or rows after clustering
        df : pandas.DataFrame
            Dataframe that we're plotting. Need access to the rownames (df.index)
            and columns for the default labeling.
        """
        if dimension not in ['row', 'col']:
            raise ValueError('Argument "dimension" must be one of "row" or '
                             '"col", not "{}"'.format(dimension))
        axis = 0 if dimension == 'row' else 1

        if kws['label_loc'] not in ['heatmap', 'dendrogram']:
            raise ValueError('Parameter {}_kws["label_loc"] must be one of '
                             '"heatmap" or "dendrogram", not "{}"'.format(
                kws[
                    "label_loc"]))
        ax = heatmap_ax if kws['label_loc'] == 'heatmap' else dendrogram_ax

        # Need to scale the ticklabels by 10 if we're labeling the dendrogram_ax
        scale = 1 if kws['label_loc'] == 'heatmap' else 10

        # Remove all ticks from the other axes
        other_ax = dendrogram_ax if kws[
                                        'label_loc'] == 'heatmap' else heatmap_ax
        other_ax_axis = other_ax.yaxis if dimension == 'row' else other_ax.xaxis
        other_ax_axis.set_ticks([])

        ax_axis = ax.yaxis if dimension == 'row' else ax.xaxis

        if isinstance(kws['label'], Iterable):
            if len(kws['label']) == df.shape[axis]:
                ticklabels = kws['label']
                kws['label'] = True
            else:
                raise AssertionError("Length of '{0}_kws['label']' must be "
                                     "the same as "
                                     "df.shape[{1}] (len({0}_kws['label'])={2}, "
                                     "df.shape[{1}]={3})".format(dimension,
                                                                 axis,
                                                                 len(kws[
                                                                     'label']),
                                                                 df.shape[
                                                                     axis]))
        elif kws['label']:
            ticklabels = df.index if dimension == 'row' else df.columns
        else:
            ax_axis.set_ticklabels([])

        if kws['label']:
            ticklabels_ordered = [ticklabels[i] for i in
                                  dendrogram['leaves']]
            # pdb.set_trace()
            # despine(ax=ax, bottom=True, left=True)
            ticks = (np.arange(df.shape[axis]) + 0.5) * scale
            ax_axis.set_ticks(ticks)
            ticklabels = ax_axis.set_ticklabels(ticklabels_ordered, )
            ax.tick_params(labelsize=kws['fontsize'])
            if dimension == 'row':
                ax_axis.set_ticks_position('right')
            else:
                for label in ticklabels:
                    label.set_rotation(90)

    def plot(self, fig, data2d, title, title_fontsize):
        edgecolor = self.pcolormesh_kws['edgecolor']
        linewidth = self.pcolormesh_kws['linewidth']

        width_ratios = self.get_width_ratios(data2d.shape,
                                         self.row_kws['side_colors'],
                                         # colorbar_kws['loc'],
                                         dimension='width')

        height_ratios = self.get_width_ratios(data2d.shape,
                                          self.col_kws['side_colors'],
                                          # colorbar_kws['loc'],
                                          dimension='height')
        nrows = 3 if self.col_kws['side_colors'] is None else 4
        ncols = 3 if self.row_kws['side_colors'] is None else 4

        heatmap_gridspec = \
            gridspec.GridSpec(nrows, ncols,  #wspace=.1, hspace=.1,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios)

        ### col dendrogram ###
        col_dendrogram_ax, self.col_dendrogram = \
            self.plot_dendrogram(fig, self.col_kws,
                                  heatmap_gridspec[1, ncols - 1],
                                  self.col_linkage)

        # TODO: Allow for array of color labels
        # TODO: allow for groupby and then auto-selecting of colors
        ### col colorbar ###
        self.plot_sidecolors(fig, self.col_kws,
                              heatmap_gridspec[2, ncols - 1],
                              self.col_dendrogram,
                              edgecolor, linewidth)

        ### row dendrogram ##
        row_dendrogram_ax, self.row_dendrogram = \
            self.plot_dendrogram(fig, self.row_kws,
                             heatmap_gridspec[nrows - 1, 1],
                             self.row_linkage, orientation='right')


        ### row colorbar ###if dimension == 'row' else ax.xaxis
        self.plot_sidecolors(fig, self.col_kws,
                         heatmap_gridspec[nrows - 1, 2], self.col_dendrogram,
                         edgecolor, linewidth)

        ### heatmap ####
        heatmap_ax = fig.add_subplot(heatmap_gridspec[nrows - 1, ncols - 1])
        heatmap_ax_pcolormesh = \
            heatmap_ax.pcolormesh(data2d.ix[
                                      self.row_dendrogram['leaves'],
                                      self.col_dendrogram['leaves']].values,
                                  cmap=self.cmap,
                                  norm=self.norm,
                                  **self.pcolormesh_kws)
        utils.despine(ax=heatmap_ax, left=True, bottom=True)
        heatmap_ax.set_ylim(0, data2d.shape[0])
        heatmap_ax.set_xlim(0, data2d.shape[1])

        ## row labels ##
        self.label_dimension('row', self.row_kws, heatmap_ax,
                         row_dendrogram_ax, self.row_dendrogram, data2d)

        # Add title if there is one:
        if title is not None:
            col_dendrogram_ax.set_title(title, fontsize=title_fontsize)

        ## col labels ##
        self.label_dimension('col', self.col_kws, heatmap_ax,
                         col_dendrogram_ax, self.col_dendrogram, data2d)

        ### scale colorbar ###
        scale_colorbar_ax = fig.add_subplot(
            heatmap_gridspec[0:(nrows - 1), 0])  # colorbar for scale in upper
        # left corner
        colorbar_ticklabel_fontsize = self.colorbar_kws.pop('fontsize')
        cb = fig.colorbar(heatmap_ax_pcolormesh,
                          cax=scale_colorbar_ax, **self.colorbar_kws)

        ## Setting the number of colorbar ticks to at most 3 (nbins+1) currently
        ## only works for divergent and linear colormaps.
        if self.color_scale == 'log':
            #TODO: get at most 3 ticklabels showing on the colorbar for log-scaled
            pass
            # This next stuff does not work...
            # tick_locator = mpl.ticker.LogLocator(numticks=3)
            # pdb.set_trace()
            # tick_locator.tick_values(pcolormesh_kws['vmin'],
            #                          pcolormesh_kws['vmax'])
        elif self.divergent:
            # TODO: get at most 3 ticklabels working for non-divergent data
            tick_locator = mpl.ticker.MaxNLocator(nbins=2,
                                                  symmetric=self.divergent,
                                                  prune=None, trim=False)
            cb.ax.set_yticklabels(
                tick_locator.bin_boundaries(self.pcolormesh_kws['vmin'],
                                            self.pcolormesh_kws['vmax']))
            cb.ax.yaxis.set_major_locator(tick_locator)

        # move ticks to left side of colorbar to avoid problems with tight_layout
        cb.ax.yaxis.set_ticks_position('left')
        if colorbar_ticklabel_fontsize is not None:
            cb.ax.tick_params(labelsize=colorbar_ticklabel_fontsize)
        cb.outline.set_linewidth(0)

        heatmap_gridspec.tight_layout(fig)


def clusteredheatmap(data, pivot_kws=None, title=None, title_fontsize=12,
                     color_scale='linear', linkage_method='average',
                     metric='euclidean', figsize=None, pcolormesh_kws=None,
                     row_kws=None, col_kws=None, colorbar_kws=None,
                     data_na_ok=None, use_fastcluster=False, fig=None):
    """Plot a hierarchically clustered heatmap of a pandas DataFrame

    This is liberally borrowed (with permission) from http://bit.ly/1eWcYWc
    Many thanks to Christopher DeBoever and Mike Lovci for providing
    heatmap/gridspec/colorbar positioning guidance.

    Parameters
    ----------
    data: DataFrame
        Data for clustering. Should be a dataframe with no NAs. If you
        still want to plot a dataframe with NAs, provide a non-NA dataframe
        to data (with NAs replaced by 0 or something by your choice) and your
        NA-full dataframe to data_na_ok
    pivot_kws : dict
        If the data is in "tidy" format, reshape the data with these pivot
        keyword arguments
    title: string, optional
        Title of the figure. Default None
    title_fontsize: int, optional
        Size of the plot title. Default 12
    color_scale: string, 'log' or 'linear'
        How to scale the colors plotted in the heatmap. Default "linear"
    linkage_method: string
        Which linkage method to use for calculating clusters.
        See scipy.cluster.hierarchy.linkage documentation for more information:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        Default "average"
    metric: string
        Distance metric to use for the data. Default is "euclidean." See
        scipy.spatial.distance.pdist documentation for more options
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    figsize: tuple of two ints
        Size of the figure to create. Default is a function of the dataframe
        size.
    pcolormesh_kws : dict
        Keyword arguments to pass to the heatmap pcolormesh plotter. E.g.
        vmin, vmax, cmap, norm. If these are none, they are auto-detected
        from your data. If the data is divergent, i.e. has values both
        above and below zero, then the colormap is blue-red with blue
        as negative values and red as positive. If the data is not
        divergent, then the colormap is YlGnBu.
        Default:
        dict(vmin=None, vmax=None, edgecolor='white', linewidth=0, cmap=None,
        norm=None)
    {row,col}_kws : dict
        Keyword arguments for rows and columns. Specify the tick label
        location as either on the dendrogram or heatmap via
        label_loc='heatmap'. Can turn of labeling altogether with
        label=False. Can specify you own linkage matrix via linkage_matrix.
        Can also specify side colors labels via side_colors=colors, which are
        useful for evaluating whether samples within a group are clustered
        together.
        Default:
        dict(linkage_matrix=None, cluster=True, label_loc='dendrogram',
        label=True, fontsize=None, side_colors=None)
    colorbar_kws : dict
        Keyword arguments for the colorbar. The ticklabel fontsize is
        extracted from this dict, then removed.
        dict(fontsize=None, label='values')
    data_na_ok: Dataframe
        The dataframe to plot. May have NAs, unlike the "data" argument.
    use_fastcluster: bool
        Whether or not to use the "fastcluster" module in Python,
        which calculates linkage several times faster than scipy.cluster.hierachy
        Default False except for datasets with more than 1000 rows or columns.

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
    plotter = _ClusteredHeatmapPlotter(data, pivot_kws=pivot_kws,
                                       color_scale=color_scale,
                                       linkage_method=linkage_method,
                                       metric=metric,
                                       pcolormesh_kws=pcolormesh_kws,
                                       row_kws=row_kws,
                                       col_kws=col_kws,
                                       colorbar_kws=colorbar_kws,
                                       use_fastcluster=use_fastcluster)
    # TODO: do plt.gcf() if there is a current figure else make a new one
    # with the correct dimensions
    if fig is None:
        if figsize is None:
            width = data.shape[1] * 0.25
            height = min(data.shape[0] * .75, 40)
            figsize = (width, height)
        fig = plt.figure(figsize=figsize)


    if data_na_ok is None:
        data_na_ok = plotter.data2d

    if (data_na_ok.index != plotter.data2d.index).any():
        raise ValueError(
            'data_na_ok must have the exact same indices as the 2d data')
    if (data_na_ok.columns != plotter.data2d.columns).any():
        raise ValueError(
            'data_na_ok must have the exact same columns as the 2d data')

    plotter.plot(fig, data_na_ok, title, title_fontsize)

    return fig, plotter.row_dendrogram, plotter.col_dendrogram

