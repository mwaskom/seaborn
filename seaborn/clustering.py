from collections import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings

from . import utils


class _MatrixPlotter(object):
    """Plotter for 2D matrix data

    This will be used by the `clusteredheatmap`
    """

    def establish_variables(self, data, **kws):
        """Extract variables from data or use directly."""
        self.data = data

        # Either the data is already in 2d matrix format, or need to do a pivot
        if 'pivot_kws' in kws and kws['pivot_kws'] is not None:
            self.data2d = self.data.pivot(**kws['pivot_kws'])
        else:
            self.data2d = self.data

    def plot(self, *args, **kwargs):
        raise NotImplementedError


class _ClusteredHeatmapPlotter(_MatrixPlotter):
    """Plotter of 2d matrix data, hierarchically clustered with dendrograms

    """

    def __init__(self, data, pivot_kws=None,
                 color_scale='linear', linkage_method='average',
                 metric='euclidean', pcolormesh_kws=None,
                 dendrogram_kws=None,
                 row_kws=None, col_kws=None,
                 colorbar_kws=None,
                 use_fastcluster=False,
                 data_na_ok=None):
        self.data = data
        self.pivot_kws = pivot_kws
        self.color_scale = color_scale
        self.linkage_method = linkage_method
        self.metric = metric
        self.use_fastcluster = use_fastcluster

        self.establish_variables(data, pivot_kws=pivot_kws)
        self.validate_data_na_ok(data_na_ok)
        self.interpret_kws(row_kws, col_kws, pcolormesh_kws,
                           dendrogram_kws, colorbar_kws)
        self.calculate_linkage()
        self.row_dendrogram = self.calculate_dendrogram(self.row_kws,
                                                        self.row_linkage)
        self.col_dendrogram = self.calculate_dendrogram(self.col_kws,
                                                        self.col_linkage)

    def establish_axes(self, fig=None, figsize=None):
        # TODO: do plt.gcf() if there is a current figure else make a new one
        # with the correct dimensions
        if fig is None:
            if figsize is None:
                width = self.data2d.shape[1] * 0.5
                height = min(self.data2d.shape[0] * .5, 40)
                figsize = (width, height)
            fig = plt.figure(figsize=figsize)

        self.fig = fig
        width_ratios = self.get_fig_width_ratios(self.row_kws['side_colors'],
                                                 # colorbar_kws['loc'],
                                                 dimension='width')

        height_ratios = self.get_fig_width_ratios(self.col_kws['side_colors'],
                                                  dimension='height')
        nrows = 3 if self.col_kws['side_colors'] is None else 4
        ncols = 3 if self.row_kws['side_colors'] is None else 4

        self.gs = gridspec.GridSpec(nrows, ncols,
                                    width_ratios=width_ratios,
                                    height_ratios=height_ratios)

        self.row_dendrogram_ax = self.fig.add_subplot(self.gs[nrows - 1, 1])
        self.col_dendrogram_ax = self.fig.add_subplot(self.gs[1, ncols - 1])

        self.row_side_colors_ax = None
        self.col_side_colors_ax = None

        if self.col_kws['side_colors'] is not None:
            self.col_side_colors_ax = self.fig.add_subplot(
                self.gs[2, ncols - 1])
        if self.row_kws['side_colors'] is not None:
            self.row_side_colors_ax = self.fig.add_subplot(
                self.gs[nrows - 1, 2])

        self.heatmap_ax = self.fig.add_subplot(self.gs[nrows - 1, ncols - 1])

        # colorbar for scale in upper left corner
        self.colorbar_ax = self.fig.add_subplot(self.gs[0:(nrows - 1), 0])

    def interpret_kws(self, row_kws, col_kws, pcolormesh_kws,
                      dendrogram_kws, colorbar_kws):
        """Set defaults for keyword arguments
        """
        # Interpret keyword arguments
        self.row_kws = {} if row_kws is None else row_kws
        self.col_kws = {} if col_kws is None else col_kws

        if 'side_colors' in self.row_kws and self.row_kws['side_colors'] \
                is not None:
            assert len(self.row_kws['side_colors']) == self.data2d.shape[0]
        if 'side_colors' in self.col_kws and self.col_kws['side_colors'] \
                is not None:
            assert len(self.col_kws['side_colors']) == self.data2d.shape[1]

        for kws in (self.row_kws, self.col_kws):
            kws.setdefault('linkage_matrix', None)
            kws.setdefault('cluster', True)
            kws.setdefault('label_loc', 'dendrogram')
            kws.setdefault('label', True)
            kws.setdefault('fontsize', None)
            kws.setdefault('side_colors', None)

        self.colorbar_kws = {} if colorbar_kws is None else colorbar_kws
        self.colorbar_kws.setdefault('fontsize', 14)
        self.colorbar_kws.setdefault('label', 'values')

        self.dendrogram_kws = {} if dendrogram_kws is None else dendrogram_kws
        self.dendrogram_kws.setdefault('color_threshold', np.inf)
        self.dendrogram_kws.setdefault('color_list', ['k'])
        # even if the user specified no_plot as False, override because we
        # have to control the plotting
        if 'no_plot' in self.dendrogram_kws and \
                not self.dendrogram_kws['no_plot']:
            warnings.warn('Cannot specify "no_plot" as False in '
                          'dendrogram_kws')
        self.dendrogram_kws['no_plot'] = True

        # Pcolormesh keyword arguments take more work
        self.pcolormesh_kws = {} if pcolormesh_kws is None else pcolormesh_kws
        self.pcolormesh_kws.setdefault('edgecolor', 'white')
        self.pcolormesh_kws.setdefault('linewidth', 0)

        self.vmin = None if 'vmin' not in self.pcolormesh_kws else \
            self.pcolormesh_kws['vmin']
        self.vmax = None if 'vmax' not in self.pcolormesh_kws else \
            self.pcolormesh_kws['vmax']
        self.norm = None if 'norm' not in self.pcolormesh_kws else \
            self.pcolormesh_kws['norm']
        self.cmap = None if 'cmap' not in self.pcolormesh_kws else \
            self.pcolormesh_kws['cmap']
        self.edgecolor = self.pcolormesh_kws['edgecolor']
        self.linewidth = self.pcolormesh_kws['linewidth']

        # Check if the matrix has values both above and below zero, or only
        # above or only below zero. If both above and below, then the data is
        # "divergent" and we will use a colormap with 0 centered at white,
        # negative values blue, and positive values red. Otherwise, we will use
        # the YlGnBu colormap.
        vmax = self.data2d.max().max()
        vmin = self.data2d.min().min()
        log = self.color_scale == 'log'
        self.divergent = (vmax > 0 and vmin < 0) and not log

        if self.color_scale == 'log':
            if self.vmin is None:
                self.vmin = self.data2d.replace(0, np.nan).dropna(
                    how='all').min().dropna().min()
            if self.vmax is None:
                self.vmax = self.data2d.dropna(how='all').max().dropna().max()
            if self.norm is None:
                self.norm = mpl.colors.LogNorm(self.vmin, self.vmax)
        elif self.divergent:
            abs_max = abs(self.data2d.max().max())
            abs_min = abs(self.data2d.min().min())
            vmaxx = max(abs_max, abs_min)
            self.vmin = -vmaxx
            self.vmax = vmaxx
            self.norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        else:
            self.pcolormesh_kws.setdefault('vmin', self.data2d.min().min())
            self.pcolormesh_kws.setdefault('vmax', self.data2d.max().max())

        if self.cmap is None:
            self.cmap = mpl.cm.RdBu_r if self.divergent else mpl.cm.YlGnBu
            self.cmap.set_bad('white')
        # Make sure there's no trailing `cmap` or `vmin` or `vmax` values
        if 'cmap' in self.pcolormesh_kws:
            self.pcolormesh_kws.pop('cmap')
        if 'vmin' in self.pcolormesh_kws:
            self.pcolormesh_kws.pop('vmin')
        if 'vmax' in self.pcolormesh_kws:
            self.pcolormesh_kws.pop('vmax')

    def validate_data_na_ok(self, data_na_ok):
        if data_na_ok is None:
            self.data2d_na_ok = self.data2d
        else:
            self.data2d_na_ok = self.data_na_ok

        if (self.data2d_na_ok.index != self.data2d.index).any():
            raise ValueError(
                'data_na_ok must have the exact same indices as the 2d data')
        if (self.data2d_na_ok.columns != self.data2d.columns).any():
            raise ValueError(
                'data_na_ok must have the exact same columns as the 2d data')

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

    def get_fig_width_ratios(self, side_colors,
                             dimension, side_colors_ratio=0.05):

        """
        Figures out the ratio of each subfigure within the larger figure.
        The dendrograms currently are 2*half_dendrogram, which is a proportion
        of the dataframe shape. Right now, this only supports the colormap in
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
        half_dendrogram = self.data2d.shape[i] * 0.1 / self.data2d.shape[i]
        if dimension not in ('height', 'width'):
            raise AssertionError("{} is not a valid 'dimension' (valid: "
                                 "'height', 'width')".format(dimension))

        ratios = [half_dendrogram, half_dendrogram]
        if side_colors:
            # Add room for the colors
            ratios += [side_colors_ratio]

        # Add the ratio for the heatmap itself
        return ratios + [1]

    @staticmethod
    def color_list_to_matrix_and_cmap(colors, ind, row=True):
        """Turns a list of colors into a numpy matrix and matplotlib colormap
        For 'heatmap()'
        This only works for 1-column color lists..

        These arguments can now be plotted using matplotlib.pcolormesh(matrix,
        cmap) and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        row : bool
            Is this to label the rows or columns? Default True.

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each corresponds to a color
            from the originally provided list of colors
        cmap : matplotlib.colors.ListedColormap

        """
        # TODO: Support multiple color labels on an element in the heatmap
        import matplotlib as mpl

        colors_original = colors
        colors = set(colors)
        col_to_value = dict((col, i) for i, col in enumerate(colors))
        matrix = np.array([col_to_value[col] for col in colors_original])[ind]

        # Is this row-side or column side?
        if row:
            # shape of matrix: nrows x 1
            new_shape = (len(colors_original), 1)
        else:
            # shape of matrix: 1 x ncols
            new_shape = (1, len(colors_original))
        matrix = matrix.reshape(new_shape)

        cmap = mpl.colors.ListedColormap(colors)
        return matrix, cmap

    @staticmethod
    def get_linkage_function(shape, use_fastcluster):
        """
        Parameters
        ----------
        shape : tuple
            (nrow, ncol) tuple of the shape of the dataframe
        use_fastcluster : bool
            Whether to use fastcluster (3rd party module) for clustering,
            which is faster than the default scipy.cluster.hierarchy.linkage
            module

        Returns
        -------
        linkage_function : function
            Linkage function to use for clustering

        .. warning:: If the product of the number of rows and cols exceeds
        10000, this wil try to import fastcluster, and raise a warning if it
        does not exist. Vanilla scipy.cluster.hierarchy.linkage will take a
        long time on these matrices.
        """
        if np.product(shape) >= 10000 or use_fastcluster:
            try:
                import fastcluster

                linkage_function = fastcluster.linkage
            except ImportError:
                raise warnings.warn(
                    'Module "fastcluster" not found. The dataframe provided '
                    'has shape {}, and one of the dimensions has greater than '
                    '1000 variables. Calculating linkage on such a matrix will'
                    ' take a long time with vanilla '
                    '"scipy.cluster.hierarchy.linkage", and we suggest '
                    'fastcluster for such large datasets'.format(shape),
                    RuntimeWarning)
        else:
            import scipy.cluster.hierarchy as sch

            linkage_function = sch.linkage
        return linkage_function

    def calculate_dendrogram(self, kws, linkage):
        """Calculates a dendrogram based on the linkage matrix

        Parameters
        ----------
        kws : dict
            Keyword arguments for column or row plotting passed to clusterplot
        linkage : numpy.array
            Linkage matrix, usually created by scipy.cluster.hierarchy.linkage
        orientation : str
            (docstring stolen from scipy.cluster.hierarchy.linkage)
            The direction to plot the dendrogram, which can be any
            of the following strings:

            'top' plots the root at the top, and plot descendent
              links going downwards. (default).

            'bottom'- plots the root at the bottom, and plot descendent
              links going upwards.

            'left'- plots the root at the left, and plot descendent
              links going right.

            'right'- plots the root at the right, and plot descendent
              links going left.

        Returns
        -------
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is "leaves" which
            tells the ordering of the matrix
        """
        import scipy.cluster.hierarchy as sch

        sch.set_link_color_palette(['k'])

        if kws['cluster']:
            dendrogram = sch.dendrogram(linkage, **self.dendrogram_kws)
        else:
            dendrogram = {'leaves': list(range(linkage.shape[0]))}
        return dendrogram

    def plot_dendrogram(self, ax, dendrogram, row=True):
        """Plots a dendrogram on the figure at the gridspec location using
        the linkage matrix

        Both the computation and plotting must be in this same function because
        scipy.cluster.hierarchy.dendrogram does ax = plt.gca() and cannot be
        specified its own ax object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted
        """
        if row:
            X = dendrogram['dcoord']
            Y = dendrogram['icoord']
        else:
            X = dendrogram['icoord']
            Y = dendrogram['dcoord']

        for x, y in zip(X, Y):
            ax.plot(x, y, color='k', linewidth=0.5)

        if row:
            ax.invert_xaxis()

        utils.despine(ax=ax, bottom=True, left=True)
        ax.set_axis_bgcolor('white')
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])

    def plot_side_colors(self, ax, kws, dendrogram, row=True):
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
        # TODO: Allow for array of color labels
        # TODO: allow for groupby and then auto-selecting of colors
        if ax is not None and kws['side_colors'] is not None:
            side_matrix, cmap = self.color_list_to_matrix_and_cmap(
                kws['side_colors'],
                ind=dendrogram['leaves'],
                row=row)
            ax.pcolormesh(side_matrix, cmap=cmap, edgecolor=self.edgecolor,
                          linewidth=self.linewidth)
            ax.set_xlim(0, side_matrix.shape[1])
            ax.set_yticks([])
            ax.set_xticks([])
            utils.despine(ax=ax, left=True, bottom=True)

    def label_dimension(self, dimension, kws, heatmap_ax, dendrogram_ax,
                        dendrogram):
        """Label either the rows or columns of a heatmap
        Parameters
        ----------
        dimension : str
            either "row" or "col", which dimension we are labeling
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
        data2d : pandas.DataFrame
            Dataframe that we're plotting. Need access to the rownames
            (data2d.index) and columns for the default labeling.
        """
        if dimension not in ['row', 'col']:
            raise ValueError('Argument "dimension" must be one of "row" or '
                             '"col", not "{}"'.format(dimension))
        axis = 0 if dimension == 'row' else 1

        if kws['label_loc'] not in ['heatmap', 'dendrogram']:
            raise ValueError(
                'Parameter {}_kws["label_loc"] must be one of '
                '"heatmap" or "dendrogram", not "{}"'.format(kws["label_loc"]))

        ax = heatmap_ax if kws['label_loc'] == 'heatmap' else dendrogram_ax

        # Need to scale the ticklabels by 10 if labeling the dendrogram_ax
        scale = 1 if kws['label_loc'] == 'heatmap' else 10

        # Remove all ticks from the other axes
        other_ax = dendrogram_ax \
            if kws['label_loc'] == 'heatmap' else heatmap_ax
        other_ax_axis = other_ax.yaxis \
            if dimension == 'row' else other_ax.xaxis
        other_ax_axis.set_ticks([])

        ax_axis = ax.yaxis if dimension == 'row' else ax.xaxis

        if isinstance(kws['label'], Iterable):
            if len(kws['label']) == self.data2d.shape[axis]:
                ticklabels = kws['label']
                kws['label'] = True
            else:
                raise AssertionError(
                    "Length of '{0}_kws['label']' must be the same as "
                    "data2d.shape[{1}] (len({0}_kws['label'])={2}, "
                    "data2d.shape[{1}]={3})".format(dimension, axis,
                                                    len(kws['label']),
                                                    self.data2d.shape[axis]))
        elif kws['label']:
            ticklabels = self.data2d.index if dimension == 'row' else self \
                .data2d.columns
        else:
            ax_axis.set_ticklabels([])

        if kws['label']:

            # Need to set the position first, then the labels because of some
            # odd matplotlib bug
            if dimension == 'row':
                # pass
                ax_axis.set_ticks_position('right')

            ticklabels_ordered = [ticklabels[i] for i in
                                  dendrogram['leaves']]
            # pdb.set_trace()
            # despine(ax=ax, bottom=True, left=True)
            ticks = (np.arange(self.data2d.shape[axis]) + 0.5) * scale
            ax_axis.set_ticks(ticks)
            ax_axis.set_ticklabels(ticklabels_ordered)
            ax.tick_params(labelsize=kws['fontsize'])
            if dimension == 'col':
                for label in ax_axis.get_ticklabels():
                    label.set_rotation(90)

    def plot_heatmap(self):
        """Plot the heatmap of the data.

        Specifically plots data_na_ok so that user can specify different
        dataframes for the linkage calculation and the plotting.
        """
        ax = self.heatmap_ax
        rows_ordered = self.row_dendrogram['leaves']
        cols_ordered = self.col_dendrogram['leaves']
        data_ordered = self.data2d_na_ok.ix[rows_ordered, cols_ordered].values

        self.heatmap_ax_pcolormesh = ax.pcolormesh(data_ordered,
                                                   cmap=self.cmap,
                                                   norm=self.norm,
                                                   vmin=self.vmin,
                                                   vmax=self.vmax,
                                                   **self.pcolormesh_kws)
        utils.despine(ax=ax, left=True, bottom=True)
        ax.set_ylim(0, self.data2d.shape[0])
        ax.set_xlim(0, self.data2d.shape[1])

    def set_title(self, title, title_fontsize=12):
        """Add title if there is one
        """
        if title is not None:
            self.col_dendrogram_ax.set_title(title, fontsize=title_fontsize)

    def colorbar(self):
        """Create the colorbar describing the hue-to-value in the heatmap
        """
        ax = self.colorbar_ax
        colorbar_ticklabel_fontsize = self.colorbar_kws.pop('fontsize')
        cb = self.fig.colorbar(self.heatmap_ax_pcolormesh,
                               cax=ax, **self.colorbar_kws)

        # Setting the number of colorbar ticks to at most 3 (nbins+1) currently
        # only works for divergent and linear colormaps.
        if self.color_scale == 'log':
            # TODO: get at most 3 ticklabels showing on the colorbar for
            # log-scaled
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
                tick_locator.bin_boundaries(self.vmin,
                                            self.vmax))
            cb.ax.yaxis.set_major_locator(tick_locator)

        # move ticks to left side of colorbar to avoid problems with
        # tight_layout
        cb.ax.yaxis.set_ticks_position('left')
        if colorbar_ticklabel_fontsize is not None:
            cb.ax.tick_params(labelsize=colorbar_ticklabel_fontsize)
        cb.outline.set_linewidth(0)

    def plot_col_side(self):
        """Plot the dendrogram and potentially sidecolors for the column
        dimension
        """
        self.plot_dendrogram(self.col_dendrogram_ax, self.col_dendrogram,
                             row=False)
        self.plot_side_colors(self.col_side_colors_ax, self.col_kws,
                              self.col_dendrogram, row=False)

    def plot_row_side(self):
        """Plot the dendrogram and potentially sidecolors for the row dimension
        """
        self.plot_dendrogram(self.row_dendrogram_ax, self.row_dendrogram,
                             row=True)
        self.plot_side_colors(self.row_side_colors_ax, self.row_kws,
                              self.row_dendrogram, row=True)

    def label(self):
        """Label the rows and columns either at the dendrogram or heatmap
        """
        self.label_dimension('row', self.row_kws, self.heatmap_ax,
                             self.row_dendrogram_ax, self.row_dendrogram)

        self.label_dimension('col', self.col_kws, self.heatmap_ax,
                             self.col_dendrogram_ax, self.col_dendrogram)

    def plot(self, fig=None, figsize=None, title=None, title_fontsize=12):
        """Plot the heatmap!

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure instance
            if None, create a new figure, or plot this onto the provided figure
        figsize : None or tuple of ints
            if None, auto-pick the figure size based on the size of the
            dataframe
        title : str
            Title of the plot. Default is no title
        title_fontsize : float
            Fontsize of the title. Default is 12pt

        Returns
        -------


        Raises
        ------

        """
        self.establish_axes(fig, figsize)
        self.plot_row_side()
        self.plot_col_side()
        self.plot_heatmap()
        self.set_title(title, title_fontsize)
        self.label()
        self.colorbar()

        # gs = gridspec
        self.gs.tight_layout(self.fig)


def clusteredheatmap(data, pivot_kws=None, title=None, title_fontsize=12,
                     color_scale='linear', linkage_method='average',
                     metric='euclidean', figsize=None, pcolormesh_kws=None,
                     dendrogram_kws=None,
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
        If the "data" argument has NAs, can supply a separate dataframe to plot
    use_fastcluster: bool
        Whether or not to use the "fastcluster" module in Python,
        which calculates linkage several times faster than
        scipy.cluster.hierachy
        Default False except for datasets with more than 1000 rows or columns.

    Returns
    -------
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
                                       dendrogram_kws=dendrogram_kws,
                                       row_kws=row_kws,
                                       col_kws=col_kws,
                                       colorbar_kws=colorbar_kws,
                                       use_fastcluster=use_fastcluster,
                                       data_na_ok=data_na_ok)

    plotter.plot(fig, figsize, title, title_fontsize)

    return plotter.row_dendrogram, plotter.col_dendrogram
