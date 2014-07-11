import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
import scipy
from numpy.testing.decorators import skipif

from .. import clustering as cl
from ..palettes import color_palette

try:
    import fastcluster
    _no_fastcluster = False
except ImportError:
    _no_fastcluster = True

class TestMatrixPlotter(object):
    shape = (10, 20)
    np.random.seed(2013)
    index = pd.Index(list('abcdefghij'), name='rownames')
    columns = pd.Index(list('ABCDEFGHIJKLMNOPQRST'), name='colnames')
    data2d = pd.DataFrame(np.random.randn(*shape), index=index,
                          columns=columns)
    data2d.ix[0:5, 10:20] += 3
    data2d.ix[5:10, 0:5] -= 3

    df = pd.melt(data2d.reset_index(), id_vars='rownames')

    def test_establish_variables_from_frame(self):
        p = cl._MatrixPlotter()
        p.establish_variables(self.df, pivot_kws=dict(index='rownames',
                                                 columns='colnames',
                                                 values='value'))
        pdt.assert_frame_equal(p.data2d, self.data2d)
        pdt.assert_frame_equal(p.data, self.df)

    def test_establish_variables_from_2d(self):
        p = cl._MatrixPlotter()
        p.establish_variables(self.data2d)
        pdt.assert_frame_equal(p.data2d, self.data2d)
        pdt.assert_frame_equal(p.data, self.data2d)


class TestClusteredHeatmapPlotter(object):
    shape = (10, 20)
    np.random.seed(2013)
    index = pd.Index(list('abcdefghij'), name='rownames')
    columns = pd.Index(list('ABCDEFGHIJKLMNOPQRST'), name='colnames')
    data2d = pd.DataFrame(np.random.randn(*shape), index=index,
                          columns=columns)
    data2d.ix[0:5, 10:20] += 3
    data2d.ix[5:10, 0:5] -= 3

    df = pd.melt(data2d.reset_index(), id_vars='rownames')

    default_dim_kws = {'linkage_matrix': None, 'side_colors': None,
                       'label_loc': 'dendrogram', 'label': True,
                       'cluster': True, 'fontsize': None}

    default_pcolormesh_kws = {'linewidth': 0, 'edgecolor': 'white'}
    default_colorbar_kws = {'fontsize': None, 'label': 'values'}

    default_dendrogram_kws = {'color_threshold': np.inf,
                              'color_list': ['k'],
                              'no_plot': True}

    # Side colors for plotting
    colors = color_palette(name='Set2', n_colors=3)
    col_color_inds = np.arange(len(colors))
    col_color_inds = [random.choice(col_color_inds) for _ in data2d.shape[1]]
    col_colors = [colors[i] for i in col_color_inds]
    
    row_color_inds = np.arange(len(colors))
    row_color_inds = [random.choice(row_color_inds) for _ in data2d.shape[1]]
    row_colors = [colors[i] for i in row_color_inds]

    def test_interpret_kws_from_none_divergent(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        dendrogram_kws=None, colorbar_kws=None)
        pdt.assert_dict_equal(p.row_kws, self.default_dim_kws)
        pdt.assert_dict_equal(p.col_kws, self.default_dim_kws)

        nt.assert_equal(p.cmap, mpl.cm.RdBu_r)
        pdt.assert_dict_equal(p.pcolormesh_kws, self.default_pcolormesh_kws)
        pdt.assert_dict_equal(p.colorbar_kws, self.default_colorbar_kws)

        abs_max = np.abs(self.data2d.max().max())
        abs_min = np.abs(self.data2d.min().min())
        vmaxx = max(abs_max, abs_min)
        nt.assert_almost_equal(p.vmin, -vmaxx)
        nt.assert_almost_equal(p.vmax, vmaxx)

    def test_interpret_kws_from_none_positive(self):
        p = cl._ClusteredHeatmapPlotter(np.abs(self.data2d))
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        dendrogram_kws=None, colorbar_kws=None)
        nt.assert_equal(p.cmap, mpl.cm.YlGnBu)
        nt.assert_is_none(p.norm)

    def test_interpret_kws_from_none_log(self):
        p = cl._ClusteredHeatmapPlotter(np.log(self.data2d), color_scale='log')
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        dendrogram_kws=None, colorbar_kws=None)
        nt.assert_is_instance(p.norm, mpl.colors.LogNorm)
        nt.assert_equal(p.cmap, mpl.cm.YlGnBu)

    def test_interpret_kws_pcolormesh_cmap(self):
        cmap = mpl.cm.PRGn
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.interpret_kws(row_kws=None, col_kws=None,
                        pcolormesh_kws={'cmap': cmap},
                        dendrogram_kws=None, colorbar_kws=None)
        nt.assert_equal(p.cmap, cmap)

    def test_calculate_linkage_linear(self):
        import scipy.spatial.distance as distance
        import scipy.cluster.hierarchy as sch
        row_pairwise_dists = distance.squareform(
            distance.pdist(self.data2d.values, metric='euclidean'))
        row_linkage = sch.linkage(row_pairwise_dists, method='average')

        col_pairwise_dists = distance.squareform(
            distance.pdist(self.data2d.values.T, metric='euclidean'))
        col_linkage = sch.linkage(col_pairwise_dists, method='average')

        p = cl._ClusteredHeatmapPlotter(self.data2d, use_fastcluster=False)
        p.calculate_linkage()
        npt.assert_array_almost_equal(p.row_linkage, row_linkage)
        npt.assert_array_almost_equal(p.col_linkage, col_linkage)

    def test_calculate_linkage_log(self):
        import scipy.spatial.distance as distance
        import scipy.cluster.hierarchy as sch

        values = np.log10(self.data2d.values)
        row_pairwise_dists = distance.squareform(
            distance.pdist(values, metric='euclidean'))
        row_linkage = sch.linkage(row_pairwise_dists, method='average')

        col_pairwise_dists = distance.squareform(
            distance.pdist(values.T, metric='euclidean'))
        col_linkage = sch.linkage(col_pairwise_dists, method='average')

        p = cl._ClusteredHeatmapPlotter(self.data2d, color_scale='log')
        p.calculate_linkage()
        npt.assert_array_equal(p.row_linkage, row_linkage)
        npt.assert_array_equal(p.col_linkage, col_linkage)

    def test_get_fig_width_ratios_side_colors_none(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        width_ratios = p.get_fig_width_ratios(side_colors=None,
                                              dimension='width')
        height_ratios = p.get_fig_width_ratios(side_colors=None,
                                               dimension='height')
        npt.assert_array_equal(width_ratios, height_ratios)

    def test_get_fig_width_ratios_side_colors(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        row_kws={'side_colors':
                                                     self.row_colors},
                                        col_kws={'side_colors':
                                                     self.col_colors})

        width_ratios = p.get_fig_width_ratios(side_colors=p.col_kws[
            'side_colors'], dimension='width')
        height_ratios = p.get_fig_width_ratios(side_colors=p.row_kws[
            'side_colors'], dimension='height')
        npt.assert_array_equal(width_ratios, height_ratios)

    def test_color_list_to_matrix_and_cmap_row(self):
        import matplotlib as mpl
        colors = color_palette(name='Set2', n_colors=3)
        np.random.seed(10)
        n = 10
        ind = np.arange(n)
        color_inds = np.random.choice(np.arange(len(colors)), size=n).tolist()
        color_list = [colors[i] for i in color_inds]
        ind = np.random.shuffle(ind)

        colors_original = color_list
        colors = set(colors_original)
        col_to_value = dict((col, i) for i, col in enumerate(colors))
        matrix = np.array([col_to_value[col] for col in colors_original])[ind]
        new_shape = (len(colors_original), 1)
        matrix = matrix.reshape(new_shape)
        cmap = mpl.colors.ListedColormap(colors)

        chp = cl._ClusteredHeatmapPlotter
        matrix2, cmap2 = chp.color_list_to_matrix_and_cmap(color_list, ind,
                                                           row=True)
        npt.assert_array_equal(matrix, matrix2)
        npt.assert_array_equal(cmap.colors, cmap2.colors)

    def test_color_list_to_matrix_and_cmap_col(self):
        import matplotlib as mpl
        colors = color_palette(name='Set2', n_colors=3)
        np.random.seed(10)
        n = 10
        ind = np.arange(n)
        color_inds = np.random.choice(np.arange(len(colors)), size=n).tolist()
        color_list = [colors[i] for i in color_inds]
        ind = np.random.shuffle(ind)

        colors_original = color_list
        colors = set(colors_original)
        col_to_value = dict((col, i) for i, col in enumerate(colors))
        matrix = np.array([col_to_value[col] for col in colors_original])[ind]
        new_shape = (1, len(colors_original))
        matrix = matrix.reshape(new_shape)
        cmap = mpl.colors.ListedColormap(colors)

        chp = cl._ClusteredHeatmapPlotter
        matrix2, cmap2 = chp.color_list_to_matrix_and_cmap(color_list, ind,
                                                           row=False)
        npt.assert_array_equal(matrix, matrix2)
        npt.assert_array_equal(cmap.colors, cmap2.colors)


    def test_get_linkage_function_scipy(self):
        import scipy.cluster.hierarchy as sch
        linkage_function = cl._ClusteredHeatmapPlotter.get_linkage_function(
            shape=self.data2d, use_fastcluster=False)
        nt.assert_is_instance(linkage_function, type(sch.linkage))

    def test_get_linkage_function_large_data(self):
        try:
            import fastcluster
            linkage = fastcluster.linkage
        except ImportError:
            import scipy.cluster.hierarchy as sch
            linkage = sch.linkage
        linkage_function = cl._ClusteredHeatmapPlotter.get_linkage_function(
            shape=(100, 100), use_fastcluster=False)
        nt.assert_is_instance(linkage_function, type(linkage))

    @skipif(_no_fastcluster)
    def test_get_linkage_function_fastcluster(self):
        import fastcluster
        linkage_function = cl._ClusteredHeatmapPlotter.get_linkage_function(
            shape=self.data2d, use_fastcluster=False)
        nt.assert_is_instance(linkage_function, type(fastcluster.linkage))

    def test_calculate_dendrogram(self):
        import scipy.spatial.distance as distance
        import scipy.cluster.hierarchy as sch
        sch.set_link_color_palette(['k'])

        row_pairwise_dists = distance.squareform(
            distance.pdist(self.data2d.values, metric='euclidean'))
        row_linkage = sch.linkage(row_pairwise_dists, method='average')
        dendrogram = sch.dendrogram(row_linkage, **self.default_dendrogram_kws)

        p = cl._ClusteredHeatmapPlotter(self.data2d)
        dendrogram2 = p.calculate_dendrogram(p.row_kws, row_linkage)
        npt.assert_equal(dendrogram, dendrogram2)

    def test_plot_dendrogram_row(self):
        f, ax = plt.subplots()
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        dendrogram = p.calculate_dendrogram(p.row_kws, p.row_linkage)
        p.plot_dendrogram(ax, dendrogram)
        npt.assert_equal(len(ax.get_lines()), self.data2d.shape[0]-1)
        plt.close("all")

    def test_plot_dendrogram_col(self):
        f, ax = plt.subplots()
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        dendrogram = p.calculate_dendrogram(p.col_kws, p.col_linkage)
        p.plot_dendrogram(ax, dendrogram, row=False)
        npt.assert_equal(len(ax.get_lines()), self.data2d.shape[1]-1)
        plt.close("all")

    def test_plot_side_colors_none(self):
        f, ax = plt.subplots()

        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        col_kws={'side_colors': None})
        dendrogram = p.calculate_dendrogram(p.col_kws, p.col_linkage)
        p.plot_side_colors(ax, p.col_kws, dendrogram, row=False)
        npt.assert_equal(len(ax.collections), 0)
        plt.close('all')

    def test_plot_side_colors_col(self):
        f, ax = plt.subplots()
        colors = color_palette(name='Set2', n_colors=3)
        np.random.seed(10)
        n = self.data2d.shape[1]
        color_inds = np.random.choice(np.arange(len(colors)), size=n).tolist()
        color_list = [colors[i] for i in color_inds]

        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        col_kws={'side_colors': color_list})
        dendrogram = p.calculate_dendrogram(p.col_kws, p.col_linkage)
        p.plot_side_colors(ax, p.col_kws, dendrogram, row=False)
        npt.assert_equal(len(ax.collections), 1)
        plt.close('all')

    def test_establish_axes_no_side_colors(self):
        width = self.data2d.shape[1] * 0.25
        height = min(self.data2d.shape[0] * .75, 40)
        figsize = (width, height)

        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.establish_axes()
        nt.assert_equal(len(p.fig.axes), 4)
        npt.assert_array_equal(p.fig.get_size_inches(), figsize)
        plt.close('all')

    def test_establish_axes_no_side_colors_figsize(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)

        figsize = (10, 10)
        p.establish_axes(figsize=figsize)

        nt.assert_equal(len(p.fig.axes), 4)
        npt.assert_array_equal(p.fig.get_size_inches(), figsize)
        plt.close('all')

    def test_establish_axes_col_side_colors(self):
        colors = color_palette(name='Set2', n_colors=3)
        np.random.seed(10)
        n = self.data2d.shape[1]
        color_inds = np.random.choice(np.arange(len(colors)), size=n).tolist()
        color_list = [colors[i] for i in color_inds]

        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        col_kws={'side_colors': color_list})
        p.establish_axes()

        nt.assert_equal(len(p.fig.axes), 5)
        nt.assert_equal(p.gs.get_geometry(), (4, 3))
        plt.close('all')

    def test_establish_axes_row_side_colors(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        row_kws={'side_colors':
                                                     self.row_colors})
        p.establish_axes()

        nt.assert_equal(len(p.fig.axes), 5)
        nt.assert_equal(p.gs.get_geometry(), (3, 4))
        plt.close('all')

    def test_establish_axes_both_side_colors(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        row_kws={'side_colors':
                                                     self.row_colors},
                                        col_kws={'side_colors':
                                                     self.col_colors})
        p.establish_axes()

        nt.assert_equal(len(p.fig.axes), 6)
        nt.assert_equal(p.gs.get_geometry(), (4, 4))
        plt.close('all')

    def test_plot_heatmap(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()
        nt.assert_equal(len(p.heatmap_ax.collections), 1)
        plt.close('all')

    def test_set_title(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()

        title = 'asdf'
        p.set_title(title)

        nt.assert_equal(p.col_dendrogram_ax.get_title(), title)
        plt.close('all')

    def test_label_dimension_none(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d, col_kws=dict(
            label=False), row_kws=dict(label=False))
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()
        p.label()

        nt.assert_equal(len(p.col_dendrogram_ax.get_yticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_xticklabels()), 0)
        nt.assert_equal(len(p.col_dendrogram_ax.get_xticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_yticklabels()), 0)
        plt.close('all')

    def test_label_dimension_both(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d, col_kws=dict(label=True),
                                        row_kws=dict(label=True))
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()
        p.label()

        # Make sure there aren't labels where there aren't supposed to be
        nt.assert_equal(len(p.col_dendrogram_ax.get_yticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_xticklabels()), 0)

        # Make sure the correct labels are where they're supposed to be
        xticklabels = map(lambda x: x._text,
                          p.col_dendrogram_ax.get_xticklabels())
        col_reordered = self.data2d.columns[p.col_dendrogram['leaves']].values

        yticklabels = map(lambda x: x._text,
                          p.row_dendrogram_ax.get_ymajorticklabels())
        row_reordered = self.data2d.index[p.row_dendrogram['leaves']].values

        npt.assert_equal(xticklabels, col_reordered)
        npt.assert_equal(yticklabels, row_reordered)
        plt.close('all')

    def test_label_dimension_col(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d, row_kws=dict(label=False))
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()
        p.label()

        # Make sure there aren't labels where there aren't supposed to be
        nt.assert_equal(len(p.col_dendrogram_ax.get_yticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_xticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_yticklabels()), 0)

        # Make sure the correct labels are where they're supposed to be
        xticklabels = map(lambda x: x._text,
                          p.col_dendrogram_ax.get_xticklabels())
        col_reordered = self.data2d.columns[p.col_dendrogram['leaves']].values

        npt.assert_equal(xticklabels, col_reordered)
        plt.close('all')

    def test_label_dimension_row(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d, col_kws=dict(label=True),
                                        row_kws=dict(label=True))
        p.establish_axes()
        p.plot_row_side()
        p.plot_col_side()
        p.plot_heatmap()
        p.label()

        # Make sure there aren't labels where there aren't supposed to be
        nt.assert_equal(len(p.col_dendrogram_ax.get_yticklabels()), 0)
        nt.assert_equal(len(p.row_dendrogram_ax.get_xticklabels()), 0)

        # Make sure the correct labels are where they're supposed to be
        yticklabels = map(lambda x: x._text,
                          p.row_dendrogram_ax.get_ymajorticklabels())
        row_reordered = self.data2d.index[p.row_dendrogram['leaves']].values

        npt.assert_equal(yticklabels, row_reordered)
        plt.close('all')

    def test_colorbar(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d, col_kws=dict(label=True),
                                        row_kws=dict(label=True))
        p.establish_axes()
        p.plot_heatmap()
        p.colorbar()

        nt.assert_equal(len(p.colorbar_ax.collections), 1)
        plt.close('all')

    def test_plot_col_side(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.establish_axes()
        p.plot_col_side()

        nt.assert_equal(len(p.col_dendrogram_ax.lines),
                        len(p.col_dendrogram['icoord']))
        nt.assert_equal(len(p.col_dendrogram_ax.lines),
                        len(p.col_dendrogram['dcoord']))
        plt.close('all')

    def test_plot_col_side_colors(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        col_kws={'side_colors':
                                                     self.col_colors})
        p.establish_axes()
        p.plot_col_side()

        nt.assert_equal(len(p.col_side_colors_ax.collections), 1)
        # Make sure xlim was set correctly
        nt.assert_equal(p.col_side_colors_ax.get_xlim(),
                        tuple(map(float, (0, p.data2d.shape[1]))))
        plt.close('all')

    def test_plot_row_side(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        row_kws={'side_colors':
                                                     self.row_colors})
        p.establish_axes()
        p.plot_row_side()

        nt.assert_equal(len(p.row_dendrogram_ax.lines),
                        len(p.row_dendrogram['icoord']))
        nt.assert_equal(len(p.row_dendrogram_ax.lines),
                        len(p.row_dendrogram['dcoord']))
        plt.close('all')

    def test_plot_row_side_colors(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d,
                                        row_kws={'side_colors':
                                                     self.row_colors})
        p.establish_axes()
        p.plot_row_side()

        nt.assert_equal(len(p.row_side_colors_ax.collections), 1)

        # Make sure ylim was set correctly
        nt.assert_equal(p.row_side_colors_ax.get_ylim(),
                        tuple(map(float, (0, p.data2d.shape[0]))))
        plt.close('all')

    def test_plot(self):
        # Should this test ALL the commands that are in
        # ClusteredHeatmapPlotter.plot? If so, how? Seems overkill to do all
        # the tests for each individual part again.
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.plot()

        fig, figsize, title, title_fontsize = None, None, None, None
        p2 = cl._ClusteredHeatmapPlotter(self.data2d)
        p2.establish_axes(fig, figsize)
        p2.plot_row_side()
        p2.plot_col_side()
        p2.plot_heatmap()
        p2.set_title(title, title_fontsize)
        p2.label()
        p2.colorbar()

        # gs = gridspec
        p2.gs.tight_layout(p2.fig)

        # Check that tight_layout was applied correctly
        nt.assert_equal(p.gs.bottom, p2.gs.bottom)
        nt.assert_equal(p.gs.hspace, p2.gs.hspace)
        nt.assert_equal(p.gs.left, p2.gs.left)
        nt.assert_equal(p.gs.right, p2.gs.right)
        nt.assert_equal(p.gs.top, p2.gs.top)
        nt.assert_equal(p.gs.wspace, p2.gs.wspace)
