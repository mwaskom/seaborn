import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        width_ratios = p.get_fig_width_ratios(side_colors=['asdf'],
                                        dimension='width')
        height_ratios = p.get_fig_width_ratios(side_colors=['asdf'],
                                              dimension='height')
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
            shape=self.data2d)
        npt.assert_equal(linkage_function, sch.linkage)

    def test_get_linkage_function_large_data(self):
        try:
            import fastcluster
            linkage = fastcluster.linkage
        except ImportError:
            import scipy.cluster.hierarchy as sch
            linkage = sch.linkage
        linkage_function = cl._ClusteredHeatmapPlotter.get_linkage_function(
            shape=(100, 100))
        npt.assert_equal(linkage_function, linkage)

    @skipif(_no_fastcluster)
    def test_get_linkage_function_fastcluster(self):
        import fastcluster
        linkage_function = cl._ClusteredHeatmapPlotter.get_linkage_function(
            shape=self.data2d)
        npt.assert_equal(linkage_function, fastcluster.linkage)

    def test_plot_dendrogram(self):
        f, ax = plt.subplots()
        p = cl._ClusteredHeatmapPlotter(self.data2d)


    def test_plot_sidecolors(self):
        pass

    def test_label_dimension(self):
        pass

    def test_plot_heatmap(self):
        pass

    def test_set_title(self):
        pass

    def test_colorbar(self):
        pass

    def test_plot_col_side(self):
        pass

    def test_plot_row_side(self):
        pass

    def test_label(self):
        pass

    def test_plot(self):
        pass


