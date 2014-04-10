import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
from numpy.testing.decorators import skipif

from .. import clustering as cl
from ..palettes import color_palette


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

    def test_interpret_kws_from_none_divergent(self):
        p = cl._ClusteredHeatmapPlotter(self.data2d)
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        colorbar_kws=None)


        pdt.assert_dict_equal(p.row_kws, self.default_dim_kws)
        pdt.assert_dict_equal(p.col_kws, self.default_dim_kws)

        nt.assert_is_instance(p.cmap, mpl.cm.BuRd_r)
        pdt.assert_dict_equal(p.pcolormesh_kws, self.default_pcolormesh_kws)
        pdt.assert_dict_equal(p.colorbar_kws, self.default_colorbar_kws)
        nt.assert_is_none(p.norm)

    def test_interpret_kws_from_none_positive(self):
        p = cl._ClusteredHeatmapPlotter(np.abs(self.data2d))
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        colorbar_kws=None)
        nt.assert_is_instance(p.cmap, mpl.cm.YlGnBu)
        nt.assert_is_none(p.norm)

    def test_interpret_kws_from_none_log(self):
        p = cl._ClusteredHeatmapPlotter(np.abs(self.data2d))
        p.interpret_kws(row_kws=None, col_kws=None, pcolormesh_kws=None,
                        colorbar_kws=None, color_scale='log')
        nt.assert_is_instance(p.norm, mpl.colors.LogNorm)
        nt.assert_is_instance(p.cmap, mpl.cm.YlGnBu)
        pass

    def test_calculate_linkage(self):
        pass

    def test_get_width_ratios(self):
        pass

    def test_color_list_to_matrix_and_cmap(self):
        pass

    def test_get_linkage_function(self):
        pass

    def test_plot_dendrogram(self):
        pass

    def test_plot_sidecolors(self):
        pass

    def test_label_dimension(self):
        pass

