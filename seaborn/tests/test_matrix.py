import itertools
import tempfile

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from scipy.cluster import hierarchy

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
from numpy.testing.decorators import skipif

from .. import matrix as mat
from .. import color_palette
from ..external.six.moves import range

try:
    import fastcluster

    assert fastcluster
    _no_fastcluster = False
except ImportError:
    _no_fastcluster = True


class TestHeatmap(object):
    rs = np.random.RandomState(sum(map(ord, "heatmap")))

    x_norm = rs.randn(4, 8)
    letters = pd.Series(["A", "B", "C", "D"], name="letters")
    df_norm = pd.DataFrame(x_norm, index=letters)

    x_unif = rs.rand(20, 13)
    df_unif = pd.DataFrame(x_unif)

    default_kws = dict(vmin=None, vmax=None, cmap=None, center=None,
                       robust=False, annot=False, fmt=".2f", annot_kws=None,
                       cbar=True, cbar_kws=None, mask=None)

    def test_ndarray_input(self):

        p = mat._HeatMapper(self.x_norm, **self.default_kws)
        npt.assert_array_equal(p.plot_data, self.x_norm[::-1])
        pdt.assert_frame_equal(p.data, pd.DataFrame(self.x_norm).ix[::-1])

        npt.assert_array_equal(p.xticklabels, np.arange(8))
        npt.assert_array_equal(p.yticklabels, np.arange(4)[::-1])

        nt.assert_equal(p.xlabel, "")
        nt.assert_equal(p.ylabel, "")

    def test_df_input(self):

        p = mat._HeatMapper(self.df_norm, **self.default_kws)
        npt.assert_array_equal(p.plot_data, self.x_norm[::-1])
        pdt.assert_frame_equal(p.data, self.df_norm.ix[::-1])

        npt.assert_array_equal(p.xticklabels, np.arange(8))
        npt.assert_array_equal(p.yticklabels, ["D", "C", "B", "A"])

        nt.assert_equal(p.xlabel, "")
        nt.assert_equal(p.ylabel, "letters")

    def test_df_multindex_input(self):

        df = self.df_norm.copy()
        index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2),
                                           ("C", 3), ("D", 4)],
                                          names=["letter", "number"])
        index.name = "letter-number"
        df.index = index

        p = mat._HeatMapper(df, **self.default_kws)

        npt.assert_array_equal(p.yticklabels, ["D-4", "C-3", "B-2", "A-1"])
        nt.assert_equal(p.ylabel, "letter-number")

        p = mat._HeatMapper(df.T, **self.default_kws)

        npt.assert_array_equal(p.xticklabels, ["A-1", "B-2", "C-3", "D-4"])
        nt.assert_equal(p.xlabel, "letter-number")

    def test_mask_input(self):
        kws = self.default_kws.copy()

        mask = self.x_norm > 0
        kws['mask'] = mask
        p = mat._HeatMapper(self.x_norm, **kws)
        plot_data = np.ma.masked_where(mask, self.x_norm)

        npt.assert_array_equal(p.plot_data, plot_data[::-1])

    def test_default_sequential_vlims(self):

        p = mat._HeatMapper(self.df_unif, **self.default_kws)
        nt.assert_equal(p.vmin, self.x_unif.min())
        nt.assert_equal(p.vmax, self.x_unif.max())
        nt.assert_true(not p.divergent)

    def test_default_diverging_vlims(self):

        p = mat._HeatMapper(self.df_norm, **self.default_kws)
        vlim = max(abs(self.x_norm.min()), abs(self.x_norm.max()))
        nt.assert_equal(p.vmin, -vlim)
        nt.assert_equal(p.vmax, vlim)
        nt.assert_true(p.divergent)

    def test_robust_sequential_vlims(self):

        kws = self.default_kws.copy()
        kws["robust"] = True
        p = mat._HeatMapper(self.df_unif, **kws)

        nt.assert_equal(p.vmin, np.percentile(self.x_unif, 2))
        nt.assert_equal(p.vmax, np.percentile(self.x_unif, 98))

    def test_custom_sequential_vlims(self):

        kws = self.default_kws.copy()
        kws["vmin"] = 0
        kws["vmax"] = 1
        p = mat._HeatMapper(self.df_unif, **kws)

        nt.assert_equal(p.vmin, 0)
        nt.assert_equal(p.vmax, 1)

    def test_custom_diverging_vlims(self):

        kws = self.default_kws.copy()
        kws["vmin"] = -4
        kws["vmax"] = 5
        p = mat._HeatMapper(self.df_norm, **kws)

        nt.assert_equal(p.vmin, -5)
        nt.assert_equal(p.vmax, 5)

    def test_array_with_nans(self):

        x1 = self.rs.rand(10, 10)
        nulls = np.zeros(10) * np.nan
        x2 = np.c_[x1, nulls]

        m1 = mat._HeatMapper(x1, **self.default_kws)
        m2 = mat._HeatMapper(x2, **self.default_kws)

        nt.assert_equal(m1.vmin, m2.vmin)
        nt.assert_equal(m1.vmax, m2.vmax)

    def test_mask(self):

        df = pd.DataFrame(data={'a': [1, 1, 1],
                                'b': [2, np.nan, 2],
                                'c': [3, 3, np.nan]})

        kws = self.default_kws.copy()
        kws["mask"] = np.isnan(df.values)

        m = mat._HeatMapper(df, **kws)

        npt.assert_array_equal(np.isnan(m.plot_data.data),
                               m.plot_data.mask)

    def test_custom_cmap(self):

        kws = self.default_kws.copy()
        kws["cmap"] = "BuGn"
        p = mat._HeatMapper(self.df_unif, **kws)
        nt.assert_equal(p.cmap, "BuGn")

    def test_centered_vlims(self):

        kws = self.default_kws.copy()
        kws["center"] = .5

        p = mat._HeatMapper(self.df_unif, **kws)

        nt.assert_true(p.divergent)
        nt.assert_equal(p.vmax - .5, .5 - p.vmin)

    def test_tickabels_off(self):
        kws = self.default_kws.copy()
        kws['xticklabels'] = False
        kws['yticklabels'] = False
        p = mat._HeatMapper(self.df_norm, **kws)
        nt.assert_equal(p.xticklabels, ['' for _ in range(
            self.df_norm.shape[1])])
        nt.assert_equal(p.yticklabels, ['' for _ in range(
            self.df_norm.shape[0])])

    def test_custom_ticklabels(self):
        kws = self.default_kws.copy()
        xticklabels = list('iheartheatmaps'[:self.df_norm.shape[1]])
        yticklabels = list('heatmapsarecool'[:self.df_norm.shape[0]])
        kws['xticklabels'] = xticklabels
        kws['yticklabels'] = yticklabels
        p = mat._HeatMapper(self.df_norm, **kws)
        nt.assert_equal(p.xticklabels, xticklabels)
        nt.assert_equal(p.yticklabels, yticklabels[::-1])

    def test_custom_ticklabel_interval(self):

        kws = self.default_kws.copy()
        kws['xticklabels'] = 2
        kws['yticklabels'] = 3
        p = mat._HeatMapper(self.df_norm, **kws)

        nx, ny = self.df_norm.T.shape
        ystart = (ny - 1) % 3
        npt.assert_array_equal(p.xticks, np.arange(0, nx, 2) + .5)
        npt.assert_array_equal(p.yticks, np.arange(ystart, ny, 3) + .5)
        npt.assert_array_equal(p.xticklabels,
                               self.df_norm.columns[::2])
        npt.assert_array_equal(p.yticklabels,
                               self.df_norm.index[::-1][ystart:ny:3])

    def test_heatmap_annotation(self):

        ax = mat.heatmap(self.df_norm, annot=True, fmt=".1f",
                         annot_kws={"fontsize": 14})
        for val, text in zip(self.x_norm[::-1].flat, ax.texts):
            nt.assert_equal(text.get_text(), "{:.1f}".format(val))
            nt.assert_equal(text.get_fontsize(), 14)
        plt.close("all")

    def test_heatmap_annotation_with_mask(self):

        df = pd.DataFrame(data={'a': [1, 1, 1],
                                'b': [2, np.nan, 2],
                                'c': [3, 3, np.nan]})
        mask = np.isnan(df.values)
        df_masked = np.ma.masked_where(mask, df)
        ax = mat.heatmap(df, annot=True, fmt='.1f', mask=mask)
        nt.assert_equal(len(df_masked[::-1].compressed()), len(ax.texts))
        for val, text in zip(df_masked[::-1].compressed(), ax.texts):
            nt.assert_equal("{:.1f}".format(val), text.get_text())
        plt.close("all")

    def test_heatmap_cbar(self):

        f = plt.figure()
        mat.heatmap(self.df_norm)
        nt.assert_equal(len(f.axes), 2)
        plt.close(f)

        f = plt.figure()
        mat.heatmap(self.df_norm, cbar=False)
        nt.assert_equal(len(f.axes), 1)
        plt.close(f)

        f, (ax1, ax2) = plt.subplots(2)
        mat.heatmap(self.df_norm, ax=ax1, cbar_ax=ax2)
        nt.assert_equal(len(f.axes), 2)
        plt.close(f)

    def test_heatmap_axes(self):

        ax = mat.heatmap(self.df_norm)

        xtl = [int(l.get_text()) for l in ax.get_xticklabels()]
        nt.assert_equal(xtl, list(self.df_norm.columns))
        ytl = [l.get_text() for l in ax.get_yticklabels()]
        nt.assert_equal(ytl, list(self.df_norm.index[::-1]))

        nt.assert_equal(ax.get_xlabel(), "")
        nt.assert_equal(ax.get_ylabel(), "letters")

        nt.assert_equal(ax.get_xlim(), (0, 8))
        nt.assert_equal(ax.get_ylim(), (0, 4))

        plt.close("all")

    def test_heatmap_ticklabel_rotation(self):

        f, ax = plt.subplots(figsize=(2, 2))
        mat.heatmap(self.df_norm, ax=ax)

        for t in ax.get_xticklabels():
            nt.assert_equal(t.get_rotation(), 0)

        for t in ax.get_yticklabels():
            nt.assert_equal(t.get_rotation(), 90)

        plt.close(f)

        df = self.df_norm.copy()
        df.columns = [str(c) * 10 for c in df.columns]
        df.index = [i * 10 for i in df.index]

        f, ax = plt.subplots(figsize=(2, 2))
        mat.heatmap(df, ax=ax)

        for t in ax.get_xticklabels():
            nt.assert_equal(t.get_rotation(), 90)

        for t in ax.get_yticklabels():
            nt.assert_equal(t.get_rotation(), 0)

        plt.close(f)

    def test_heatmap_inner_lines(self):

        c = (0, 0, 1, 1)
        ax = mat.heatmap(self.df_norm, linewidths=2, linecolor=c)
        mesh = ax.collections[0]
        nt.assert_equal(mesh.get_linewidths()[0], 2)
        nt.assert_equal(tuple(mesh.get_edgecolor()[0]), c)

        plt.close("all")

    def test_square_aspect(self):

        ax = mat.heatmap(self.df_norm, square=True)
        nt.assert_equal(ax.get_aspect(), "equal")
        plt.close("all")

    def test_mask_validation(self):

        mask = mat._matrix_mask(self.df_norm, None)
        nt.assert_equal(mask.shape, self.df_norm.shape)
        nt.assert_equal(mask.values.sum(), 0)

        with nt.assert_raises(ValueError):
            bad_array_mask = self.rs.randn(3, 6) > 0
            mat._matrix_mask(self.df_norm, bad_array_mask)

        with nt.assert_raises(ValueError):
            bad_df_mask = pd.DataFrame(self.rs.randn(4, 8) > 0)
            mat._matrix_mask(self.df_norm, bad_df_mask)

    def test_missing_data_mask(self):

        data = pd.DataFrame(np.arange(4, dtype=np.float).reshape(2, 2))
        data.loc[0, 0] = np.nan
        mask = mat._matrix_mask(data, None)
        npt.assert_array_equal(mask, [[True, False], [False, False]])

        mask_in = np.array([[False, True], [False, False]])
        mask_out = mat._matrix_mask(data, mask_in)
        npt.assert_array_equal(mask_out, [[True, True], [False, False]])


class TestDendrogram(object):
    rs = np.random.RandomState(sum(map(ord, "dendrogram")))

    x_norm = rs.randn(4, 8) + np.arange(8)
    x_norm = (x_norm.T + np.arange(4)).T
    letters = pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"],
                        name="letters")

    df_norm = pd.DataFrame(x_norm, columns=letters)
    try:
        import fastcluster

        x_norm_linkage = fastcluster.linkage_vector(x_norm.T,
                                                    metric='euclidean',
                                                    method='single')
    except ImportError:
        x_norm_distances = distance.squareform(
            distance.pdist(x_norm.T, metric='euclidean'))
        x_norm_linkage = hierarchy.linkage(x_norm_distances, method='single')
    x_norm_dendrogram = hierarchy.dendrogram(x_norm_linkage, no_plot=True,
                                             color_list=['k'],
                                             color_threshold=-np.inf)
    x_norm_leaves = x_norm_dendrogram['leaves']
    df_norm_leaves = np.asarray(df_norm.columns[x_norm_leaves])

    default_kws = dict(linkage=None, metric='euclidean', method='single',
                       axis=1, label=True, rotate=False)

    def test_ndarray_input(self):
        p = mat._DendrogramPlotter(self.x_norm, **self.default_kws)
        npt.assert_array_equal(p.array.T, self.x_norm)
        pdt.assert_frame_equal(p.data.T, pd.DataFrame(self.x_norm))

        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        nt.assert_dict_equal(p.dendrogram, self.x_norm_dendrogram)

        npt.assert_array_equal(p.reordered_ind, self.x_norm_leaves)

        npt.assert_array_equal(p.xticklabels, self.x_norm_leaves)
        npt.assert_array_equal(p.yticklabels, [])

        nt.assert_equal(p.xlabel, None)
        nt.assert_equal(p.ylabel, '')

    def test_df_input(self):
        p = mat._DendrogramPlotter(self.df_norm, **self.default_kws)
        npt.assert_array_equal(p.array.T, np.asarray(self.df_norm))
        pdt.assert_frame_equal(p.data.T, self.df_norm)

        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        nt.assert_dict_equal(p.dendrogram, self.x_norm_dendrogram)

        npt.assert_array_equal(p.xticklabels,
                               np.asarray(self.df_norm.columns)[
                                   self.x_norm_leaves])
        npt.assert_array_equal(p.yticklabels, [])

        nt.assert_equal(p.xlabel, 'letters')
        nt.assert_equal(p.ylabel, '')

    def test_df_multindex_input(self):

        df = self.df_norm.copy()
        index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2),
                                           ("C", 3), ("D", 4)],
                                          names=["letter", "number"])
        index.name = "letter-number"
        df.index = index
        kws = self.default_kws.copy()
        kws['label'] = True

        p = mat._DendrogramPlotter(df.T, **kws)

        xticklabels = ["A-1", "B-2", "C-3", "D-4"]
        xticklabels = [xticklabels[i] for i in p.reordered_ind]
        npt.assert_array_equal(p.xticklabels, xticklabels)
        npt.assert_array_equal(p.yticklabels, [])
        nt.assert_equal(p.xlabel, "letter-number")

    def test_axis0_input(self):
        kws = self.default_kws.copy()
        kws['axis'] = 0
        p = mat._DendrogramPlotter(self.df_norm.T, **kws)

        npt.assert_array_equal(p.array, np.asarray(self.df_norm.T))
        pdt.assert_frame_equal(p.data, self.df_norm.T)

        npt.assert_array_equal(p.linkage, self.x_norm_linkage)
        nt.assert_dict_equal(p.dendrogram, self.x_norm_dendrogram)

        npt.assert_array_equal(p.xticklabels, self.df_norm_leaves)
        npt.assert_array_equal(p.yticklabels, [])

        nt.assert_equal(p.xlabel, 'letters')
        nt.assert_equal(p.ylabel, '')

    def test_rotate_input(self):
        kws = self.default_kws.copy()
        kws['rotate'] = True
        p = mat._DendrogramPlotter(self.df_norm, **kws)
        npt.assert_array_equal(p.array.T, np.asarray(self.df_norm))
        pdt.assert_frame_equal(p.data.T, self.df_norm)

        npt.assert_array_equal(p.xticklabels, [])
        npt.assert_array_equal(p.yticklabels, self.df_norm_leaves)

        nt.assert_equal(p.xlabel, '')
        nt.assert_equal(p.ylabel, 'letters')

    def test_rotate_axis0_input(self):
        kws = self.default_kws.copy()
        kws['rotate'] = True
        kws['axis'] = 0
        p = mat._DendrogramPlotter(self.df_norm.T, **kws)

        npt.assert_array_equal(p.reordered_ind, self.x_norm_leaves)

    def test_custom_linkage(self):
        kws = self.default_kws.copy()

        try:
            import fastcluster

            linkage = fastcluster.linkage_vector(self.x_norm, method='single',
                                                 metric='euclidean')
        except ImportError:
            d = distance.squareform(distance.pdist(self.x_norm,
                                                   metric='euclidean'))
            linkage = hierarchy.linkage(d, method='single')
        dendrogram = hierarchy.dendrogram(linkage, no_plot=True,
                                          color_list=['k'],
                                          color_threshold=-np.inf)
        kws['linkage'] = linkage
        p = mat._DendrogramPlotter(self.df_norm, **kws)

        npt.assert_array_equal(p.linkage, linkage)
        nt.assert_dict_equal(p.dendrogram, dendrogram)

    def test_label_false(self):
        kws = self.default_kws.copy()
        kws['label'] = False
        p = mat._DendrogramPlotter(self.df_norm, **kws)
        nt.assert_equal(p.xticks, [])
        nt.assert_equal(p.yticks, [])
        nt.assert_equal(p.xticklabels, [])
        nt.assert_equal(p.yticklabels, [])
        nt.assert_equal(p.xlabel, "")
        nt.assert_equal(p.ylabel, "")

    def test_linkage_scipy(self):
        p = mat._DendrogramPlotter(self.x_norm, **self.default_kws)

        scipy_linkage = p._calculate_linkage_scipy()

        from scipy.spatial import distance
        from scipy.cluster import hierarchy

        dists = distance.squareform(distance.pdist(self.x_norm.T,
                                                   metric=self.default_kws[
                                                       'metric']))
        linkage = hierarchy.linkage(dists, method=self.default_kws['method'])

        npt.assert_array_equal(scipy_linkage, linkage)

    @skipif(_no_fastcluster)
    def test_fastcluster_other_method(self):
        import fastcluster

        kws = self.default_kws.copy()
        kws['method'] = 'average'
        linkage = fastcluster.linkage(self.x_norm.T, method='average',
                                      metric='euclidean')
        p = mat._DendrogramPlotter(self.x_norm, **kws)
        npt.assert_array_equal(p.linkage, linkage)

    @skipif(_no_fastcluster)
    def test_fastcluster_non_euclidean(self):
        import fastcluster

        kws = self.default_kws.copy()
        kws['metric'] = 'cosine'
        kws['method'] = 'average'
        linkage = fastcluster.linkage(self.x_norm.T, method=kws['method'],
                                      metric=kws['metric'])
        p = mat._DendrogramPlotter(self.x_norm, **kws)
        npt.assert_array_equal(p.linkage, linkage)

    def test_dendrogram_plot(self):
        d = mat.dendrogram(self.x_norm, **self.default_kws)

        ax = plt.gca()
        d.xmin, d.xmax = ax.get_xlim()
        xmax = min(map(min, d.X)) + max(map(max, d.X))
        nt.assert_equal(d.xmin, 0)
        nt.assert_equal(d.xmax, xmax)

        nt.assert_equal(len(ax.get_lines()), len(d.X))
        nt.assert_equal(len(ax.get_lines()), len(d.Y))
        plt.close('all')

    def test_dendrogram_rotate(self):
        kws = self.default_kws.copy()
        kws['rotate'] = True

        d = mat.dendrogram(self.x_norm, **kws)

        ax = plt.gca()
        d.ymin, d.ymax = ax.get_ylim()
        ymax = min(map(min, d.Y)) + max(map(max, d.Y))
        nt.assert_equal(d.ymin, 0)
        nt.assert_equal(d.ymax, ymax)
        plt.close('all')

    def test_dendrogram_ticklabel_rotation(self):
        f, ax = plt.subplots(figsize=(2, 2))
        mat.dendrogram(self.df_norm, ax=ax)

        for t in ax.get_xticklabels():
            nt.assert_equal(t.get_rotation(), 0)

        plt.close(f)

        df = self.df_norm.copy()
        df.columns = [str(c) * 10 for c in df.columns]
        df.index = [i * 10 for i in df.index]

        f, ax = plt.subplots(figsize=(2, 2))
        mat.dendrogram(df, ax=ax)

        for t in ax.get_xticklabels():
            nt.assert_equal(t.get_rotation(), 90)

        plt.close(f)

        f, ax = plt.subplots(figsize=(2, 2))
        mat.dendrogram(df.T, axis=0, rotate=True)
        for t in ax.get_yticklabels():
            nt.assert_equal(t.get_rotation(), 0)
        plt.close(f)


class TestClustermap(object):
    rs = np.random.RandomState(sum(map(ord, "clustermap")))

    x_norm = rs.randn(4, 8) + np.arange(8)
    x_norm = (x_norm.T + np.arange(4)).T
    letters = pd.Series(["A", "B", "C", "D", "E", "F", "G", "H"],
                        name="letters")

    df_norm = pd.DataFrame(x_norm, columns=letters)
    try:
        import fastcluster

        x_norm_linkage = fastcluster.linkage_vector(x_norm.T,
                                                    metric='euclidean',
                                                    method='single')
    except ImportError:
        x_norm_distances = distance.squareform(
            distance.pdist(x_norm.T, metric='euclidean'))
        x_norm_linkage = hierarchy.linkage(x_norm_distances, method='single')
    x_norm_dendrogram = hierarchy.dendrogram(x_norm_linkage, no_plot=True,
                                             color_list=['k'],
                                             color_threshold=-np.inf)
    x_norm_leaves = x_norm_dendrogram['leaves']
    df_norm_leaves = np.asarray(df_norm.columns[x_norm_leaves])

    default_kws = dict(pivot_kws=None, z_score=None, standard_scale=None,
                       figsize=None, row_colors=None, col_colors=None)

    default_plot_kws = dict(metric='euclidean', method='average',
                            colorbar_kws=None,
                            row_cluster=True, col_cluster=True,
                            row_linkage=None, col_linkage=None)

    row_colors = color_palette('Set2', df_norm.shape[0])
    col_colors = color_palette('Dark2', df_norm.shape[1])

    def test_ndarray_input(self):
        cm = mat.ClusterGrid(self.x_norm, **self.default_kws)
        pdt.assert_frame_equal(cm.data, pd.DataFrame(self.x_norm))
        nt.assert_equal(len(cm.fig.axes), 4)
        nt.assert_equal(cm.ax_row_colors, None)
        nt.assert_equal(cm.ax_col_colors, None)

        plt.close('all')

    def test_df_input(self):
        cm = mat.ClusterGrid(self.df_norm, **self.default_kws)
        pdt.assert_frame_equal(cm.data, self.df_norm)

        plt.close('all')

    def test_corr_df_input(self):
        df = self.df_norm.corr()
        cg = mat.ClusterGrid(df, **self.default_kws)
        cg.plot(**self.default_plot_kws)
        diag = cg.data2d.values[np.diag_indices_from(cg.data2d)]
        npt.assert_array_equal(diag, np.ones(cg.data2d.shape[0]))

        plt.close('all')

    def test_pivot_input(self):
        df_norm = self.df_norm.copy()
        df_norm.index.name = 'numbers'
        df_long = pd.melt(df_norm.reset_index(), var_name='letters',
                          id_vars='numbers')
        kws = self.default_kws.copy()
        kws['pivot_kws'] = dict(index='numbers', columns='letters',
                                values='value')
        cm = mat.ClusterGrid(df_long, **kws)

        pdt.assert_frame_equal(cm.data2d, df_norm)

        plt.close('all')

    def test_colors_input(self):
        kws = self.default_kws.copy()

        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        cm = mat.ClusterGrid(self.df_norm, **kws)
        npt.assert_array_equal(cm.row_colors, self.row_colors)
        npt.assert_array_equal(cm.col_colors, self.col_colors)

        nt.assert_equal(len(cm.fig.axes), 6)
        plt.close('all')

    def test_nested_colors_input(self):
        kws = self.default_kws.copy()

        row_colors = [self.row_colors, self.row_colors]
        col_colors = [self.col_colors, self.col_colors]
        kws['row_colors'] = row_colors
        kws['col_colors'] = col_colors

        cm = mat.ClusterGrid(self.df_norm, **kws)
        npt.assert_array_equal(cm.row_colors, row_colors)
        npt.assert_array_equal(cm.col_colors, col_colors)

        nt.assert_equal(len(cm.fig.axes), 6)
        plt.close('all')

    def test_colors_input_custom_cmap(self):
        kws = self.default_kws.copy()

        kws['cmap'] = mpl.cm.PRGn
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        cm = mat.clustermap(self.df_norm, **kws)
        npt.assert_array_equal(cm.row_colors, self.row_colors)
        npt.assert_array_equal(cm.col_colors, self.col_colors)

        nt.assert_equal(len(cm.fig.axes), 6)
        plt.close('all')

    def test_z_score(self):
        df = self.df_norm.copy()
        df = (df - df.mean()) / df.var()
        kws = self.default_kws.copy()
        kws['z_score'] = 1

        cm = mat.ClusterGrid(self.df_norm, **kws)
        pdt.assert_frame_equal(cm.data2d, df)

        plt.close('all')

    def test_z_score_axis0(self):
        df = self.df_norm.copy()
        df = df.T
        df = (df - df.mean()) / df.var()
        df = df.T
        kws = self.default_kws.copy()
        kws['z_score'] = 0

        cm = mat.ClusterGrid(self.df_norm, **kws)
        pdt.assert_frame_equal(cm.data2d, df)

        plt.close('all')

    def test_standard_scale(self):
        df = self.df_norm.copy()
        df = (df - df.min()) / (df.max() - df.min())
        kws = self.default_kws.copy()
        kws['standard_scale'] = 1

        cm = mat.ClusterGrid(self.df_norm, **kws)
        pdt.assert_frame_equal(cm.data2d, df)

        plt.close('all')

    def test_standard_scale_axis0(self):
        df = self.df_norm.copy()
        df = df.T
        df = (df - df.min()) / (df.max() - df.min())
        df = df.T
        kws = self.default_kws.copy()
        kws['standard_scale'] = 0

        cm = mat.ClusterGrid(self.df_norm, **kws)
        pdt.assert_frame_equal(cm.data2d, df)

        plt.close('all')

    def test_z_score_standard_scale(self):
        kws = self.default_kws.copy()
        kws['z_score'] = True
        kws['standard_scale'] = True
        with nt.assert_raises(ValueError):
            cm = mat.ClusterGrid(self.df_norm, **kws)

        plt.close('all')

    def test_color_list_to_matrix_and_cmap(self):
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            self.col_colors, self.x_norm_leaves)

        colors_set = set(self.col_colors)
        col_to_value = dict((col, i) for i, col in enumerate(colors_set))
        matrix_test = np.array([col_to_value[col] for col in
                                self.col_colors])[self.x_norm_leaves]
        shape = len(self.col_colors), 1
        matrix_test = matrix_test.reshape(shape)
        cmap_test = mpl.colors.ListedColormap(colors_set)
        npt.assert_array_equal(matrix, matrix_test)
        npt.assert_array_equal(cmap.colors, cmap_test.colors)

        plt.close('all')

    def test_nested_color_list_to_matrix_and_cmap(self):
        colors = [self.col_colors, self.col_colors]
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            colors, self.x_norm_leaves)

        all_colors = set(itertools.chain(*colors))
        color_to_value = dict((col, i) for i, col in enumerate(all_colors))
        matrix_test = np.array(
            [color_to_value[c] for color in colors for c in color])
        shape = len(colors), len(colors[0])
        matrix_test = matrix_test.reshape(shape)
        matrix_test = matrix_test[:, self.x_norm_leaves]
        matrix_test = matrix_test.T

        cmap_test = mpl.colors.ListedColormap(all_colors)
        npt.assert_array_equal(matrix, matrix_test)
        npt.assert_array_equal(cmap.colors, cmap_test.colors)

        plt.close('all')

    def test_color_list_to_matrix_and_cmap_axis1(self):
        matrix, cmap = mat.ClusterGrid.color_list_to_matrix_and_cmap(
            self.col_colors, self.x_norm_leaves, axis=1)

        colors_set = set(self.col_colors)
        col_to_value = dict((col, i) for i, col in enumerate(colors_set))
        matrix_test = np.array([col_to_value[col] for col in
                                self.col_colors])[self.x_norm_leaves]
        shape = 1, len(self.col_colors)
        matrix_test = matrix_test.reshape(shape)
        cmap_test = mpl.colors.ListedColormap(colors_set)
        npt.assert_array_equal(matrix, matrix_test)
        npt.assert_array_equal(cmap.colors, cmap_test.colors)

        plt.close('all')

    def test_savefig(self):
        # Not sure if this is the right way to test....
        cm = mat.ClusterGrid(self.df_norm, **self.default_kws)
        cm.plot(**self.default_plot_kws)
        cm.savefig(tempfile.NamedTemporaryFile(), format='png')

        plt.close('all')

    def test_plot_dendrograms(self):
        cm = mat.clustermap(self.df_norm, **self.default_kws)

        nt.assert_equal(len(cm.ax_row_dendrogram.get_lines()),
                        len(cm.dendrogram_row.X))
        nt.assert_equal(len(cm.ax_col_dendrogram.get_lines()),
                        len(cm.dendrogram_col.X))
        data2d = self.df_norm.iloc[cm.dendrogram_row.reordered_ind,
                                   cm.dendrogram_col.reordered_ind]
        pdt.assert_frame_equal(cm.data2d, data2d)
        plt.close('all')

    def test_cluster_false(self):
        kws = self.default_kws.copy()
        kws['row_cluster'] = False
        kws['col_cluster'] = False

        cm = mat.clustermap(self.df_norm, **kws)
        nt.assert_equal(len(cm.ax_row_dendrogram.lines), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.lines), 0)

        nt.assert_equal(len(cm.ax_row_dendrogram.get_xticks()), 0)
        nt.assert_equal(len(cm.ax_row_dendrogram.get_yticks()), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.get_xticks()), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.get_yticks()), 0)

        pdt.assert_frame_equal(cm.data2d, self.df_norm)
        plt.close('all')

    def test_row_col_colors(self):
        kws = self.default_kws.copy()
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        cm = mat.clustermap(self.df_norm, **kws)

        nt.assert_equal(len(cm.ax_row_colors.collections), 1)
        nt.assert_equal(len(cm.ax_col_colors.collections), 1)

        plt.close('all')

    def test_cluster_false_row_col_colors(self):
        kws = self.default_kws.copy()
        kws['row_cluster'] = False
        kws['col_cluster'] = False
        kws['row_colors'] = self.row_colors
        kws['col_colors'] = self.col_colors

        cm = mat.clustermap(self.df_norm, **kws)
        nt.assert_equal(len(cm.ax_row_dendrogram.lines), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.lines), 0)

        nt.assert_equal(len(cm.ax_row_dendrogram.get_xticks()), 0)
        nt.assert_equal(len(cm.ax_row_dendrogram.get_yticks()), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.get_xticks()), 0)
        nt.assert_equal(len(cm.ax_col_dendrogram.get_yticks()), 0)
        nt.assert_equal(len(cm.ax_row_colors.collections), 1)
        nt.assert_equal(len(cm.ax_col_colors.collections), 1)

        pdt.assert_frame_equal(cm.data2d, self.df_norm)
        plt.close('all')

    def test_mask_reorganization(self):

        kws = self.default_kws.copy()
        kws["mask"] = self.df_norm > 0

        g = mat.clustermap(self.df_norm, **kws)
        npt.assert_array_equal(g.data2d.index, g.mask.index)
        npt.assert_array_equal(g.data2d.columns, g.mask.columns)

        npt.assert_array_equal(g.mask.index,
                               self.df_norm.index[
                                   g.dendrogram_row.reordered_ind])
        npt.assert_array_equal(g.mask.columns,
                               self.df_norm.columns[
                                   g.dendrogram_col.reordered_ind])

        plt.close("all")

    def test_ticklabel_reorganization(self):

        kws = self.default_kws.copy()
        xtl = np.arange(self.df_norm.shape[1])
        kws["xticklabels"] = list(xtl)
        ytl = self.letters.ix[:self.df_norm.shape[0]]
        kws["yticklabels"] = ytl

        g = mat.clustermap(self.df_norm, **kws)

        xtl_actual = [t.get_text() for t in g.ax_heatmap.get_xticklabels()]
        ytl_actual = [t.get_text() for t in g.ax_heatmap.get_yticklabels()]

        xtl_want = xtl[g.dendrogram_col.reordered_ind].astype("<U1")
        ytl_want = ytl[g.dendrogram_row.reordered_ind].astype("<U1")[::-1]

        npt.assert_array_equal(xtl_actual, xtl_want)
        npt.assert_array_equal(ytl_actual, ytl_want)

        plt.close("all")
