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
from .. import utils

try:
    import fastcluster

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
                       cbar=True, cbar_kws=None)

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
        nt.assert_equal(p.xticklabels, ['' for _ in xrange(
            self.df_norm.shape[1])])
        nt.assert_equal(p.yticklabels, ['' for _ in xrange(
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

    def test_heatmap_annotation(self):

        ax = mat.heatmap(self.df_norm, annot=True, fmt=".1f",
                         annot_kws={"fontsize": 14})
        for val, text in zip(self.x_norm[::-1].flat, ax.texts):
            nt.assert_equal(text.get_text(), "{:.1f}".format(val))
            nt.assert_equal(text.get_fontsize(), 14)

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
                                                    method='median')
    except ImportError:
        x_norm_distances = distance.squareform(
            distance.pdist(x_norm.T, metric='euclidean'))
        x_norm_linkage = hierarchy.linkage(x_norm_distances, method='median')
    x_norm_dendrogram = hierarchy.dendrogram(x_norm_linkage, no_plot=True,
                                             color_list=['k'],
                                             color_threshold=-np.inf)
    x_norm_leaves = x_norm_dendrogram['leaves']
    df_norm_leaves = np.asarray(df_norm.columns[x_norm_leaves])

    default_kws = dict(linkage=None, metric='euclidean', method='median',
                       axis=1, ax=None, label=True, rotate=False)

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

        npt.assert_array_equal(p.reordered_ind, self.x_norm_leaves[::-1])

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

    @skipif(_no_fastcluster)
    def test_fastcluster_indexerror(self):
        import fastcluster

        kws = self.default_kws.copy()
        kws['method'] = 'average'
        linkage = fastcluster.linkage(self.x_norm.T, method='average',
                                      metric='euclidean')
        p = mat._DendrogramPlotter(self.x_norm, **kws)
        npt.assert_array_equal(p.linkage, linkage)

    def test_dendrogram_plot(self):
        d = mat.dendrogram(self.x_norm, **self.default_kws)

        d.xmin, d.xmax = d.ax.get_xlim()
        xmax = min(map(min, d.X)) + max(map(max, d.X))
        nt.assert_equal(d.xmin, 0)
        nt.assert_equal(d.xmax, xmax)

        nt.assert_equal(len(d.ax.get_lines()), len(d.X))
        nt.assert_equal(len(d.ax.get_lines()), len(d.Y))
        plt.close('all')

    def test_dendrogram_rotate(self):
        kws = self.default_kws.copy()
        kws['rotate'] = True

        d = mat.dendrogram(self.x_norm, **kws)

        d.ymin, d.ymax = d.ax.get_ylim()
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
        df.columns = [str(c) * 20 for c in df.columns]
        df.index = [i * 20 for i in df.index]

        f, ax = plt.subplots(figsize=(2, 2))
        mat.dendrogram(df, ax=ax)

        for t in ax.get_xticklabels():
            nt.assert_equal(t.get_rotation(), 90)

        plt.close(f)

        f, ax = plt.subplots(figsize=(2, 2))
        mat.dendrogram(df.T, axis=0, rotate=True)
        for t in ax.get_yticklabels():
            nt.assert_equal(t.get_rotation(), 0)