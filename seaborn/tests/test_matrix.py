import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
from numpy.testing.decorators import skipif

from .. import matrix as mat
from .. import utils


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
