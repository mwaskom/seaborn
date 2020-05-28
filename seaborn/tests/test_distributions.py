import itertools
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats, integrate

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .. import distributions as dist
from ..palettes import (
    color_palette,
)
from .._core import (
    categorical_order,
)
from ..distributions import (
    rugplot,
    kdeplot,
)

_no_statsmodels = not dist._has_statsmodels

if not _no_statsmodels:
    import statsmodels.nonparametric as smnp
else:
    _old_statsmodels = False


class TestDistPlot(object):

    rs = np.random.RandomState(0)
    x = rs.randn(100)

    def test_hist_bins(self):

        try:
            fd_edges = np.histogram_bin_edges(self.x, "fd")
        except AttributeError:
            pytest.skip("Requires numpy >= 1.15")
        ax = dist.distplot(x=self.x)
        for edge, bar in zip(fd_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

        plt.close(ax.figure)
        n = 25
        n_edges = np.histogram_bin_edges(self.x, n)
        ax = dist.distplot(x=self.x, bins=n)
        for edge, bar in zip(n_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

    def test_elements(self):

        n = 10
        ax = dist.distplot(x=self.x, bins=n,
                           hist=True, kde=False, rug=False, fit=None)
        assert len(ax.patches) == 10
        assert len(ax.lines) == 0
        assert len(ax.collections) == 0

        plt.close(ax.figure)
        ax = dist.distplot(x=self.x,
                           hist=False, kde=True, rug=False, fit=None)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

        plt.close(ax.figure)
        ax = dist.distplot(x=self.x,
                           hist=False, kde=False, rug=True, fit=None)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0
        assert len(ax.collections) == 1

        plt.close(ax.figure)
        ax = dist.distplot(x=self.x,
                           hist=False, kde=False, rug=False, fit=stats.norm)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

    def test_distplot_with_nans(self):

        f, (ax1, ax2) = plt.subplots(2)
        x_null = np.append(self.x, [np.nan])

        dist.distplot(x=self.x, ax=ax1)
        dist.distplot(x=x_null, ax=ax2)

        line1 = ax1.lines[0]
        line2 = ax2.lines[0]
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

        for bar1, bar2 in zip(ax1.patches, ax2.patches):
            assert bar1.get_xy() == bar2.get_xy()
            assert bar1.get_height() == bar2.get_height()

    def test_a_parameter_deprecation(self):

        n = 10
        with pytest.warns(UserWarning):
            ax = dist.distplot(a=self.x, bins=n)
        assert len(ax.patches) == n


class TestRugPlot:

    def assert_rug_equal(self, a, b):

        assert_array_equal(a.get_segments(), b.get_segments())

    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_long_data(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, np.asarray(vector), vector.tolist(),
        ]

        f, ax = plt.subplots()
        for vector in vectors:
            rugplot(data=long_df, **{variable: vector})

        for a, b in itertools.product(ax.collections, ax.collections):
            self.assert_rug_equal(a, b)

    def test_bivariate_data(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        rugplot(data=long_df, x="x", y="y", ax=ax1)
        rugplot(data=long_df, x="x", ax=ax2)
        rugplot(data=long_df, y="y", ax=ax2)

        self.assert_rug_equal(ax1.collections[0], ax2.collections[0])
        self.assert_rug_equal(ax1.collections[1], ax2.collections[1])

    def test_wide_vs_long_data(self, wide_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)
        rugplot(data=wide_df, ax=ax1)
        for col in wide_df:
            rugplot(data=wide_df, x=col, ax=ax2)

        wide_segments = np.sort(
            np.array(ax1.collections[0].get_segments())
        )
        long_segments = np.sort(
            np.concatenate([c.get_segments() for c in ax2.collections])
        )

        assert_array_equal(wide_segments, long_segments)

    def test_flat_vector(self, long_df):

        f, ax = plt.subplots()
        rugplot(data=long_df["x"])
        rugplot(x=long_df["x"])
        self.assert_rug_equal(*ax.collections)

    def test_empty_data(self):

        ax = rugplot(x=[])
        assert not ax.collections

    def test_a_deprecation(self, flat_series):

        f, ax = plt.subplots()

        with pytest.warns(FutureWarning):
            rugplot(a=flat_series)
        rugplot(x=flat_series)

        self.assert_rug_equal(*ax.collections)

    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_axis_deprecation(self, flat_series, variable):

        f, ax = plt.subplots()

        with pytest.warns(FutureWarning):
            rugplot(flat_series, axis=variable)
        rugplot(**{variable: flat_series})

        self.assert_rug_equal(*ax.collections)

    def test_vertical_deprecation(self, flat_series):

        f, ax = plt.subplots()

        with pytest.warns(FutureWarning):
            rugplot(flat_series, vertical=True)
        rugplot(y=flat_series)

        self.assert_rug_equal(*ax.collections)

    def test_rug_data(self, flat_array):

        height = .05
        ax = rugplot(x=flat_array, height=height)
        segments = np.stack(ax.collections[0].get_segments())

        n = flat_array.size
        assert_array_equal(segments[:, 0, 1], np.zeros(n))
        assert_array_equal(segments[:, 1, 1], np.full(n, height))
        assert_array_equal(segments[:, 1, 0], flat_array)

    def test_rug_colors(self, long_df):

        ax = rugplot(data=long_df, x="x", hue="a")

        order = categorical_order(long_df["a"])
        palette = color_palette()

        expected_colors = np.ones((len(long_df), 4))
        for i, val in enumerate(long_df["a"]):
            expected_colors[i, :3] = palette[order.index(val)]

        assert_array_equal(ax.collections[0].get_color(), expected_colors)

    def test_expand_margins(self, flat_array):

        f, ax = plt.subplots()
        x1, y1 = ax.margins()
        rugplot(x=flat_array, expand_margins=False)
        x2, y2 = ax.margins()
        assert x1 == x2
        assert y1 == y2

        f, ax = plt.subplots()
        x1, y1 = ax.margins()
        height = .05
        rugplot(x=flat_array, height=height)
        x2, y2 = ax.margins()
        assert x1 == x2
        assert y1 + height * 2 == pytest.approx(y2)

    def test_matplotlib_kwargs(self, flat_series):

        lw = 2
        alpha = .2
        ax = rugplot(y=flat_series, linewidth=lw, alpha=alpha)
        rug = ax.collections[0]
        assert np.all(rug.get_alpha() == alpha)
        assert np.all(rug.get_linewidth() == lw)

    def test_axis_labels(self, flat_series):

        ax = rugplot(x=flat_series)
        assert ax.get_xlabel() == flat_series.name
        assert not ax.get_ylabel()


class TestKDEPlot:

    @pytest.mark.parametrize(
        "variable", ["x", "y"],
    )
    def test_long_vectors(self, long_df, variable):

        vector = long_df[variable]
        vectors = [
            variable, vector, np.asarray(vector), vector.tolist(),
        ]

        f, ax = plt.subplots()
        for vector in vectors:
            kdeplot(data=long_df, **{variable: vector})

        xdata = [l.get_xdata() for l in ax.lines]
        for a, b in itertools.product(xdata, xdata):
            assert_array_equal(a, b)

        ydata = [l.get_ydata() for l in ax.lines]
        for a, b in itertools.product(ydata, ydata):
            assert_array_equal(a, b)

    def test_wide_vs_long_data(self, wide_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(data=wide_df, ax=ax1, common_norm=False, common_grid=False)
        for col in wide_df:
            kdeplot(data=wide_df, x=col, ax=ax2)

        for l1, l2 in zip(ax1.lines[::-1], ax2.lines):
            assert_array_equal(l1.get_xydata(), l2.get_xydata())

    def test_flat_vector(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df["x"])
        kdeplot(x=long_df["x"])
        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_empty_data(self):

        ax = kdeplot(x=[])
        assert not ax.lines

    def test_singular_data(self):

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=np.ones(10))
        assert not ax.lines

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=[5])
        assert not ax.lines

    def test_variable_assignment(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", fill=True)
        kdeplot(data=long_df, y="x", fill=True)

        assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_ydata())
        assert_array_equal(ax.lines[0].get_ydata(), ax.lines[1].get_xdata())

        v0 = ax.collections[0].get_paths()[0].vertices
        v1 = ax.collections[1].get_paths()[0].vertices

        assert np.in1d(ax.lines[0].get_ydata(), v0).all()
        assert np.in1d(ax.lines[1].get_ydata(), v1).all()

    def test_vertical_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, y="x")

        with pytest.warns(FutureWarning):
            kdeplot(data=long_df, x="x", vertical=True)

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_bw_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_method="silverman")

        with pytest.warns(FutureWarning):
            kdeplot(data=long_df, x="x", bw="silverman")

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_kernel_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x")

        with pytest.warns(UserWarning):
            kdeplot(data=long_df, x="x", kernel="epi")

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    def test_shade_deprecation(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", shade=True)
        kdeplot(data=long_df, x="x", fill=True)
        fill1, fill2 = ax.collections
        assert_array_equal(
            fill1.get_paths()[0].vertices, fill2.get_paths()[0].vertices
        )

    @pytest.mark.parametrize(
        "hue_method", ["layer", "stack", "fill"],
    )
    def test_hue_colors(self, long_df, hue_method):

        ax = kdeplot(
            data=long_df, x="x", hue="a",
            hue_method=hue_method,
            fill=True, legend=False
        )

        # Note that hue order is reversed in the plot
        lines = ax.lines[::-1]
        fills = ax.collections[::-1]

        palette = color_palette()

        for line, fill, color in zip(lines, fills, palette):
            assert line.get_color() == color
            assert tuple(fill.get_facecolor().squeeze()) == color + (.25,)

    def test_hue_stacking(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="a",
            hue_method="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            hue_method="stack",
            legend=False, ax=ax2,
        )

        layered_densities = np.stack([
            l.get_ydata() for l in ax1.lines
        ])
        stacked_densities = np.stack([
            l.get_ydata() for l in ax2.lines
        ])

        assert_array_equal(layered_densities.cumsum(axis=0), stacked_densities)

    def test_hue_filling(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="a",
            hue_method="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            hue_method="fill",
            legend=False, ax=ax2,
        )

        layered = np.stack([l.get_ydata() for l in ax1.lines])
        filled = np.stack([l.get_ydata() for l in ax2.lines])

        assert_array_almost_equal(
            (layered / layered.sum(axis=0)).cumsum(axis=0),
            filled,
        )

    def test_cut(self, rng):

        x = rng.normal(0, 3, 1000)

        f, ax = plt.subplots()
        kdeplot(x=x, cut=0, legend=False)

        xdata_0 = ax.lines[0].get_xdata()
        assert xdata_0.min() == x.min()
        assert xdata_0.max() == x.max()

        kdeplot(x=x, cut=2, legend=False)

        xdata_2 = ax.lines[1].get_xdata()
        assert xdata_2.min() < xdata_0.min()
        assert xdata_2.max() > xdata_0.max()

        assert len(xdata_0) == len(xdata_2)

    def test_clip(self, rng):

        x = rng.normal(0, 3, 1000)

        clip = -1, 1
        ax = kdeplot(x=x, clip=clip)

        xdata = ax.lines[0].get_xdata()

        assert xdata.min() >= clip[0]
        assert xdata.max() <= clip[1]

    def test_line_is_density(self, long_df):

        ax = kdeplot(data=long_df, x="x", cut=5)
        x, y = ax.lines[0].get_xydata().T
        assert integrate.trapz(y, x) == pytest.approx(1)

    def test_cumulative(self, long_df):

        ax = kdeplot(data=long_df, x="x", cut=5, cumulative=True)
        y = ax.lines[0].get_ydata()
        assert y[0] == pytest.approx(0)
        assert y[-1] == pytest.approx(1)

    def test_common_norm(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(
            data=long_df, x="x", hue="a", common_norm=True, cut=10, ax=ax1
        )
        kdeplot(
            data=long_df, x="x", hue="a", common_norm=False, cut=10, ax=ax2
        )

        total_area = 0
        for line in ax1.lines:
            xdata, ydata = line.get_xydata().T
            total_area += integrate.trapz(ydata, xdata)
        assert total_area == pytest.approx(1)

        for line in ax2.lines:
            xdata, ydata = line.get_xydata().T
            assert integrate.trapz(ydata, xdata) == pytest.approx(1)

    def test_common_grid(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        order = "a", "b", "c"

        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=False, cut=0, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=True, cut=0, ax=ax2,
        )

        for line, level in zip(ax1.lines[::-1], order):
            xdata = line.get_xdata()
            assert xdata.min() == long_df.loc[long_df["a"] == level, "x"].min()
            assert xdata.max() == long_df.loc[long_df["a"] == level, "x"].max()

        for line in ax2.lines:
            xdata = line.get_xdata().T
            assert xdata.min() == long_df["x"].min()
            assert xdata.max() == long_df["x"].max()

    def test_bw_method(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_method=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_method=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_method=3.0, legend=False)

        l1, l2, l3 = ax.lines

        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    def test_bw_adjust(self, long_df):

        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_adjust=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=3.0, legend=False)

        l1, l2, l3 = ax.lines

        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    def test_log_scale_implicit(self, rng):

        x = rng.lognormal(0, 1, 100)

        f, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.set_xscale("log")

        kdeplot(x=x, ax=ax1)
        kdeplot(x=x, ax=ax1)

        xdata_log = ax1.lines[0].get_xdata()
        assert (xdata_log > 0).all()
        assert (np.diff(xdata_log, 2) > 0).all()
        assert np.allclose(np.diff(np.log(xdata_log), 2), 0)

        f, ax = plt.subplots()
        ax.set_yscale("log")
        kdeplot(y=x, ax=ax)
        assert_array_equal(ax.lines[0].get_xdata(), ax1.lines[0].get_ydata())

    def test_log_scale_explicit(self, rng):

        x = rng.lognormal(0, 1, 100)

        f, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        ax1.set_xscale("log")
        kdeplot(x=x, ax=ax1)
        kdeplot(x=x, log_scale=True, ax=ax2)
        kdeplot(x=x, log_scale=10, ax=ax3)

        for ax in f.axes:
            assert ax.get_xscale() == "log"

        supports = [ax.lines[0].get_xdata() for ax in f.axes]
        for a, b in itertools.product(supports, supports):
            assert_array_equal(a, b)

        densities = [ax.lines[0].get_ydata() for ax in f.axes]
        for a, b in itertools.product(densities, densities):
            assert_array_equal(a, b)

        f, ax = plt.subplots()
        kdeplot(y=x, log_scale=True, ax=ax)
        assert ax.get_yscale() == "log"

    def test_log_scale_with_hue(self, rng):

        data = rng.lognormal(0, 1, 50), rng.lognormal(0, 2, 100)
        ax = kdeplot(data=data, log_scale=True, common_grid=True)
        assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_xdata())

    def test_log_scale_normalization(self, rng):

        x = rng.lognormal(0, 1, 100)
        ax = kdeplot(x=x, log_scale=True, cut=10)
        xdata, ydata = ax.lines[0].get_xydata().T
        integral = integrate.trapz(ydata, np.log10(xdata))
        assert integral == pytest.approx(1)

    @pytest.mark.skipif(
        LooseVersion(scipy.__version__) < "1.2.0",
        reason="Weights require scipy >= 1.2.0"
    )
    def test_weights(self):

        x = [1, 2]
        weights = [2, 1]

        ax = kdeplot(x=x, weights=weights)

        xdata, ydata = ax.lines[0].get_xydata().T

        y1 = ydata[np.argwhere(np.abs(xdata - 1).min())]
        y2 = ydata[np.argwhere(np.abs(xdata - 2).min())]

        assert y1 == pytest.approx(2 * y2)

    def test_sticky_edges(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(data=long_df, x="x", fill=True, ax=ax1)
        assert ax1.get_ylim()[0] == 0

        kdeplot(
            data=long_df, x="x", hue="a", hue_method="fill", fill=True, ax=ax2
        )
        assert ax2.get_ylim() == pytest.approx((0, 1))  # old mpl needs approx?

    def test_line_kws(self, flat_array):

        lw = 3
        color = (.2, .5, .8)
        ax = kdeplot(x=flat_array, linewidth=lw, color=color)
        line, = ax.lines
        assert line.get_linewidth() == lw
        assert line.get_color() == color

    def test_fill_kws(self, flat_array):

        color = (.2, .5, .8)
        alpha = .5
        fill_kws = dict(
            alpha=alpha,
        )
        ax = kdeplot(
            x=flat_array, fill=True, fill_kws=fill_kws, color=color
        )
        fill = ax.collections[0]
        assert tuple(fill.get_facecolor().squeeze()) == color + (alpha,)

    def test_input_checking(self, long_df):

        err = (
            "kdeplot requires a numeric 'x' variable, "
            "but a datetime was passed"
        )
        with pytest.raises(TypeError, match=err):
            kdeplot(data=long_df, x="t")

        err = (
            "kdeplot requires a numeric 'x' variable, "
            "but a categorical was passed"
        )
        with pytest.raises(TypeError, match=err):
            kdeplot(data=long_df, x="a")

    def test_axis_labels(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(data=long_df, x="x", ax=ax1)
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "Density"

        kdeplot(data=long_df, y="y", ax=ax2)
        assert ax2.get_xlabel() == "Density"
        assert ax2.get_ylabel() == "y"

    def test_legend(self, long_df):

        ax = kdeplot(data=long_df, x="x", hue="a")

        assert ax.legend_.get_title().get_text() == "a"

        legend_labels = ax.legend_.get_texts()
        order = categorical_order(long_df["a"])
        for label, level in zip(legend_labels, order):
            assert label.get_text() == level

        legend_artists = ax.legend_.findobj(mpl.lines.Line2D)[::2]
        palette = color_palette()
        for artist, color in zip(legend_artists, palette):
            assert artist.get_color() == color

        ax.clear()

        kdeplot(data=long_df, x="x", hue="a", legend=False)

        assert ax.legend_ is None


class TestKDE(object):

    rs = np.random.RandomState(0)
    x = rs.randn(50)
    y = rs.randn(50)
    kernel = "gau"
    bw = "scott"
    gridsize = 128
    clip = (-np.inf, np.inf)
    cut = 3

    def test_kde_1d_input_output(self):
        """Test that array/series/list inputs give the same output."""
        f, ax = plt.subplots()

        dist.kdeplot(x=self.x)
        dist.kdeplot(x=pd.Series(self.x))
        dist.kdeplot(x=self.x.tolist())
        dist.kdeplot(data=self.x)

        supports = [l.get_xdata() for l in ax.lines]
        for a, b in itertools.product(supports, supports):
            assert np.array_equal(a, b)

        densities = [l.get_ydata() for l in ax.lines]
        for a, b in itertools.product(densities, densities):
            assert np.array_equal(a, b)

    def test_kde_2d_input_output(self):
        """Test that array/series/list inputs give the same output."""
        f, ax = plt.subplots()

        dist.kdeplot(x=self.x, y=self.y)
        dist.kdeplot(x=pd.Series(self.x), y=pd.Series(self.y))
        dist.kdeplot(x=self.x.tolist(), y=self.y.tolist())

        contours = ax.collections
        n = len(contours) // 3

        for i in range(n):
            for a, b in itertools.product(contours[i::n], contours[i::n]):
                assert np.array_equal(a.get_offsets(), b.get_offsets())

    def test_scipy_univariate_kde(self):
        """Test the univariate KDE estimation with scipy."""
        grid, y = dist._scipy_univariate_kde(self.x, self.bw, self.gridsize,
                                             self.cut, self.clip)
        assert len(grid) == self.gridsize
        assert len(y) == self.gridsize
        for bw in ["silverman", .2]:
            dist._scipy_univariate_kde(self.x, bw, self.gridsize,
                                       self.cut, self.clip)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_univariate_kde(self):
        """Test the univariate KDE estimation with statsmodels."""
        grid, y = dist._statsmodels_univariate_kde(self.x, self.kernel,
                                                   self.bw, self.gridsize,
                                                   self.cut, self.clip)
        assert len(grid) == self.gridsize
        assert len(y) == self.gridsize
        for bw in ["silverman", .2]:
            dist._statsmodels_univariate_kde(self.x, self.kernel, bw,
                                             self.gridsize, self.cut,
                                             self.clip)

    def test_scipy_bivariate_kde(self):
        """Test the bivariate KDE estimation with scipy."""
        clip = [self.clip, self.clip]
        x, y, z = dist._scipy_bivariate_kde(self.x, self.y, self.bw,
                                            self.gridsize, self.cut, clip)
        assert x.shape == (self.gridsize, self.gridsize)
        assert y.shape == (self.gridsize, self.gridsize)
        assert len(z) == self.gridsize

        # Test a specific bandwidth
        clip = [self.clip, self.clip]
        x, y, z = dist._scipy_bivariate_kde(self.x, self.y, 1,
                                            self.gridsize, self.cut, clip)

        # Test that we get an error with an invalid bandwidth
        with pytest.raises(ValueError):
            dist._scipy_bivariate_kde(self.x, self.y, (1, 2),
                                      self.gridsize, self.cut, clip)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_bivariate_kde(self):
        """Test the bivariate KDE estimation with statsmodels."""
        clip = [self.clip, self.clip]
        x, y, z = dist._statsmodels_bivariate_kde(self.x, self.y, self.bw,
                                                  self.gridsize,
                                                  self.cut, clip)
        assert x.shape == (self.gridsize, self.gridsize)
        assert y.shape == (self.gridsize, self.gridsize)
        assert len(z) == self.gridsize

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_kde_cumulative(self):
        """Test computation of cumulative KDE."""
        grid, y = dist._statsmodels_univariate_kde(self.x, self.kernel,
                                                   self.bw, self.gridsize,
                                                   self.cut, self.clip,
                                                   cumulative=True)
        assert len(grid) == self.gridsize
        assert len(y) == self.gridsize
        # make sure y is monotonically increasing
        assert (np.diff(y) > 0).all()

    def test_kde_cummulative_2d(self):
        """Check error if args indicate bivariate KDE and cumulative."""
        with pytest.raises(TypeError):
            dist.kdeplot(x=self.x, y=self.y, cumulative=True)

    def test_kde_singular(self):
        """Check that kdeplot warns and skips on singular inputs."""
        with pytest.warns(UserWarning):
            ax = dist.kdeplot(np.ones(10))
        assert not ax.lines

        # line = ax.lines[0]
        # assert not line.get_xydata().size

        # with pytest.warns(UserWarning):
        #     ax = dist.kdeplot(np.ones(10) * np.nan)
        # line = ax.lines[1]
        # assert not line.get_xydata().size

    def test_data2_input_deprecation(self):
        """Using data2 kwarg should warn but still draw a bivariate plot."""
        with pytest.warns(FutureWarning):
            ax = dist.kdeplot(self.x, data2=self.y)
        assert len(ax.collections)

    @pytest.mark.skip
    def test_statsmodels_zero_bandwidth(self):
        """Test handling of 0 bandwidth data in statsmodels."""
        x = np.zeros(100)
        x[0] = 1

        try:

            smnp.kde.bandwidths.select_bandwidth(x, "scott", "gau")

        except RuntimeError:

            # Only execute the actual test in the except clause, this should
            # allot the test to pass on versions of statsmodels predating 0.11
            # and keep the test from failing in the future if statsmodels
            # reverts its behavior to avoid raising the error in the futures
            # Track at https://github.com/statsmodels/statsmodels/issues/5419

            with pytest.warns(UserWarning):
                ax = dist.kdeplot(x)
            line = ax.lines[0]
            assert not line.get_xydata().size

    @pytest.mark.parametrize("cumulative", [True, False])
    def test_kdeplot_with_nans(self, cumulative):

        if cumulative and _no_statsmodels:
            pytest.skip("no statsmodels")

        x_missing = np.append(self.x, [np.nan, np.nan])

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, cumulative=cumulative)
        dist.kdeplot(x=x_missing, cumulative=cumulative)

        line1, line2 = ax.lines
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

    def test_bivariate_kde_series(self):
        df = pd.DataFrame({'x': self.x, 'y': self.y})

        ax_series = dist.kdeplot(x=df.x, y=df.y)
        ax_values = dist.kdeplot(x=df.x.values, y=df.y.values)

        assert len(ax_series.collections) == len(ax_values.collections)
        assert (
            ax_series.collections[0].get_paths()
            == ax_values.collections[0].get_paths()
        )

    def test_bivariate_kde_colorbar(self):

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, y=self.y,
                     cbar=True, cbar_kws=dict(label="density"),
                     ax=ax)
        assert len(f.axes) == 2
        assert f.axes[1].get_ylabel() == "density"

    def test_legend(self):

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, y=self.y, label="test1")
        line = ax.lines[-1]
        assert line.get_label() == "test1"

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, y=self.y, shade=True, label="test2")
        fill = ax.collections[-1]
        assert fill.get_label() == "test2"

    def test_contour_color(self):

        rgb = (.1, .5, .7)
        f, ax = plt.subplots()

        dist.kdeplot(x=self.x, y=self.y, color=rgb)
        contour = ax.collections[-1]
        assert np.array_equal(contour.get_color()[0, :3], rgb)
        low = ax.collections[0].get_color().mean()
        high = ax.collections[-1].get_color().mean()
        assert low < high

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, y=self.y, shade=True, color=rgb)
        contour = ax.collections[-1]
        low = ax.collections[0].get_facecolor().mean()
        high = ax.collections[-1].get_facecolor().mean()
        assert low > high

        f, ax = plt.subplots()
        dist.kdeplot(x=self.x, y=self.y, shade=True, colors=[rgb])
        for level in ax.collections:
            level_rgb = tuple(level.get_facecolor().squeeze()[:3])
            assert level_rgb == rgb


class TestRugPlotter:

    @pytest.fixture
    def list_data(self):
        return np.random.randn(20).tolist()

    @pytest.fixture
    def array_data(self):
        return np.random.randn(20)

    @pytest.fixture
    def series_data(self):
        return pd.Series(np.random.randn(20))

    def test_rugplot(self, list_data, array_data, series_data):

        h = .1

        for x in [list_data, array_data, series_data]:

            f, ax = plt.subplots()
            rugplot(x=x, height=h)
            rug, = ax.collections
            segments = np.array(rug.get_segments())

            assert len(segments) == len(x)
            assert np.array_equal(segments[:, 0, 0], x)
            assert np.array_equal(segments[:, 1, 0], x)
            assert np.array_equal(segments[:, 0, 1], np.zeros_like(x))
            assert np.array_equal(segments[:, 1, 1], np.ones_like(x) * h)

            plt.close(f)

            f, ax = plt.subplots()
            rugplot(x=x, height=h, axis="y")
            rug, = ax.collections
            segments = np.array(rug.get_segments())

            assert len(segments) == len(x)
            assert np.array_equal(segments[:, 0, 1], x)
            assert np.array_equal(segments[:, 1, 1], x)
            assert np.array_equal(segments[:, 0, 0], np.zeros_like(x))
            assert np.array_equal(segments[:, 1, 0], np.ones_like(x) * h)

            plt.close(f)

        f, ax = plt.subplots()
        rugplot(x=x, axis="y")
        rugplot(x=x, vertical=True)
        c1, c2 = ax.collections
        assert np.array_equal(c1.get_segments(), c2.get_segments())
        plt.close(f)

        f, ax = plt.subplots()
        rugplot(x=x)
        rugplot(x=x, lw=2)
        rugplot(x=x, linewidth=3, alpha=.5)
        for c, lw in zip(ax.collections, [1, 2, 3]):
            assert np.squeeze(c.get_linewidth()).item() == lw
        assert c.get_alpha() == .5
        plt.close(f)

    def test_a_parameter_deprecation(self, series_data):

        with pytest.warns(FutureWarning):
            ax = rugplot(a=series_data)
        rug, = ax.collections
        segments = np.array(rug.get_segments())
        assert len(segments) == len(series_data)
