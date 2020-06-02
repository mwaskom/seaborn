import itertools
from distutils.version import LooseVersion

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import scipy
from scipy import stats, integrate

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .. import distributions as dist
from ..palettes import (
    color_palette,
    light_palette,
)
from .._core import (
    categorical_order,
)
from .._statistics import (
    KDE
)
from ..distributions import (
    _KDEPlotter,
    rugplot,
    kdeplot,
)


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


class TestKDEPlotUnivariate:

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

        v0 = ax.collections[0].get_paths()[0].vertices
        v1 = ax.collections[1].get_paths()[0].vertices[:, [1, 0]]

        assert_array_equal(v0, v1)

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

    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_hue_colors(self, long_df, multiple):

        ax = kdeplot(
            data=long_df, x="x", hue="a",
            multiple=multiple,
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
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="stack", fill=False,
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
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="fill", fill=False,
            legend=False, ax=ax2,
        )

        layered = np.stack([l.get_ydata() for l in ax1.lines])
        filled = np.stack([l.get_ydata() for l in ax2.lines])

        assert_array_almost_equal(
            (layered / layered.sum(axis=0)).cumsum(axis=0),
            filled,
        )

    @pytest.mark.parametrize("multiple", ["stack", "fill"])
    def test_fill_default(self, long_df, multiple):

        ax = kdeplot(
            data=long_df, x="x", hue="a", multiple=multiple, fill=None
        )

        assert len(ax.collections) > 0

    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_fill_nondefault(self, long_df, multiple):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kws = dict(data=long_df, x="x", hue="a")
        kdeplot(**kws, multiple=multiple, fill=False, ax=ax1)
        kdeplot(**kws, multiple=multiple, fill=True, ax=ax2)

        assert len(ax1.collections) == 0
        assert len(ax2.collections) > 0

    def test_color_cycle_interaction(self, flat_series):

        color = (.2, 1, .6)
        C0, C1 = to_rgb("C0"), to_rgb("C1")

        f, ax = plt.subplots()
        kdeplot(flat_series)
        kdeplot(flat_series)
        assert to_rgb(ax.lines[0].get_color()) == C0
        assert to_rgb(ax.lines[1].get_color()) == C1
        plt.close(f)

        f, ax = plt.subplots()
        kdeplot(flat_series, color=color)
        kdeplot(flat_series)
        assert ax.lines[0].get_color() == color
        assert to_rgb(ax.lines[1].get_color()) == C0
        plt.close(f)

        f, ax = plt.subplots()
        kdeplot(flat_series, fill=True)
        kdeplot(flat_series, fill=True)
        assert (
            to_rgba(ax.collections[0].get_facecolor().squeeze())
            == to_rgba(C0, .25)
        )
        assert (
            to_rgba(ax.collections[1].get_facecolor().squeeze())
            == to_rgba(C1, .25)
        )
        plt.close(f)

    def test_color(self, long_df):

        color = (.2, 1, .6)
        alpha = .5

        f, ax = plt.subplots()

        kdeplot(long_df["x"], fill=True, color=color)
        c = ax.collections[-1]
        assert (
            to_rgba(c.get_facecolor().squeeze())
            == to_rgba(color, .25)
        )

        kdeplot(long_df["x"], fill=True, color=color, alpha=alpha)
        c = ax.collections[-1]
        assert (
            to_rgba(c.get_facecolor().squeeze())
            == to_rgba(color, alpha)
        )

    def test_multiple_input_check(self, long_df):

        with pytest.raises(ValueError, match="multiple must be"):
            kdeplot(data=long_df, x="x", hue="a", multiple="bad_input")

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
            data=long_df, x="x", hue="c", common_norm=True, cut=10, ax=ax1
        )
        kdeplot(
            data=long_df, x="x", hue="c", common_norm=False, cut=10, ax=ax2
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
        assert ax1.collections[0].sticky_edges.y[:] == [0, np.inf]

        kdeplot(
            data=long_df, x="x", hue="a", multiple="fill", fill=True, ax=ax2
        )
        assert ax2.collections[0].sticky_edges.y[:] == [0, 1]

    def test_line_kws(self, flat_array):

        lw = 3
        color = (.2, .5, .8)
        ax = kdeplot(x=flat_array, linewidth=lw, color=color)
        line, = ax.lines
        assert line.get_linewidth() == lw
        assert line.get_color() == color

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


class TestKDEPlotBivariate:

    def test_long_vectors(self, long_df):

        ax1 = kdeplot(data=long_df, x="x", y="y")

        x = long_df["x"]
        x_values = [x, np.asarray(x), x.tolist()]

        y = long_df["y"]
        y_values = [y, np.asarray(y), y.tolist()]

        for x, y in zip(x_values, y_values):
            f, ax2 = plt.subplots()
            kdeplot(x=x, y=y, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert_array_equal(c1.get_offsets(), c2.get_offsets())

    def test_singular_data(self):

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=np.ones(10), y=np.arange(10))
        assert not ax.lines

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=[5], y=[6])
        assert not ax.lines

    def test_fill_artists(self, long_df):

        for fill in [True, False]:
            f, ax = plt.subplots()
            kdeplot(data=long_df, x="x", y="y", hue="c", fill=fill)
            for c in ax.collections:
                if fill:
                    assert isinstance(c, mpl.collections.PathCollection)
                else:
                    assert isinstance(c, mpl.collections.LineCollection)

    def test_common_norm(self, rng):

        hue = np.repeat(["a", "a", "a", "b"], 40)
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], len(hue)).T
        x[hue == "a"] -= 2
        x[hue == "b"] += 2

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, hue=hue, common_norm=True, ax=ax1)
        kdeplot(x=x, y=y, hue=hue, common_norm=False, ax=ax2)

        n_seg_1 = sum([len(c.get_segments()) > 0 for c in ax1.collections])
        n_seg_2 = sum([len(c.get_segments()) > 0 for c in ax2.collections])
        assert n_seg_2 > n_seg_1

    def test_log_scale(self, rng):

        x = rng.lognormal(0, 1, 100)
        y = rng.uniform(0, 1, 100)

        levels = .2, .5, 1

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, log_scale=True, levels=levels, ax=ax1)
        assert ax1.get_xscale() == "log"
        assert ax1.get_yscale() == "log"

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, log_scale=(10, False), levels=levels, ax=ax1)
        assert ax1.get_xscale() == "log"
        assert ax1.get_yscale() == "linear"

        p = _KDEPlotter()
        kde = KDE()
        density, (xx, yy) = kde(np.log10(x), y)
        levels = p._find_contour_levels(density, levels)
        ax2.contour(10 ** xx, yy, density, levels=levels)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert_array_equal(c1.get_segments(), c2.get_segments())

    def test_bandwiddth(self, rng):

        n = 100
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], n).T

        f, (ax1, ax2) = plt.subplots(ncols=2)

        kdeplot(x=x, y=y, ax=ax1)
        kdeplot(x=x, y=y, bw_adjust=2, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            seg1, seg2 = c1.get_segments(), c2.get_segments()
            if seg1 + seg2:
                x1 = seg1[0][:, 0]
                x2 = seg2[0][:, 0]
                assert np.abs(x2).max() > np.abs(x1).max()

    @pytest.mark.skipif(
        LooseVersion(scipy.__version__) < "1.2.0",
        reason="Weights require scipy >= 1.2.0"
    )
    def test_weights(self, rng):

        n = 100
        x, y = rng.multivariate_normal([1, 3], [(.2, .5), (.5, 2)], n).T
        hue = np.repeat([0, 1], n // 2)
        weights = rng.uniform(0, 1, n)

        f, (ax1, ax2) = plt.subplots(ncols=2)
        kdeplot(x=x, y=y, hue=hue, ax=ax1)
        kdeplot(x=x, y=y, hue=hue, weights=weights, ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            if c1.get_segments():
                assert not np.array_equal(c1.get_segments(), c2.get_segments())

    def test_hue_ignores_cmap(self, long_df):

        with pytest.warns(UserWarning, match="cmap parameter ignored"):
            ax = kdeplot(data=long_df, x="x", y="y", hue="c", cmap="viridis")

        color = tuple(ax.collections[0].get_color().squeeze())
        assert color == mpl.colors.colorConverter.to_rgba("C0")

    def test_contour_line_colors(self, long_df):

        color = (.2, .9, .8, 1)
        ax = kdeplot(data=long_df, x="x", y="y", color=color)

        for c in ax.collections:
            assert tuple(c.get_color().squeeze()) == color

    def test_contour_fill_colors(self, long_df):

        n = 6
        color = (.2, .9, .8, 1)
        ax = kdeplot(
            data=long_df, x="x", y="y", fill=True, color=color, levels=n,
        )

        cmap = light_palette(color, reverse=True, as_cmap=True)
        lut = cmap(np.linspace(0, 1, 256))
        for c in ax.collections:
            color = c.get_facecolor().squeeze()
            assert color in lut

    def test_colorbar(self, long_df):

        ax = kdeplot(data=long_df, x="x", y="y", fill=True, cbar=True)
        assert len(ax.figure.axes) == 2

    def test_levels_and_thresh(self, long_df):

        f, (ax1, ax2) = plt.subplots(ncols=2)

        n = 8
        thresh = .1
        plot_kws = dict(data=long_df, x="x", y="y")
        kdeplot(**plot_kws, levels=n, thresh=thresh, ax=ax1)
        kdeplot(**plot_kws, levels=np.linspace(thresh, 1, n), ax=ax2)

        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert_array_equal(c1.get_segments(), c2.get_segments())

        with pytest.raises(ValueError):
            kdeplot(**plot_kws, levels=[0, 1, 2])

    def test_contour_levels(self, rng):

        x = rng.uniform(0, 1, 100000)
        isoprop = np.linspace(.1, 1, 6)

        levels = _KDEPlotter()._find_contour_levels(x, isoprop)
        for h, p in zip(levels, isoprop):
            assert (x[x <= h].sum() / x.sum()) == pytest.approx(p, abs=1e-4)

    def test_input_checking(self, long_df):

        with pytest.raises(TypeError, match="kdeplot requires a numeric 'x'"):
            kdeplot(data=long_df, x="a", y="y")
