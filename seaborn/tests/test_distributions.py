import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import pytest
import nose.tools as nt
import numpy.testing as npt

from .. import distributions as dist

_no_statsmodels = not dist._has_statsmodels


class TestDistPlot(object):

    rs = np.random.RandomState(0)
    x = rs.randn(100)

    def test_hist_bins(self):

        try:
            fd_edges = np.histogram_bin_edges(self.x, "fd")
        except AttributeError:
            pytest.skip("Requires numpy >= 1.15")
        ax = dist.distplot(self.x)
        for edge, bar in zip(fd_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

        plt.close(ax.figure)
        n = 25
        n_edges = np.histogram_bin_edges(self.x, n)
        ax = dist.distplot(self.x, bins=n)
        for edge, bar in zip(n_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

    def test_elements(self):

        n = 10
        ax = dist.distplot(self.x, bins=n,
                           hist=True, kde=False, rug=False, fit=None)
        assert len(ax.patches) == 10
        assert len(ax.lines) == 0
        assert len(ax.collections) == 0

        plt.close(ax.figure)
        ax = dist.distplot(self.x,
                           hist=False, kde=True, rug=False, fit=None)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

        plt.close(ax.figure)
        ax = dist.distplot(self.x,
                           hist=False, kde=False, rug=True, fit=None)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0
        assert len(ax.collections) == 1

        plt.close(ax.figure)
        ax = dist.distplot(self.x,
                           hist=False, kde=False, rug=False, fit=stats.norm)
        assert len(ax.patches) == 0
        assert len(ax.lines) == 1
        assert len(ax.collections) == 0

    def test_distplot_with_nans(self):

        f, (ax1, ax2) = plt.subplots(2)
        x_null = np.append(self.x, [np.nan])

        dist.distplot(self.x, ax=ax1)
        dist.distplot(x_null, ax=ax2)

        line1 = ax1.lines[0]
        line2 = ax2.lines[0]
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

        for bar1, bar2 in zip(ax1.patches, ax2.patches):
            assert bar1.get_xy() == bar2.get_xy()
            assert bar1.get_height() == bar2.get_height()


class TestKDE(object):

    rs = np.random.RandomState(0)
    x = rs.randn(50)
    y = rs.randn(50)
    kernel = "gau"
    bw = "scott"
    gridsize = 128
    clip = (-np.inf, np.inf)
    cut = 3

    def test_scipy_univariate_kde(self):
        """Test the univariate KDE estimation with scipy."""
        grid, y = dist._scipy_univariate_kde(self.x, self.bw, self.gridsize,
                                             self.cut, self.clip)
        nt.assert_equal(len(grid), self.gridsize)
        nt.assert_equal(len(y), self.gridsize)
        for bw in ["silverman", .2]:
            dist._scipy_univariate_kde(self.x, bw, self.gridsize,
                                       self.cut, self.clip)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_univariate_kde(self):
        """Test the univariate KDE estimation with statsmodels."""
        grid, y = dist._statsmodels_univariate_kde(self.x, self.kernel,
                                                   self.bw, self.gridsize,
                                                   self.cut, self.clip)
        nt.assert_equal(len(grid), self.gridsize)
        nt.assert_equal(len(y), self.gridsize)
        for bw in ["silverman", .2]:
            dist._statsmodels_univariate_kde(self.x, self.kernel, bw,
                                             self.gridsize, self.cut,
                                             self.clip)

    def test_scipy_bivariate_kde(self):
        """Test the bivariate KDE estimation with scipy."""
        clip = [self.clip, self.clip]
        x, y, z = dist._scipy_bivariate_kde(self.x, self.y, self.bw,
                                            self.gridsize, self.cut, clip)
        nt.assert_equal(x.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(y.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(len(z), self.gridsize)

        # Test a specific bandwidth
        clip = [self.clip, self.clip]
        x, y, z = dist._scipy_bivariate_kde(self.x, self.y, 1,
                                            self.gridsize, self.cut, clip)

        # Test that we get an error with an invalid bandwidth
        with nt.assert_raises(ValueError):
            dist._scipy_bivariate_kde(self.x, self.y, (1, 2),
                                      self.gridsize, self.cut, clip)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_bivariate_kde(self):
        """Test the bivariate KDE estimation with statsmodels."""
        clip = [self.clip, self.clip]
        x, y, z = dist._statsmodels_bivariate_kde(self.x, self.y, self.bw,
                                                  self.gridsize,
                                                  self.cut, clip)
        nt.assert_equal(x.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(y.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(len(z), self.gridsize)

    @pytest.mark.skipif(_no_statsmodels, reason="no statsmodels")
    def test_statsmodels_kde_cumulative(self):
        """Test computation of cumulative KDE."""
        grid, y = dist._statsmodels_univariate_kde(self.x, self.kernel,
                                                   self.bw, self.gridsize,
                                                   self.cut, self.clip,
                                                   cumulative=True)
        nt.assert_equal(len(grid), self.gridsize)
        nt.assert_equal(len(y), self.gridsize)
        # make sure y is monotonically increasing
        npt.assert_((np.diff(y) > 0).all())

    def test_kde_cummulative_2d(self):
        """Check error if args indicate bivariate KDE and cumulative."""
        with npt.assert_raises(TypeError):
            dist.kdeplot(self.x, data2=self.y, cumulative=True)

    def test_kde_singular(self):

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(np.ones(10))
        line = ax.lines[0]
        assert not line.get_xydata().size

        with pytest.warns(UserWarning):
            ax = dist.kdeplot(np.ones(10) * np.nan)
        line = ax.lines[1]
        assert not line.get_xydata().size

    @pytest.mark.parametrize("cumulative", [True, False])
    def test_kdeplot_with_nans(self, cumulative):

        if cumulative and _no_statsmodels:
            pytest.skip("no statsmodels")

        x_missing = np.append(self.x, [np.nan, np.nan])

        f, ax = plt.subplots()
        dist.kdeplot(self.x, cumulative=cumulative)
        dist.kdeplot(x_missing, cumulative=cumulative)

        line1, line2 = ax.lines
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

    def test_bivariate_kde_series(self):
        df = pd.DataFrame({'x': self.x, 'y': self.y})

        ax_series = dist.kdeplot(df.x, df.y)
        ax_values = dist.kdeplot(df.x.values, df.y.values)

        nt.assert_equal(len(ax_series.collections),
                        len(ax_values.collections))
        nt.assert_equal(ax_series.collections[0].get_paths(),
                        ax_values.collections[0].get_paths())

    def test_bivariate_kde_colorbar(self):

        f, ax = plt.subplots()
        dist.kdeplot(self.x, self.y,
                     cbar=True, cbar_kws=dict(label="density"),
                     ax=ax)
        nt.assert_equal(len(f.axes), 2)
        nt.assert_equal(f.axes[1].get_ylabel(), "density")

    def test_legend(self):

        f, ax = plt.subplots()
        dist.kdeplot(self.x, self.y, label="test1")
        line = ax.lines[-1]
        assert line.get_label() == "test1"

        f, ax = plt.subplots()
        dist.kdeplot(self.x, self.y, shade=True, label="test2")
        fill = ax.collections[-1]
        assert fill.get_label() == "test2"

    def test_contour_color(self):

        rgb = (.1, .5, .7)
        f, ax = plt.subplots()

        dist.kdeplot(self.x, self.y, color=rgb)
        contour = ax.collections[-1]
        assert np.array_equal(contour.get_color()[0, :3], rgb)
        low = ax.collections[0].get_color().mean()
        high = ax.collections[-1].get_color().mean()
        assert low < high

        f, ax = plt.subplots()
        dist.kdeplot(self.x, self.y, shade=True, color=rgb)
        contour = ax.collections[-1]
        low = ax.collections[0].get_facecolor().mean()
        high = ax.collections[-1].get_facecolor().mean()
        assert low > high

        f, ax = plt.subplots()
        dist.kdeplot(self.x, self.y, shade=True, colors=[rgb])
        for level in ax.collections:
            level_rgb = tuple(level.get_facecolor().squeeze()[:3])
            assert level_rgb == rgb


class TestRugPlot(object):

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

        for data in [list_data, array_data, series_data]:

            f, ax = plt.subplots()
            dist.rugplot(data, h)
            rug, = ax.collections
            segments = np.array(rug.get_segments())

            assert len(segments) == len(data)
            assert np.array_equal(segments[:, 0, 0], data)
            assert np.array_equal(segments[:, 1, 0], data)
            assert np.array_equal(segments[:, 0, 1], np.zeros_like(data))
            assert np.array_equal(segments[:, 1, 1], np.ones_like(data) * h)

            plt.close(f)

            f, ax = plt.subplots()
            dist.rugplot(data, h, axis="y")
            rug, = ax.collections
            segments = np.array(rug.get_segments())

            assert len(segments) == len(data)
            assert np.array_equal(segments[:, 0, 1], data)
            assert np.array_equal(segments[:, 1, 1], data)
            assert np.array_equal(segments[:, 0, 0], np.zeros_like(data))
            assert np.array_equal(segments[:, 1, 0], np.ones_like(data) * h)

            plt.close(f)

        f, ax = plt.subplots()
        dist.rugplot(data, axis="y")
        dist.rugplot(data, vertical=True)
        c1, c2 = ax.collections
        assert np.array_equal(c1.get_segments(), c2.get_segments())
        plt.close(f)

        f, ax = plt.subplots()
        dist.rugplot(data)
        dist.rugplot(data, lw=2)
        dist.rugplot(data, linewidth=3, alpha=.5)
        for c, lw in zip(ax.collections, [1, 2, 3]):
            assert np.squeeze(c.get_linewidth()).item() == lw
        assert c.get_alpha() == .5
        plt.close(f)
