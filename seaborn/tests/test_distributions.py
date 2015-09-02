import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import nose.tools as nt
import numpy.testing as npt
from numpy.testing.decorators import skipif

from . import PlotTestCase
from .. import distributions as dist

try:
    import statsmodels.nonparametric.api
    assert statsmodels.nonparametric.api
    _no_statsmodels = False
except ImportError:
    _no_statsmodels = True


class TestKDE(PlotTestCase):

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

    @skipif(_no_statsmodels)
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

    @skipif(_no_statsmodels)
    def test_statsmodels_bivariate_kde(self):
        """Test the bivariate KDE estimation with statsmodels."""
        clip = [self.clip, self.clip]
        x, y, z = dist._statsmodels_bivariate_kde(self.x, self.y, self.bw,
                                                  self.gridsize,
                                                  self.cut, clip)
        nt.assert_equal(x.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(y.shape, (self.gridsize, self.gridsize))
        nt.assert_equal(len(z), self.gridsize)

    @skipif(_no_statsmodels)
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

    def test_bivariate_kde_series(self):
        df = pd.DataFrame({'x': self.x, 'y': self.y})

        ax_series = dist.kdeplot(df.x, df.y)
        ax_values = dist.kdeplot(df.x.values, df.y.values)

        nt.assert_equal(len(ax_series.collections),
                        len(ax_values.collections))
        nt.assert_equal(ax_series.collections[0].get_paths(),
                        ax_values.collections[0].get_paths())


class TestJointPlot(PlotTestCase):

    rs = np.random.RandomState(sum(map(ord, "jointplot")))
    x = rs.randn(100)
    y = rs.randn(100)
    data = pd.DataFrame(dict(x=x, y=y))

    def test_scatter(self):

        g = dist.jointplot("x", "y", self.data)
        nt.assert_equal(len(g.ax_joint.collections), 1)

        x, y = g.ax_joint.collections[0].get_offsets().T
        npt.assert_array_equal(self.x, x)
        npt.assert_array_equal(self.y, y)

        x_bins = dist._freedman_diaconis_bins(self.x)
        nt.assert_equal(len(g.ax_marg_x.patches), x_bins)

        y_bins = dist._freedman_diaconis_bins(self.y)
        nt.assert_equal(len(g.ax_marg_y.patches), y_bins)

    def test_reg(self):

        g = dist.jointplot("x", "y", self.data, kind="reg")
        nt.assert_equal(len(g.ax_joint.collections), 2)

        x, y = g.ax_joint.collections[0].get_offsets().T
        npt.assert_array_equal(self.x, x)
        npt.assert_array_equal(self.y, y)

        x_bins = dist._freedman_diaconis_bins(self.x)
        nt.assert_equal(len(g.ax_marg_x.patches), x_bins)

        y_bins = dist._freedman_diaconis_bins(self.y)
        nt.assert_equal(len(g.ax_marg_y.patches), y_bins)

        nt.assert_equal(len(g.ax_joint.lines), 1)
        nt.assert_equal(len(g.ax_marg_x.lines), 1)
        nt.assert_equal(len(g.ax_marg_y.lines), 1)

    def test_resid(self):

        g = dist.jointplot("x", "y", self.data, kind="resid")
        nt.assert_equal(len(g.ax_joint.collections), 1)
        nt.assert_equal(len(g.ax_joint.lines), 1)
        nt.assert_equal(len(g.ax_marg_x.lines), 0)
        nt.assert_equal(len(g.ax_marg_y.lines), 1)

    def test_hex(self):

        g = dist.jointplot("x", "y", self.data, kind="hex")
        nt.assert_equal(len(g.ax_joint.collections), 1)

        x_bins = dist._freedman_diaconis_bins(self.x)
        nt.assert_equal(len(g.ax_marg_x.patches), x_bins)

        y_bins = dist._freedman_diaconis_bins(self.y)
        nt.assert_equal(len(g.ax_marg_y.patches), y_bins)

    def test_kde(self):

        g = dist.jointplot("x", "y", self.data, kind="kde")

        nt.assert_true(len(g.ax_joint.collections) > 0)
        nt.assert_equal(len(g.ax_marg_x.collections), 1)
        nt.assert_equal(len(g.ax_marg_y.collections), 1)

        nt.assert_equal(len(g.ax_marg_x.lines), 1)
        nt.assert_equal(len(g.ax_marg_y.lines), 1)

    def test_color(self):

        g = dist.jointplot("x", "y", self.data, color="purple")

        purple = mpl.colors.colorConverter.to_rgb("purple")
        scatter_color = g.ax_joint.collections[0].get_facecolor()[0, :3]
        nt.assert_equal(tuple(scatter_color), purple)

        hist_color = g.ax_marg_x.patches[0].get_facecolor()[:3]
        nt.assert_equal(hist_color, purple)

    def test_annotation(self):

        g = dist.jointplot("x", "y", self.data)
        nt.assert_equal(len(g.ax_joint.legend_.get_texts()), 1)

        g = dist.jointplot("x", "y", self.data, stat_func=None)
        nt.assert_is(g.ax_joint.legend_, None)

    def test_hex_customise(self):

        # test that default gridsize can be overridden
        g = dist.jointplot("x", "y", self.data, kind="hex",
                           joint_kws=dict(gridsize=5))
        nt.assert_equal(len(g.ax_joint.collections), 1)
        a = g.ax_joint.collections[0].get_array()
        nt.assert_equal(28, a.shape[0])  # 28 hexagons expected for gridsize 5

    def test_bad_kind(self):

        with nt.assert_raises(ValueError):
            dist.jointplot("x", "y", self.data, kind="not_a_kind")
