import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import nose.tools as nt
import numpy.testing as npt
from numpy.testing.decorators import skipif

from .. import distributions as dist

try:
    import statsmodels
    assert statsmodels
    _no_statsmodels = False
except ImportError:
    _no_statsmodels = True


class TestBoxReshaping(object):
    """Tests for function that preps boxplot/violinplot data."""
    n_total = 60
    rs = np.random.RandomState(0)
    x = rs.randn(n_total / 3, 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    g = pd.Series(rs.choice(list("abc"), n_total), name="small")
    df = pd.DataFrame(dict(y=y, g=g))

    def test_1d_values(self):
        """Test boxplot prep for 1D data in various forms."""
        x_1d_array = self.x.ravel()
        vals_1d_array = dist._box_reshape(x_1d_array, None, None, None)[0]
        nt.assert_equal(len(vals_1d_array), 1)
        nt.assert_equal(len(vals_1d_array[0]), self.n_total)

        x_1d_list = x_1d_array.tolist()
        vals_1d_list = dist._box_reshape(x_1d_list, None, None, None)[0]
        nt.assert_equal(len(vals_1d_list), 1)
        nt.assert_equal(len(vals_1d_list[0]), self.n_total)

    def test_list_of_array_values(self):
        """Test boxplot prep for 2D data that is a list of arrays."""
        x_list = self.x.T.tolist()
        vals = dist._box_reshape(x_list, None, None, None)[0]
        nt.assert_equal(len(vals), 3)
        lengths = [len(v_i) for v_i in vals]
        nt.assert_equal(lengths, [self.n_total / 3] * 3)

    def test_array_values(self):
        """Test boxplot prep for a basic array input."""
        vals = dist._box_reshape(self.x, None, None, None)[0]
        nt.assert_equal(np.shape(vals), (3, self.n_total / 3))
        npt.assert_array_equal(vals, self.x.T)

    def test_dataframe_values(self):
        """Test boxplot prep for a DataFrame input."""
        vals = dist._box_reshape(self.x_df, None, None, None)[0]
        nt.assert_equal(np.shape(vals), (3, self.n_total / 3))
        npt.assert_array_equal(vals, self.x.T)

    def test_series_groupby(self):
        """Test boxplot groupby using a series of data labels."""
        vals = dist._box_reshape(self.df.y, self.df.g, None, None)[0]
        nt.assert_equal(len(vals), 3)
        want_lengths = pd.value_counts(self.df.g)[["a", "b", "c"]]
        got_lengths = [len(a) for a in vals]
        npt.assert_array_equal(want_lengths, got_lengths)

    def test_series_groupby_order(self):
        """Test a series-based groupby with a forced ordering."""
        order = ["c", "a", "b"]
        vals = dist._box_reshape(self.df.y, self.df.g, None, order)[0]
        want_lengths = pd.value_counts(self.df.g)[order]
        got_lengths = [len(a) for a in vals]
        npt.assert_array_equal(want_lengths, got_lengths)

    def test_function_groupby(self):
        """Test boxplot groupby using a grouping function."""
        grouper = lambda ix: self.df.y.ix[ix] > 0
        vals = dist._box_reshape(self.df.y, grouper, None, None)[0]
        nt.assert_equal(len(vals), 2)
        low, high = vals
        nt.assert_true(low.max() <= 0)
        nt.assert_true(high.min() > 0)

    def test_dict_groupby(self):
        """Test boxplot groupby using a dictionary."""
        grouper = {i: "A" if i % 2 else "B" for i in self.df.y.index}
        vals = dist._box_reshape(self.df.y, grouper, None, None)[0]
        nt.assert_equal(len(vals), 2)
        a, b = vals
        npt.assert_array_equal(self.df.y.iloc[1::2], a)
        npt.assert_array_equal(self.df.y.iloc[::2], b)

    def test_1d_labels(self):
        """Test boxplot labels for 1D data."""
        x_1d_array = self.x.ravel()
        vals, xlabel, ylabel, names = dist._box_reshape(x_1d_array,
                                                        None, None, None)
        nt.assert_is(xlabel, None)
        nt.assert_is(ylabel, None)
        nt.assert_equal(names, [1])

        vals, xlabel, ylabel, names = dist._box_reshape(x_1d_array,
                                                        None, ["A"], None)

    def test_array_labels(self):
        """Test boxplot labels for a basic array."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.x,
                                                        None, None, None)
        nt.assert_is(xlabel, None)
        nt.assert_is(ylabel, None)
        nt.assert_equal(names, list(range(1, 4)))

        want_names = list("ABC")
        vals, xlabel, ylabel, names = dist._box_reshape(self.x,
                                                        None, want_names, None)
        nt.assert_equal(names, want_names)

    def test_dataframe_labels(self):
        """Test boxplot labels with DataFrame."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df,
                                                        None, None, None)
        nt.assert_equal(xlabel, self.x_df.columns.name)
        nt.assert_equal(ylabel, None)
        npt.assert_array_equal(names, self.x_df.columns)

    def test_ordered_dataframe_labels(self):
        """Test boxplot labels with DataFrame and specified order."""
        order = list("ZYX")
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df,
                                                        None, None, order)
        nt.assert_equal(xlabel, self.x_df.columns.name)
        npt.assert_array_equal(names, order)

    def test_groupby_labels(self):
        """Test labels with groupby vals."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        None, None)
        nt.assert_equal(xlabel, self.g.name)
        nt.assert_equal(ylabel, self.y.name)
        npt.assert_array_equal(names, sorted(self.g.unique()))

    def test_ordered_groupby_labels(self):
        """Test labels with groupby vals and specified order."""
        order = list("BAC")
        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        order, None)
        nt.assert_equal(xlabel, self.g.name)
        nt.assert_equal(ylabel, self.y.name)
        npt.assert_array_equal(names, order)

    def test_pandas_names_override(self):
        """Test that names can override those inferred from Pandas objects."""
        want_names = ["ex", "why", "zee"]
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df, None,
                                                        want_names, None)
        nt.assert_equal(names, want_names)

        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        want_names, None)
        nt.assert_equal(names, want_names)

    def test_bad_order_length(self):
        """Test for error when order and names lengths mismatch."""
        with nt.assert_raises(ValueError):
            dist._box_reshape(self.x_df, None, range(5), range(6))

    def test_bad_order_type(self):
        """Test for error when trying to order with a vanilla array."""
        with nt.assert_raises(ValueError):
            dist._box_reshape(self.x, None, None, range(5))


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


class TestJointPlot(object):

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

        plt.close("all")

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

        plt.close("all")

    def test_resid(self):

        g = dist.jointplot("x", "y", self.data, kind="resid")
        nt.assert_equal(len(g.ax_joint.collections), 1)
        nt.assert_equal(len(g.ax_joint.lines), 1)
        nt.assert_equal(len(g.ax_marg_x.lines), 0)
        nt.assert_equal(len(g.ax_marg_y.lines), 1)

        plt.close("all")

    def test_hex(self):

        g = dist.jointplot("x", "y", self.data, kind="hex")
        nt.assert_equal(len(g.ax_joint.collections), 1)

        x_bins = dist._freedman_diaconis_bins(self.x)
        nt.assert_equal(len(g.ax_marg_x.patches), x_bins)

        y_bins = dist._freedman_diaconis_bins(self.y)
        nt.assert_equal(len(g.ax_marg_y.patches), y_bins)

        plt.close("all")

    def test_kde(self):

        g = dist.jointplot("x", "y", self.data, kind="kde")

        nt.assert_true(len(g.ax_joint.collections) > 0)
        nt.assert_equal(len(g.ax_marg_x.collections), 1)
        nt.assert_equal(len(g.ax_marg_y.collections), 1)

        nt.assert_equal(len(g.ax_marg_x.lines), 1)
        nt.assert_equal(len(g.ax_marg_y.lines), 1)

        plt.close("all")

    def test_color(self):

        g = dist.jointplot("x", "y", self.data, color="purple")

        purple = mpl.colors.colorConverter.to_rgb("purple")
        scatter_color = g.ax_joint.collections[0].get_facecolor()[0, :3]
        nt.assert_equal(tuple(scatter_color), purple)

        hist_color = g.ax_marg_x.patches[0].get_facecolor()[:3]
        nt.assert_equal(hist_color, purple)

        plt.close("all")

    def test_annotation(self):

        g = dist.jointplot("x", "y", self.data)
        nt.assert_equal(len(g.ax_joint.legend_.get_texts()), 1)

        g = dist.jointplot("x", "y", self.data, stat_func=None)
        nt.assert_is(g.ax_joint.legend_, None)

        plt.close("all")
