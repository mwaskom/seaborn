import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
pandas_has_categoricals = LooseVersion(pd.__version__) >= "0.15"

import nose.tools as nt
import numpy.testing as npt
from numpy.testing.decorators import skipif

from .. import categorical as cat
from .. import palettes


class CategoricalFixture(object):
    """Test boxplot (also base class for things like violinplots)."""
    rs = np.random.RandomState(30)
    n_total = 60
    x = rs.randn(n_total / 3, 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    g = pd.Series(np.repeat(list("abc"), n_total / 3), name="small")
    h = pd.Series(np.tile(list("mn"), n_total / 2), name="medium")
    u = pd.Series(np.tile(list("jkh"), n_total / 3))
    df = pd.DataFrame(dict(y=y, g=g, h=h, u=u))
    x_df["W"] = g


class TestCategoricalPlotter(CategoricalFixture):

    def test_wide_df_data(self):

        p = cat._CategoricalPlotter()

        # Test basic wide DataFrame
        p.establish_variables(data=self.x_df)

        # Check data attribute
        for x, y, in zip(p.plot_data, self.x_df[["X", "Y", "Z"]].values.T):
            npt.assert_array_equal(x, y)

        # Check semantic attributes
        nt.assert_equal(p.orient, "v")
        nt.assert_is(p.plot_hues, None)
        nt.assert_is(p.group_label, "big")
        nt.assert_is(p.value_label, None)

        # Test wide dataframe with forced horizontal orientation
        p.establish_variables(data=self.x_df, orient="horiz")
        nt.assert_equal(p.orient, "h")

        # Text exception by trying to hue-group with a wide dataframe
        with nt.assert_raises(ValueError):
            p.establish_variables(hue="d", data=self.x_df)

    def test_1d_input_data(self):

        p = cat._CategoricalPlotter()

        # Test basic vector data
        x_1d_array = self.x.ravel()
        p.establish_variables(data=x_1d_array)
        nt.assert_equal(len(p.plot_data), 1)
        nt.assert_equal(len(p.plot_data[0]), self.n_total)
        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

        # Test basic vector data in list form
        x_1d_list = x_1d_array.tolist()
        p.establish_variables(data=x_1d_list)
        nt.assert_equal(len(p.plot_data), 1)
        nt.assert_equal(len(p.plot_data[0]), self.n_total)
        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

        # Test an object array that looks 1D but isn't
        x_notreally_1d = np.array([self.x.ravel(),
                                   self.x.ravel()[:self.n_total / 2]])
        p.establish_variables(data=x_notreally_1d)
        nt.assert_equal(len(p.plot_data), 2)
        nt.assert_equal(len(p.plot_data[0]), self.n_total)
        nt.assert_equal(len(p.plot_data[1]), self.n_total / 2)
        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_2d_input_data(self):

        p = cat._CategoricalPlotter()

        x = self.x[:, 0]

        # Test vector data that looks 2D but doesn't really have columns
        p.establish_variables(data=x[:, np.newaxis])
        nt.assert_equal(len(p.plot_data), 1)
        nt.assert_equal(len(p.plot_data[0]), self.x.shape[0])
        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

        # Test vector data that looks 2D but doesn't really have rows
        p.establish_variables(data=x[np.newaxis, :])
        nt.assert_equal(len(p.plot_data), 1)
        nt.assert_equal(len(p.plot_data[0]), self.x.shape[0])
        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_3d_input_data(self):

        p = cat._CategoricalPlotter()

        # Test that passing actually 3D data raises
        x = np.zeros((5, 5, 5))
        with nt.assert_raises(ValueError):
            p.establish_variables(data=x)

    def test_list_of_array_input_data(self):

        p = cat._CategoricalPlotter()

        # Test 2D input in list form
        x_list = self.x.T.tolist()
        p.establish_variables(data=x_list)
        nt.assert_equal(len(p.plot_data), 3)

        lengths = [len(v_i) for v_i in p.plot_data]
        nt.assert_equal(lengths, [self.n_total / 3] * 3)

        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_wide_array_input_data(self):

        p = cat._CategoricalPlotter()

        # Test 2D input in array form
        p.establish_variables(data=self.x)
        nt.assert_equal(np.shape(p.plot_data), (3, self.n_total / 3))
        npt.assert_array_equal(p.plot_data, self.x.T)

        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_single_long_direct_inputs(self):

        p = cat._CategoricalPlotter()

        # Test passing a series to the x variable
        p.establish_variables(x=self.y)
        npt.assert_equal(p.plot_data, [self.y])
        nt.assert_equal(p.orient, "h")
        nt.assert_equal(p.value_label, "y_data")
        nt.assert_is(p.group_label, None)

        # Test passing a series to the y variable
        p.establish_variables(y=self.y)
        npt.assert_equal(p.plot_data, [self.y])
        nt.assert_equal(p.orient, "v")
        nt.assert_equal(p.value_label, "y_data")
        nt.assert_is(p.group_label, None)

        # Test passing an array to the y variable
        p.establish_variables(y=self.y.values)
        npt.assert_equal(p.plot_data, [self.y])
        nt.assert_equal(p.orient, "v")
        nt.assert_is(p.value_label, None)
        nt.assert_is(p.group_label, None)

    def test_single_long_indirect_inputs(self):

        p = cat._CategoricalPlotter()

        # Test referencing a DataFrame series in the x variable
        p.establish_variables(x="y", data=self.df)
        npt.assert_equal(p.plot_data, [self.y])
        nt.assert_equal(p.orient, "h")
        nt.assert_equal(p.value_label, "y")
        nt.assert_is(p.group_label, None)

        # Test referencing a DataFrame series in the y variable
        p.establish_variables(y="y", data=self.df)
        npt.assert_equal(p.plot_data, [self.y])
        nt.assert_equal(p.orient, "v")
        nt.assert_equal(p.value_label, "y")
        nt.assert_is(p.group_label, None)

    def test_longform_groupby(self):

        p = cat._CategoricalPlotter()

        # Test a vertically oriented grouped and nested plot
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(len(p.plot_data), 3)
        nt.assert_equal(len(p.plot_hues), 3)
        nt.assert_equal(p.orient, "v")
        nt.assert_equal(p.value_label, "y")
        nt.assert_equal(p.group_label, "g")
        nt.assert_equal(p.hue_title, "h")

        for group, vals in zip(["a", "b", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        for group, hues in zip(["a", "b", "c"], p.plot_hues):
            npt.assert_array_equal(hues, self.h[self.g == group])

        # Test a grouped and nested plot with direct array value data
        p.establish_variables("g", self.y.values, "h", self.df)
        nt.assert_is(p.value_label, None)
        nt.assert_equal(p.group_label, "g")

        for group, vals in zip(["a", "b", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        # Test a grouped and nested plot with direct array hue data
        p.establish_variables("g", "y", self.h.values, self.df)

        for group, hues in zip(["a", "b", "c"], p.plot_hues):
            npt.assert_array_equal(hues, self.h[self.g == group])

        # Test categorical grouping data
        if pandas_has_categoricals:
            df = self.df.copy()
            df.g = df.g.astype("category")

            # Test that horizontal orientation is automatically detected
            p.establish_variables("y", "g", "h", data=df)
            nt.assert_equal(len(p.plot_data), 3)
            nt.assert_equal(len(p.plot_hues), 3)
            nt.assert_equal(p.orient, "h")
            nt.assert_equal(p.value_label, "y")
            nt.assert_equal(p.group_label, "g")
            nt.assert_equal(p.hue_title, "h")

            for group, vals in zip(["a", "b", "c"], p.plot_data):
                npt.assert_array_equal(vals, self.y[self.g == group])

            for group, hues in zip(["a", "b", "c"], p.plot_hues):
                npt.assert_array_equal(hues, self.h[self.g == group])

    def test_order(self):

        p = cat._CategoricalPlotter()

        # Test inferred order from a wide dataframe input
        p.establish_variables(data=self.x_df)
        nt.assert_equal(p.group_names, ["X", "Y", "Z"])

        # Test specified order with a wide dataframe input
        p.establish_variables(data=self.x_df, order=["Y", "Z", "X"])
        nt.assert_equal(p.group_names, ["Y", "Z", "X"])

        for group, vals in zip(["Y", "Z", "X"], p.plot_data):
            npt.assert_array_equal(vals, self.x_df[group])

        with nt.assert_raises(ValueError):
            p.establish_variables(data=self.x, order=[1, 2, 0])

        # Test inferred order from a grouped longform input
        p.establish_variables("g", "y", data=self.df)
        nt.assert_equal(p.group_names, ["a", "b", "c"])

        # Test specified order from a grouped longform input
        p.establish_variables("g", "y", data=self.df, order=["b", "a", "c"])
        nt.assert_equal(p.group_names, ["b", "a", "c"])

        for group, vals in zip(["b", "a", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        # Test inferred order from a grouped input with categorical groups
        if pandas_has_categoricals:
            df = self.df.copy()
            df.g = df.g.astype("category")
            df.g = df.g.cat.reorder_categories(["c", "b", "a"])
            p.establish_variables("g", "y", data=df)
            nt.assert_equal(p.group_names, ["c", "b", "a"])

            for group, vals in zip(["c", "b", "a"], p.plot_data):
                npt.assert_array_equal(vals, self.y[self.g == group])

    def test_hue_order(self):

        p = cat._CategoricalPlotter()

        # Test inferred hue order
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(p.hue_names, ["m", "n"])

        # Test specified hue order
        p.establish_variables("g", "y", "h", data=self.df,
                              hue_order=["n", "m"])
        nt.assert_equal(p.hue_names, ["n", "m"])

        # Test inferred hue order from a categorical hue input
        if pandas_has_categoricals:
            df = self.df.copy()
            df.h = df.h.astype("category")
            df.h = df.h.cat.reorder_categories(["n", "m"])
            p.establish_variables("g", "y", "h", data=df)
            nt.assert_equal(p.hue_names, ["n", "m"])

    def test_plot_units(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_is(p.plot_units, None)

        p.establish_variables("g", "y", "h", data=self.df, units="u")
        for group, units in zip(["a", "b", "c"], p.plot_units):
            npt.assert_array_equal(units, self.u[self.g == group])

    def test_orient_inference(self):

        p = cat._CategoricalPlotter()

        cat_series = pd.Series(["a", "b", "c"] * 10)
        num_series = pd.Series(self.rs.randn(30))

        x, y = cat_series, num_series

        nt.assert_equal(p.infer_orient(x, y, "horiz"), "h")
        nt.assert_equal(p.infer_orient(x, y, "vert"), "v")
        nt.assert_equal(p.infer_orient(x, None), "h")
        nt.assert_equal(p.infer_orient(None, y), "v")
        nt.assert_equal(p.infer_orient(x, y), "v")

        if pandas_has_categoricals:
            cat_series = cat_series.astype("category")
            y, x = cat_series, num_series
            nt.assert_equal(p.infer_orient(x, y), "h")

    def test_default_palettes(self):

        p = cat._CategoricalPlotter()

        # Test palette mapping the x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors(None, None, 1)
        nt.assert_equal(p.colors, palettes.color_palette("deep", 3))

        # Test palette mapping the hue position
        p.establish_variables("g", "y", "h", data=self.df)
        p.establish_colors(None, None, 1)
        nt.assert_equal(p.colors, palettes.color_palette("deep", 2))

    def test_default_palette_with_many_levels(self):

        with palettes.color_palette(["blue", "red"], 2):
            p = cat._CategoricalPlotter()
            p.establish_variables("g", "y", data=self.df)
            p.establish_colors(None, None, 1)
            npt.assert_array_equal(p.colors, palettes.husl_palette(3, l=.7))

    def test_specific_color(self):

        p = cat._CategoricalPlotter()

        # Test the same color for each x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors("blue", None, 1)
        blue_rgb = mpl.colors.colorConverter.to_rgb("blue")
        nt.assert_equal(p.colors, [blue_rgb] * 3)

        # Test a color-based blend for the hue mapping
        p.establish_variables("g", "y", "h", data=self.df)
        p.establish_colors("#ff0022", None, 1)
        rgba_array = palettes.light_palette("#ff0022", 2)
        npt.assert_array_almost_equal(p.colors,
                                      rgba_array[:, :3])

    def test_specific_palette(self):

        p = cat._CategoricalPlotter()

        # Test palette mapping the x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors(None, "dark", 1)
        nt.assert_equal(p.colors, palettes.color_palette("dark", 3))

        # Test that non-None `color` and `hue` raises an error
        p.establish_variables("g", "y", "h", data=self.df)
        p.establish_colors(None, "muted", 1)
        nt.assert_equal(p.colors, palettes.color_palette("muted", 2))

        # Test that specified palette overrides specified color
        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors("blue", "deep", 1)
        nt.assert_equal(p.colors, palettes.color_palette("deep", 3))

    def test_dict_as_palette(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", "h", data=self.df)
        pal = {"m": (0, 0, 1), "n": (1, 0, 0)}
        p.establish_colors(None, pal, 1)
        nt.assert_equal(p.colors, [(0, 0, 1), (1, 0, 0)])

    def test_palette_desaturation(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors((0, 0, 1), None, .5)
        nt.assert_equal(p.colors, [(.25, .25, .75)] * 3)

        p.establish_colors(None, [(0, 0, 1), (1, 0, 0), "w"], .5)
        nt.assert_equal(p.colors, [(.25, .25, .75),
                                   (.75, .25, .25),
                                   (1, 1, 1)])


class TestCategoricalStatPlotter(CategoricalFixture):

    def test_no_bootstrappig(self):

        p = cat._CategoricalStatPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.estimate_statistic(np.mean, None, 100)
        npt.assert_array_equal(p.confint, np.array([]))

        p.establish_variables("g", "y", "h", data=self.df)
        p.estimate_statistic(np.mean, None, 100)
        npt.assert_array_equal(p.confint, np.array([[], [], []]))

    def test_single_layer_stats(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y)
        p.estimate_statistic(np.mean, 95, 10000)

        nt.assert_equal(p.statistic.shape, (3,))
        nt.assert_equal(p.confint.shape, (3, 2))

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby(g).mean())

        for ci, (_, grp_y) in zip(p.confint, y.groupby(g)):
            sem = stats.sem(grp_y)
            mean = grp_y.mean()
            ci_want = mean - 1.96 * sem, mean + 1.96 * sem
            npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_single_layer_stats_with_units(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 90))
        y = pd.Series(np.random.RandomState(0).randn(270))
        u = pd.Series(np.repeat(np.tile(list("xyz"), 30), 3))
        y[u == "x"] -= 3
        y[u == "y"] += 3

        p.establish_variables(g, y)
        p.estimate_statistic(np.mean, 95, 10000)
        stat1, ci1 = p.statistic, p.confint

        p.establish_variables(g, y, units=u)
        p.estimate_statistic(np.mean, 95, 10000)
        stat2, ci2 = p.statistic, p.confint

        npt.assert_array_equal(stat1, stat2)
        ci1_size = ci1[:, 1] - ci1[:, 0]
        ci2_size = ci2[:, 1] - ci2[:, 0]
        npt.assert_array_less(ci1_size, ci2_size)

    def test_single_layer_stats_with_missing_data(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y, order=list("abdc"))
        p.estimate_statistic(np.mean, 95, 10000)

        nt.assert_equal(p.statistic.shape, (4,))
        nt.assert_equal(p.confint.shape, (4, 2))

        mean = y[g == "b"].mean()
        sem = stats.sem(y[g == "b"])
        ci = mean - 1.96 * sem, mean + 1.96 * sem
        npt.assert_equal(p.statistic[1], mean)
        npt.assert_array_almost_equal(p.confint[1], ci, 2)

        npt.assert_equal(p.statistic[2], np.nan)
        npt.assert_array_equal(p.confint[2], (np.nan, np.nan))

    def test_nested_stats(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        h = pd.Series(np.tile(list("xy"), 150))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y, h)
        p.estimate_statistic(np.mean, 95, 10000)

        nt.assert_equal(p.statistic.shape, (3, 2))
        nt.assert_equal(p.confint.shape, (3, 2, 2))

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby([g, h]).mean().unstack())

        for ci_g, (_, grp_y) in zip(p.confint, y.groupby(g)):
            for ci, hue_y in zip(ci_g, [grp_y[::2], grp_y[1::2]]):
                sem = stats.sem(hue_y)
                mean = hue_y.mean()
                ci_want = mean - 1.96 * sem, mean + 1.96 * sem
                npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_nested_stats_with_units(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 90))
        h = pd.Series(np.tile(list("xy"), 135))
        u = pd.Series(np.repeat(list("ijkijk"), 45))
        y = pd.Series(np.random.RandomState(0).randn(270))
        y[u == "i"] -= 3
        y[u == "k"] += 3

        p.establish_variables(g, y, h)
        p.estimate_statistic(np.mean, 95, 10000)
        stat1, ci1 = p.statistic, p.confint

        p.establish_variables(g, y, h, units=u)
        p.estimate_statistic(np.mean, 95, 10000)
        stat2, ci2 = p.statistic, p.confint

        npt.assert_array_equal(stat1, stat2)
        ci1_size = ci1[:, 0, 1] - ci1[:, 0, 0]
        ci2_size = ci2[:, 0, 1] - ci2[:, 0, 0]
        npt.assert_array_less(ci1_size, ci2_size)

    def test_nested_stats_with_missing_data(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        y = pd.Series(np.random.RandomState(0).randn(300))
        h = pd.Series(np.tile(list("xy"), 150))

        p.establish_variables(g, y, h,
                              order=list("abdc"),
                              hue_order=list("zyx"))
        p.estimate_statistic(np.mean, 95, 10000)

        nt.assert_equal(p.statistic.shape, (4, 3))
        nt.assert_equal(p.confint.shape, (4, 3, 2))

        mean = y[(g == "b") & (h == "x")].mean()
        sem = stats.sem(y[(g == "b") & (h == "x")])
        ci = mean - 1.96 * sem, mean + 1.96 * sem
        npt.assert_equal(p.statistic[1, 2], mean)
        npt.assert_array_almost_equal(p.confint[1, 2], ci, 2)

        npt.assert_array_equal(p.statistic[:, 0], [np.nan] * 4)
        npt.assert_array_equal(p.statistic[2], [np.nan] * 3)
        npt.assert_array_equal(p.confint[:, 0],
                               np.zeros((4, 2)) * np.nan)
        npt.assert_array_equal(p.confint[2],
                               np.zeros((3, 2)) * np.nan)

    def test_estimator_value_label(self):

        p = cat._CategoricalStatPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.estimate_statistic(np.mean, None, 100)
        nt.assert_equal(p.value_label, "mean(y)")

        p = cat._CategoricalStatPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.estimate_statistic(np.median, None, 100)
        nt.assert_equal(p.value_label, "median(y)")


class TestBoxPlotter(CategoricalFixture):

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       orient=None, color=None, palette=None,
                       saturation=.75, width=.8,
                       fliersize=5, linewidth=None)

    def test_nested_width(self):

        p = cat._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(p.nested_width, .4 * .98)

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(p.nested_width, .3 * .98)

    def test_hue_offsets(self):

        p = cat._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", "h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.2, .2])

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", "h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.15, .15])

        p = cat._BoxPlotter(**kws)
        p.establish_variables("h", "y", "g", data=self.df)
        npt.assert_array_almost_equal(p.hue_offsets, [-.2, 0, .2])

    def test_axes_data(self):

        ax = cat.boxplot("g", "y", data=self.df)
        nt.assert_equal(len(ax.artists), 3)

        plt.close("all")

        ax = cat.boxplot("g", "y", "h", data=self.df)
        nt.assert_equal(len(ax.artists), 6)

        plt.close("all")

    def test_box_colors(self):

        ax = cat.boxplot("g", "y", data=self.df, saturation=1)
        pal = palettes.color_palette("deep", 3)
        for patch, color in zip(ax.artists, pal):
            nt.assert_equal(patch.get_facecolor()[:3], color)

        plt.close("all")

        ax = cat.boxplot("g", "y", "h", data=self.df, saturation=1)
        pal = palettes.color_palette("deep", 2)
        for patch, color in zip(ax.artists, pal * 2):
            nt.assert_equal(patch.get_facecolor()[:3], color)

        plt.close("all")

    def test_axes_annotation(self):

        ax = cat.boxplot("g", "y", data=self.df)
        nt.assert_equal(ax.get_xlabel(), "g")
        nt.assert_equal(ax.get_ylabel(), "y")
        nt.assert_equal(ax.get_xlim(), (-.5, 2.5))
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])

        plt.close("all")

        ax = cat.boxplot("g", "y", "h", data=self.df)
        nt.assert_equal(ax.get_xlabel(), "g")
        nt.assert_equal(ax.get_ylabel(), "y")
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])
        npt.assert_array_equal([l.get_text() for l in ax.legend_.get_texts()],
                               ["m", "n"])

        plt.close("all")

        ax = cat.boxplot("y", "g", data=self.df, orient="h")
        nt.assert_equal(ax.get_xlabel(), "y")
        nt.assert_equal(ax.get_ylabel(), "g")
        nt.assert_equal(ax.get_ylim(), (2.5, -.5))
        npt.assert_array_equal(ax.get_yticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_yticklabels()],
                               ["a", "b", "c"])

        plt.close("all")


class TestViolinPlotter(CategoricalFixture):

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       bw="scott", cut=2, scale="area", scale_hue=True,
                       gridsize=100, width=.8, inner="box", split=False,
                       orient=None, linewidth=None,
                       color=None, palette=None, saturation=.75)

    def test_split_error(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="h", y="y", hue="g", data=self.df, split=True))

        with nt.assert_raises(ValueError):
            cat._ViolinPlotter(**kws)

    def test_no_observations(self):

        p = cat._ViolinPlotter(**self.default_kws)

        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        y[-1] = np.nan
        p.establish_variables(x, y)
        p.estimate_densities("scott", 2, "area", True, 20)

        nt.assert_equal(len(p.support[0]), 20)
        nt.assert_equal(len(p.support[1]), 0)

        nt.assert_equal(len(p.density[0]), 20)
        nt.assert_equal(len(p.density[1]), 1)

        nt.assert_equal(p.density[1].item(), 1)

        p.estimate_densities("scott", 2, "count", True, 20)
        nt.assert_equal(p.density[1].item(), 0)

        x = ["a"] * 4 + ["b"] * 2
        y = self.rs.randn(6)
        h = ["m", "n"] * 2 + ["m"] * 2

        p.establish_variables(x, y, h)
        p.estimate_densities("scott", 2, "area", True, 20)

        nt.assert_equal(len(p.support[1][0]), 20)
        nt.assert_equal(len(p.support[1][1]), 0)

        nt.assert_equal(len(p.density[1][0]), 20)
        nt.assert_equal(len(p.density[1][1]), 1)

        nt.assert_equal(p.density[1][1].item(), 1)

        p.estimate_densities("scott", 2, "count", False, 20)
        nt.assert_equal(p.density[1][1].item(), 0)

    def test_single_observation(self):

        p = cat._ViolinPlotter(**self.default_kws)

        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        p.establish_variables(x, y)
        p.estimate_densities("scott", 2, "area", True, 20)

        nt.assert_equal(len(p.support[0]), 20)
        nt.assert_equal(len(p.support[1]), 1)

        nt.assert_equal(len(p.density[0]), 20)
        nt.assert_equal(len(p.density[1]), 1)

        nt.assert_equal(p.density[1].item(), 1)

        p.estimate_densities("scott", 2, "count", True, 20)
        nt.assert_equal(p.density[1].item(), .5)

        x = ["b"] * 4 + ["a"] * 3
        y = self.rs.randn(7)
        h = (["m", "n"] * 4)[:-1]

        p.establish_variables(x, y, h)
        p.estimate_densities("scott", 2, "area", True, 20)

        nt.assert_equal(len(p.support[1][0]), 20)
        nt.assert_equal(len(p.support[1][1]), 1)

        nt.assert_equal(len(p.density[1][0]), 20)
        nt.assert_equal(len(p.density[1][1]), 1)

        nt.assert_equal(p.density[1][1].item(), 1)

        p.estimate_densities("scott", 2, "count", False, 20)
        nt.assert_equal(p.density[1][1].item(), .5)

    def test_dwidth(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="g", y="y", data=self.df))

        p = cat._ViolinPlotter(**kws)
        nt.assert_equal(p.dwidth, .4)

        kws.update(dict(width=.4))
        p = cat._ViolinPlotter(**kws)
        nt.assert_equal(p.dwidth, .2)

        kws.update(dict(hue="h", width=.8))
        p = cat._ViolinPlotter(**kws)
        nt.assert_equal(p.dwidth, .2)

        kws.update(dict(split=True))
        p = cat._ViolinPlotter(**kws)
        nt.assert_equal(p.dwidth, .4)

    def test_scale_area(self):

        kws = self.default_kws.copy()
        kws["scale"] = "area"
        p = cat._ViolinPlotter(**kws)

        # Test single layer of grouping
        p.hue_names = None
        density = [self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)]
        max_before = np.array([d.max() for d in density])
        p.scale_area(density, max_before, False)
        max_after = np.array([d.max() for d in density])
        nt.assert_equal(max_after[0], 1)

        before_ratio = max_before[1] / max_before[0]
        after_ratio = max_after[1] / max_after[0]
        nt.assert_equal(before_ratio, after_ratio)

        # Test nested grouping scaling across all densities
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)],
                   [self.rs.uniform(0, .1, 50), self.rs.uniform(0, .02, 50)]]

        max_before = np.array([[r.max() for r in row] for row in density])
        p.scale_area(density, max_before, False)
        max_after = np.array([[r.max() for r in row] for row in density])
        nt.assert_equal(max_after[0, 0], 1)

        before_ratio = max_before[1, 1] / max_before[0, 0]
        after_ratio = max_after[1, 1] / max_after[0, 0]
        nt.assert_equal(before_ratio, after_ratio)

        # Test nested grouping scaling within hue
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)],
                   [self.rs.uniform(0, .1, 50), self.rs.uniform(0, .02, 50)]]

        max_before = np.array([[r.max() for r in row] for row in density])
        p.scale_area(density, max_before, True)
        max_after = np.array([[r.max() for r in row] for row in density])
        nt.assert_equal(max_after[0, 0], 1)
        nt.assert_equal(max_after[1, 0], 1)

        before_ratio = max_before[1, 1] / max_before[1, 0]
        after_ratio = max_after[1, 1] / max_after[1, 0]
        nt.assert_equal(before_ratio, after_ratio)

    def test_scale_width(self):

        kws = self.default_kws.copy()
        kws["scale"] = "width"
        p = cat._ViolinPlotter(**kws)

        # Test single layer of grouping
        p.hue_names = None
        density = [self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)]
        p.scale_width(density)
        max_after = np.array([d.max() for d in density])
        npt.assert_array_equal(max_after, [1, 1])

        # Test nested grouping
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)],
                   [self.rs.uniform(0, .1, 50), self.rs.uniform(0, .02, 50)]]

        p.scale_width(density)
        max_after = np.array([[r.max() for r in row] for row in density])
        npt.assert_array_equal(max_after, [[1, 1], [1, 1]])

    def test_scale_count(self):

        kws = self.default_kws.copy()
        kws["scale"] = "count"
        p = cat._ViolinPlotter(**kws)

        # Test single layer of grouping
        p.hue_names = None
        density = [self.rs.uniform(0, .8, 20), self.rs.uniform(0, .2, 40)]
        counts = np.array([20, 40])
        p.scale_count(density, counts, False)
        max_after = np.array([d.max() for d in density])
        npt.assert_array_equal(max_after, [.5, 1])

        # Test nested grouping scaling across all densities
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 5), self.rs.uniform(0, .2, 40)],
                   [self.rs.uniform(0, .1, 100), self.rs.uniform(0, .02, 50)]]

        counts = np.array([[5, 40], [100, 50]])
        p.scale_count(density, counts, False)
        max_after = np.array([[r.max() for r in row] for row in density])
        npt.assert_array_equal(max_after, [[.05, .4], [1, .5]])

        # Test nested grouping scaling within hue
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 5), self.rs.uniform(0, .2, 40)],
                   [self.rs.uniform(0, .1, 100), self.rs.uniform(0, .02, 50)]]

        counts = np.array([[5, 40], [100, 50]])
        p.scale_count(density, counts, True)
        max_after = np.array([[r.max() for r in row] for row in density])
        npt.assert_array_equal(max_after, [[.125, 1], [1, .5]])

    def test_bad_scale(self):

        kws = self.default_kws.copy()
        kws["scale"] = "not_a_scale_type"
        with nt.assert_raises(ValueError):
            cat._ViolinPlotter(**kws)

    def test_kde_fit(self):

        p = cat._ViolinPlotter(**self.default_kws)
        data = self.y
        data_std = data.std(ddof=1)

        # Bandwidth behavior depends on scipy version
        if LooseVersion(scipy.__version__) < "0.11":
            # Test ignoring custom bandwidth on old scipy
            kde, bw = p.fit_kde(self.y, .2)
            nt.assert_is_instance(kde, stats.gaussian_kde)
            nt.assert_equal(kde.factor, kde.scotts_factor())

        else:
            # Test reference rule bandwidth
            kde, bw = p.fit_kde(data, "scott")
            nt.assert_is_instance(kde, stats.gaussian_kde)
            nt.assert_equal(kde.factor, kde.scotts_factor())
            nt.assert_equal(bw, kde.scotts_factor() * data_std)

            # Test numeric scale factor
            kde, bw = p.fit_kde(self.y, .2)
            nt.assert_is_instance(kde, stats.gaussian_kde)
            nt.assert_equal(kde.factor, .2)
            nt.assert_equal(bw, .2 * data_std)

    def test_draw_to_density(self):

        p = cat._ViolinPlotter(**self.default_kws)
        # p.dwidth will be 1 for easier testing
        p.width = 2

        # Test verical plots
        support = np.array([.2, .6])
        density = np.array([.1, .4])

        # Test full vertical plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .5, support, density, False)
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [.99 * -.4, .99 * .4])
        npt.assert_array_equal(y, [.5, .5])
        plt.close("all")

        # Test left vertical plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .5, support, density, "left")
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [.99 * -.4, 0])
        npt.assert_array_equal(y, [.5, .5])
        plt.close("all")

        # Test right vertical plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .5, support, density, "right")
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [0, .99 * .4])
        npt.assert_array_equal(y, [.5, .5])
        plt.close("all")

        # Switch orientation to test horizontal plots
        p.orient = "h"
        support = np.array([.2, .5])
        density = np.array([.3, .7])

        # Test full horizontal plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .6, support, density, False)
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [.6, .6])
        npt.assert_array_equal(y, [.99 * -.7, .99 * .7])
        plt.close("all")

        # Test left horizontal plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .6, support, density, "left")
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [.6, .6])
        npt.assert_array_equal(y, [.99 * -.7, 0])
        plt.close("all")

        # Test right horizontal plot
        _, ax = plt.subplots()
        p.draw_to_density(ax, 0, .6, support, density, "right")
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [.6, .6])
        npt.assert_array_equal(y, [0, .99 * .7])
        plt.close("all")

    def test_draw_single_observations(self):

        p = cat._ViolinPlotter(**self.default_kws)
        p.width = 2

        # Test vertical plot
        _, ax = plt.subplots()
        p.draw_single_observation(ax, 1, 1.5, 1)
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [0, 2])
        npt.assert_array_equal(y, [1.5, 1.5])
        plt.close("all")

        # Test horizontal plot
        p.orient = "h"
        _, ax = plt.subplots()
        p.draw_single_observation(ax, 2, 2.2, .5)
        x, y = ax.lines[0].get_xydata().T
        npt.assert_array_equal(x, [2.2, 2.2])
        npt.assert_array_equal(y, [1.5, 2.5])
        plt.close("all")

    def test_draw_box_lines(self):

        # Test vertical plot
        kws = self.default_kws.copy()
        kws.update(dict(y="y", data=self.df, inner=None))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_box_lines(ax, self.y, p.support[0], p.density[0], 0)
        nt.assert_equal(len(ax.lines), 2)

        q25, q50, q75 = np.percentile(self.y, [25, 50, 75])
        _, y = ax.lines[1].get_xydata().T
        npt.assert_array_equal(y, [q25, q75])

        _, y = ax.collections[0].get_offsets().T
        nt.assert_equal(y, q50)

        plt.close("all")

        # Test horizontal plot
        kws = self.default_kws.copy()
        kws.update(dict(x="y", data=self.df, inner=None))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_box_lines(ax, self.y, p.support[0], p.density[0], 0)
        nt.assert_equal(len(ax.lines), 2)

        q25, q50, q75 = np.percentile(self.y, [25, 50, 75])
        x, _ = ax.lines[1].get_xydata().T
        npt.assert_array_equal(x, [q25, q75])

        x, _ = ax.collections[0].get_offsets().T
        nt.assert_equal(x, q50)

        plt.close("all")

    def test_draw_quartiles(self):

        kws = self.default_kws.copy()
        kws.update(dict(y="y", data=self.df, inner=None))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_quartiles(ax, self.y, p.support[0], p.density[0], 0)
        for val, line in zip(np.percentile(self.y, [25, 50, 75]), ax.lines):
            _, y = line.get_xydata().T
            npt.assert_array_equal(y, [val, val])
        plt.close("all")

    def test_draw_points(self):

        p = cat._ViolinPlotter(**self.default_kws)

        # Test vertical plot
        _, ax = plt.subplots()
        p.draw_points(ax, self.y, 0)
        x, y = ax.collections[0].get_offsets().T
        npt.assert_array_equal(x, np.zeros_like(self.y))
        npt.assert_array_equal(y, self.y)
        plt.close("all")

        # Test horizontal plot
        p.orient = "h"
        _, ax = plt.subplots()
        p.draw_points(ax, self.y, 0)
        x, y = ax.collections[0].get_offsets().T
        npt.assert_array_equal(x, self.y)
        npt.assert_array_equal(y, np.zeros_like(self.y))
        plt.close("all")

    def test_draw_sticks(self):

        kws = self.default_kws.copy()
        kws.update(dict(y="y", data=self.df, inner=None))
        p = cat._ViolinPlotter(**kws)

        # Test vertical plot
        _, ax = plt.subplots()
        p.draw_stick_lines(ax, self.y, p.support[0], p.density[0], 0)
        for val, line in zip(self.y, ax.lines):
            _, y = line.get_xydata().T
            npt.assert_array_equal(y, [val, val])
        plt.close("all")

        # Test horizontal plot
        p.orient = "h"
        _, ax = plt.subplots()
        p.draw_stick_lines(ax, self.y, p.support[0], p.density[0], 0)
        for val, line in zip(self.y, ax.lines):
            x, _ = line.get_xydata().T
            npt.assert_array_equal(x, [val, val])
        plt.close("all")

    def test_draw_violinplots(self):

        kws = self.default_kws.copy()

        # Test single vertical violin
        kws.update(dict(y="y", data=self.df, inner=None,
                        saturation=1, color=(1, 0, 0, 1)))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 1)
        npt.assert_array_equal(ax.collections[0].get_facecolors(),
                               [(1, 0, 0, 1)])
        plt.close("all")

        # Test single horizontal violin
        kws.update(dict(x="y", y=None, color=(0, 1, 0, 1)))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 1)
        npt.assert_array_equal(ax.collections[0].get_facecolors(),
                               [(0, 1, 0, 1)])
        plt.close("all")

        # Test multiple vertical violins
        kws.update(dict(x="g", y="y", color=None,))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 3)
        for violin, color in zip(ax.collections, palettes.color_palette()):
            npt.assert_array_equal(violin.get_facecolors()[0, :-1], color)
        plt.close("all")

        # Test multiple violins with hue nesting
        kws.update(dict(hue="h"))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 6)
        for violin, color in zip(ax.collections,
                                 palettes.color_palette(n_colors=2) * 3):
            npt.assert_array_equal(violin.get_facecolors()[0, :-1], color)
        plt.close("all")

        # Test multiple split violins
        kws.update(dict(split=True, palette="muted"))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 6)
        for violin, color in zip(ax.collections,
                                 palettes.color_palette("muted",
                                                        n_colors=2) * 3):
            npt.assert_array_equal(violin.get_facecolors()[0, :-1], color)
        plt.close("all")

    def test_draw_violinplots_no_observations(self):

        kws = self.default_kws.copy()
        kws["inner"] = None

        # Test single layer of grouping
        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        y[-1] = np.nan
        kws.update(x=x, y=y)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 1)
        nt.assert_equal(len(ax.lines), 0)
        plt.close("all")

        # Test nested hue grouping
        x = ["a"] * 4 + ["b"] * 2
        y = self.rs.randn(6)
        h = ["m", "n"] * 2 + ["m"] * 2
        kws.update(x=x, y=y, hue=h)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 3)
        nt.assert_equal(len(ax.lines), 0)
        plt.close("all")

    def test_draw_violinplots_single_observations(self):

        kws = self.default_kws.copy()
        kws["inner"] = None

        # Test single layer of grouping
        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        kws.update(x=x, y=y)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 1)
        nt.assert_equal(len(ax.lines), 1)
        plt.close("all")

        # Test nested hue grouping
        x = ["b"] * 4 + ["a"] * 3
        y = self.rs.randn(7)
        h = (["m", "n"] * 4)[:-1]
        kws.update(x=x, y=y, hue=h)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 3)
        nt.assert_equal(len(ax.lines), 1)
        plt.close("all")

        # Test nested hue grouping with split
        kws["split"] = True
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        nt.assert_equal(len(ax.collections), 3)
        nt.assert_equal(len(ax.lines), 1)
        plt.close("all")

    def test_violinplots(self):

        # Smoke test the high level violinplot options

        cat.violinplot("y", data=self.df)
        plt.close("all")

        cat.violinplot(y="y", data=self.df)
        plt.close("all")

        cat.violinplot("g", "y", data=self.df)
        plt.close("all")

        cat.violinplot("y", "g", data=self.df, orient="h")
        plt.close("all")

        cat.violinplot("g", "y", "h", data=self.df)
        plt.close("all")

        cat.violinplot("y", "g", "h", data=self.df, orient="h")
        plt.close("all")

        for inner in ["box", "quart", "point", "stick", None]:
            cat.violinplot("g", "y", data=self.df, inner=inner)
            plt.close("all")

            cat.violinplot("g", "y", "h", data=self.df, inner=inner)
            plt.close("all")

            cat.violinplot("g", "y", "h", data=self.df,
                           inner=inner, split=True)
            plt.close("all")


class TestStripPlotter(CategoricalFixture):

    def test_stripplot_vertical(self):

        pal = palettes.color_palette()

        ax = cat.stripplot("g", "y", data=self.df)
        for i, (_, vals) in enumerate(self.y.groupby(self.g)):

            x, y = ax.collections[i].get_offsets().T

            npt.assert_array_equal(x, np.ones(len(x)) * i)
            npt.assert_array_equal(y, vals)

            npt.assert_equal(ax.collections[i].get_facecolors()[0, :3], pal[i])

        plt.close("all")

    @skipif(not pandas_has_categoricals)
    def test_stripplot_horiztonal(self):

        df = self.df.copy()
        df.g = df.g.astype("category")

        ax = cat.stripplot("y", "g", data=df)
        for i, (_, vals) in enumerate(self.y.groupby(self.g)):

            x, y = ax.collections[i].get_offsets().T

            npt.assert_array_equal(x, vals)
            npt.assert_array_equal(y, np.ones(len(x)) * i)

        plt.close("all")

    def test_stripplot_jitter(self):

        pal = palettes.color_palette()

        ax = cat.stripplot("g", "y", data=self.df, jitter=True)
        for i, (_, vals) in enumerate(self.y.groupby(self.g)):

            x, y = ax.collections[i].get_offsets().T

            npt.assert_array_less(np.ones(len(x)) * i - .1, x)
            npt.assert_array_less(x, np.ones(len(x)) * i + .1)
            npt.assert_array_equal(y, vals)

            npt.assert_equal(ax.collections[i].get_facecolors()[0, :3], pal[i])

        plt.close("all")

    def test_split_nested_stripplot_vertical(self):

        pal = palettes.color_palette()

        ax = cat.stripplot("g", "y", "h", data=self.df)
        for i, (_, group_vals) in enumerate(self.y.groupby(self.g)):
            for j, (_, vals) in enumerate(group_vals.groupby(self.h)):

                x, y = ax.collections[i * 2 + j].get_offsets().T

                npt.assert_array_equal(x, np.ones(len(x)) * i + [-.2, .2][j])
                npt.assert_array_equal(y, vals)

                fc = ax.collections[i * 2 + j].get_facecolors()[0, :3]
                npt.assert_equal(fc, pal[j])

        plt.close("all")

    @skipif(not pandas_has_categoricals)
    def test_split_nested_stripplot_horizontal(self):

        df = self.df.copy()
        df.g = df.g.astype("category")

        ax = cat.stripplot("y", "g", "h", data=df)
        for i, (_, group_vals) in enumerate(self.y.groupby(self.g)):
            for j, (_, vals) in enumerate(group_vals.groupby(self.h)):

                x, y = ax.collections[i * 2 + j].get_offsets().T

                npt.assert_array_equal(x, vals)
                npt.assert_array_equal(y, np.ones(len(x)) * i + [-.2, .2][j])

        plt.close("all")

    def test_unsplit_nested_stripplot_vertical(self):

        pal = palettes.color_palette()

        # Test a simple vertical strip plot
        ax = cat.stripplot("g", "y", "h", data=self.df, split=False)
        for i, (_, group_vals) in enumerate(self.y.groupby(self.g)):
            for j, (_, vals) in enumerate(group_vals.groupby(self.h)):

                x, y = ax.collections[i * 2 + j].get_offsets().T

                npt.assert_array_equal(x, np.ones(len(x)) * i)
                npt.assert_array_equal(y, vals)

                fc = ax.collections[i * 2 + j].get_facecolors()[0, :3]
                npt.assert_equal(fc, pal[j])

        plt.close("all")

    @skipif(not pandas_has_categoricals)
    def test_unsplit_nested_stripplot_horizontal(self):

        df = self.df.copy()
        df.g = df.g.astype("category")

        ax = cat.stripplot("y", "g", "h", data=df, split=False)
        for i, (_, group_vals) in enumerate(self.y.groupby(self.g)):
            for j, (_, vals) in enumerate(group_vals.groupby(self.h)):

                x, y = ax.collections[i * 2 + j].get_offsets().T

                npt.assert_array_equal(x, vals)
                npt.assert_array_equal(y, np.ones(len(x)) * i)

        plt.close("all")
