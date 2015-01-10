import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
pandas_has_categoricals = LooseVersion(pd.__version__) >= "0.15"

import nose.tools as nt
import numpy.testing as npt
from numpy.testing.decorators import skipif

from .. import distributions as dist
from .. import palettes

try:
    import statsmodels.nonparametric.api
    assert statsmodels.nonparametric.api
    _no_statsmodels = False
except ImportError:
    _no_statsmodels = True


class TestBoxPlotter(object):
    """Test boxplot (also base class for things like violinplots)."""
    rs = np.random.RandomState(30)
    n_total = 60
    x = rs.randn(n_total / 3, 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    g = pd.Series(np.repeat(list("abc"), n_total / 3), name="small")
    h = pd.Series(np.tile(list("mn"), n_total / 2), name="medium")
    df = pd.DataFrame(dict(y=y, g=g, h=h))
    x_df["W"] = g

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       orient=None, color=None, palette=None,
                       saturation=.75, alpha=None,
                       width=.8, fliersize=5, linewidth=None)

    def test_wide_df_data(self):

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

        # Test that passing actually 3D data raises
        x = np.zeros((5, 5, 5))
        with nt.assert_raises(ValueError):
            p.establish_variables(data=x)

    def test_list_of_array_input_data(self):

        p = dist._BoxPlotter(**self.default_kws)

        # Test 2D input in list form
        x_list = self.x.T.tolist()
        p.establish_variables(data=x_list)
        nt.assert_equal(len(p.plot_data), 3)

        lengths = [len(v_i) for v_i in p.plot_data]
        nt.assert_equal(lengths, [self.n_total / 3] * 3)

        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_wide_array_input_data(self):

        p = dist._BoxPlotter(**self.default_kws)

        # Test 2D input in array form
        p.establish_variables(data=self.x)
        nt.assert_equal(np.shape(p.plot_data), (3, self.n_total / 3))
        npt.assert_array_equal(p.plot_data, self.x.T)

        nt.assert_is(p.group_label, None)
        nt.assert_is(p.value_label, None)

    def test_single_long_direct_inputs(self):

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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

    def test_orient_inference(self):

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

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
            p = dist._BoxPlotter(**self.default_kws)
            p.establish_variables("g", "y", data=self.df)
            p.establish_colors(None, None, 1)
            npt.assert_array_equal(p.colors, palettes.husl_palette(3, l=.7))

    def test_specific_color(self):

        p = dist._BoxPlotter(**self.default_kws)

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

        p = dist._BoxPlotter(**self.default_kws)

        # Test palette mapping the x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors(None, "dark", 1)
        nt.assert_equal(p.colors, palettes.color_palette("dark", 3))

        # Test that non-None `color` and `hue` raises an error
        p.establish_variables("g", "y", "h", data=self.df)
        p.establish_colors(None, "muted", 1)
        nt.assert_equal(p.colors, palettes.color_palette("muted", 2))

        # Test that specified palette overrides specified color
        p = dist._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors("blue", "deep", 1)
        nt.assert_equal(p.colors, palettes.color_palette("deep", 3))

    def test_dict_as_palette(self):

        p = dist._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", "h", data=self.df)
        pal = {"m": (0, 0, 1), "n": (1, 0, 0)}
        p.establish_colors(None, pal, 1)
        nt.assert_equal(p.colors, [(0, 0, 1), (1, 0, 0)])

    def test_palette_desaturation(self):

        p = dist._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors((0, 0, 1), None, .5)
        nt.assert_equal(p.colors, [(.25, .25, .75)] * 3)

        p.establish_colors(None, [(0, 0, 1), (1, 0, 0), "w"], .5)
        nt.assert_equal(p.colors, [(.25, .25, .75),
                                   (.75, .25, .25),
                                   (1, 1, 1)])

    def test_nested_width(self):

        p = dist._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(p.nested_width, .4 * .98)

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = dist._BoxPlotter(**kws)
        p.establish_variables("g", "y", "h", data=self.df)
        nt.assert_equal(p.nested_width, .3 * .98)

    def test_hue_offsets(self):

        p = dist._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", "h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.2, .2])

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = dist._BoxPlotter(**kws)
        p.establish_variables("g", "y", "h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.15, .15])

        p = dist._BoxPlotter(**kws)
        p.establish_variables("h", "y", "g", data=self.df)
        npt.assert_array_almost_equal(p.hue_offsets, [-.2, 0, .2])

    def test_axes_data(self):

        ax = dist.boxplot("g", "y", data=self.df)
        nt.assert_equal(len(ax.artists), 3)
        nt.assert_equal(len(ax.lines), 18)

        plt.close("all")

        ax = dist.boxplot("g", "y", "h", data=self.df)
        nt.assert_equal(len(ax.artists), 6)
        nt.assert_equal(len(ax.lines), 36)

        plt.close("all")

    def test_box_colors(self):

        ax = dist.boxplot("g", "y", data=self.df, saturation=1)
        pal = palettes.color_palette("deep", 3)
        for patch, color in zip(ax.artists, pal):
            nt.assert_equal(patch.get_facecolor()[:3], color)

        plt.close("all")

        ax = dist.boxplot("g", "y", "h", data=self.df, saturation=1)
        pal = palettes.color_palette("deep", 2)
        for patch, color in zip(ax.artists, pal * 2):
            nt.assert_equal(patch.get_facecolor()[:3], color)

        plt.close("all")

    def test_axes_annotation(self):

        ax = dist.boxplot("g", "y", data=self.df)
        nt.assert_equal(ax.get_xlabel(), "g")
        nt.assert_equal(ax.get_ylabel(), "y")
        nt.assert_equal(ax.get_xlim(), (-.5, 2.5))
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])

        plt.close("all")

        ax = dist.boxplot("g", "y", "h", data=self.df)
        nt.assert_equal(ax.get_xlabel(), "g")
        nt.assert_equal(ax.get_ylabel(), "y")
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])
        npt.assert_array_equal([l.get_text() for l in ax.legend_.get_texts()],
                               ["m", "n"])

        plt.close("all")

        ax = dist.boxplot("y", "g", data=self.df, orient="h")
        nt.assert_equal(ax.get_xlabel(), "y")
        nt.assert_equal(ax.get_ylabel(), "g")
        nt.assert_equal(ax.get_ylim(), (-.5, 2.5))
        npt.assert_array_equal(ax.get_yticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_yticklabels()],
                               ["a", "b", "c"])

        plt.close("all")


class TestBoxReshaping(object):
    """Tests for function that preps boxplot/violinplot data."""
    n_total = 60
    rs = np.random.RandomState(0)
    x = rs.randn(n_total / 3, 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    g = pd.Series(np.repeat(list("abc"), n_total / 3), name="small")
    h = pd.Series(np.tile(list("mno"), n_total / 3), name="medium")
    df = pd.DataFrame(dict(y=y, g=g, h=h))

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
        plt.close("all")


class TestViolinPlot(object):

    df = pd.DataFrame(dict(x=np.random.randn(60),
                           y=list("abcdef") * 10,
                           z=list("ab") * 29 + ["a", "c"]))

    def test_single_violin(self):

        ax = dist.violinplot(self.df.x)
        nt.assert_equal(len(ax.collections), 1)
        nt.assert_equal(len(ax.lines), 5)
        plt.close("all")

    def test_multi_violins(self):

        ax = dist.violinplot(self.df.x, self.df.y)
        nt.assert_equal(len(ax.collections), 6)
        nt.assert_equal(len(ax.lines), 30)
        plt.close("all")

    def test_multi_violins_single_obs(self):

        ax = dist.violinplot(self.df.x, self.df.z)
        nt.assert_equal(len(ax.collections), 2)
        nt.assert_equal(len(ax.lines), 11)
        plt.close("all")

        data = [np.random.randn(30), [0, 0, 0]]
        ax = dist.violinplot(data)
        nt.assert_equal(len(ax.collections), 1)
        nt.assert_equal(len(ax.lines), 6)
        plt.close("all")

    @classmethod
    def teardown_class(cls):
        """Ensure that all figures are closed on exit."""
        plt.close("all")


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

    def test_hex_customise(self):

        # test that default gridsize can be overridden
        g = dist.jointplot("x", "y", self.data, kind="hex",
                           joint_kws=dict(gridsize=5))
        nt.assert_equal(len(g.ax_joint.collections), 1)
        a = g.ax_joint.collections[0].get_array()
        nt.assert_equal(28, a.shape[0])  # 28 hexagons expected for gridsize 5

        plt.close("all")

    def test_bad_kind(self):

        with nt.assert_raises(ValueError):
            dist.jointplot("x", "y", self.data, kind="not_a_kind")

    @classmethod
    def teardown_class(cls):
        """Ensure that all figures are closed on exit."""
        plt.close("all")
