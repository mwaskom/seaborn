import itertools
from functools import partial
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, same_color, to_rgb, to_rgba

import pytest
from pytest import approx
import numpy.testing as npt
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
)

from seaborn import categorical as cat
from seaborn import palettes

from seaborn.utils import _version_predates
from seaborn._oldcore import categorical_order
from seaborn.axisgrid import FacetGrid
from seaborn.categorical import (
    _CategoricalPlotterNew,
    Beeswarm,
    catplot,
    stripplot,
    swarmplot,
    barplot,
    pointplot,
)
from seaborn.palettes import color_palette
from seaborn.utils import _normal_quantile_func, _draw_figure
from seaborn._compat import get_colormap
from seaborn._testing import assert_plots_equal


PLOT_FUNCS = [
    catplot,
    stripplot,
    swarmplot,
    barplot,
]


class TestCategoricalPlotterNew:

    @pytest.mark.parametrize(
        "func,kwargs",
        itertools.product(
            PLOT_FUNCS,
            [
                {"x": "x", "y": "a"},
                {"x": "a", "y": "y"},
                {"x": "y"},
                {"y": "x"},
            ],
        ),
    )
    def test_axis_labels(self, long_df, func, kwargs):

        func(data=long_df, **kwargs)

        ax = plt.gca()
        for axis in "xy":
            val = kwargs.get(axis, "")
            label_func = getattr(ax, f"get_{axis}label")
            assert label_func() == val

    @pytest.mark.parametrize("func", PLOT_FUNCS)
    def test_empty(self, func):

        func()
        ax = plt.gca()
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines

        func(x=[], y=[])
        ax = plt.gca()
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines

    def test_redundant_hue_backcompat(self, long_df):

        p = _CategoricalPlotterNew(
            data=long_df,
            variables={"x": "s", "y": "y"},
        )

        color = None
        palette = dict(zip(long_df["s"].unique(), color_palette()))
        hue_order = None

        palette, _ = p._hue_backcompat(color, palette, hue_order, force_hue=True)

        assert p.variables["hue"] == "s"
        assert_array_equal(p.plot_data["hue"], p.plot_data["x"])
        assert all(isinstance(k, str) for k in palette)


class CategoricalFixture:
    """Test boxplot (also base class for things like violinplots)."""
    rs = np.random.RandomState(30)
    n_total = 60
    x = rs.randn(int(n_total / 3), 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    y_perm = y.reindex(rs.choice(y.index, y.size, replace=False))
    g = pd.Series(np.repeat(list("abc"), int(n_total / 3)), name="small")
    h = pd.Series(np.tile(list("mn"), int(n_total / 2)), name="medium")
    u = pd.Series(np.tile(list("jkh"), int(n_total / 3)))
    df = pd.DataFrame(dict(y=y, g=g, h=h, u=u))
    x_df["W"] = g

    def get_box_artists(self, ax):

        if _version_predates(mpl, "3.5.0b0"):
            return ax.artists
        else:
            # Exclude labeled patches, which are for the legend
            return [p for p in ax.patches if not p.get_label()]


class TestCategoricalPlotter(CategoricalFixture):

    def test_wide_df_data(self):

        p = cat._CategoricalPlotter()

        # Test basic wide DataFrame
        p.establish_variables(data=self.x_df)

        # Check data attribute
        for x, y, in zip(p.plot_data, self.x_df[["X", "Y", "Z"]].values.T):
            npt.assert_array_equal(x, y)

        # Check semantic attributes
        assert p.orient == "x"
        assert p.plot_hues is None
        assert p.group_label == "big"
        assert p.value_label is None

        # Test wide dataframe with forced horizontal orientation
        p.establish_variables(data=self.x_df, orient="horiz")
        assert p.orient == "y"

        # Test exception by trying to hue-group with a wide dataframe
        with pytest.raises(ValueError):
            p.establish_variables(hue="d", data=self.x_df)

    def test_1d_input_data(self):

        p = cat._CategoricalPlotter()

        # Test basic vector data
        x_1d_array = self.x.ravel()
        p.establish_variables(data=x_1d_array)
        assert len(p.plot_data) == 1
        assert len(p.plot_data[0]) == self.n_total
        assert p.group_label is None
        assert p.value_label is None

        # Test basic vector data in list form
        x_1d_list = x_1d_array.tolist()
        p.establish_variables(data=x_1d_list)
        assert len(p.plot_data) == 1
        assert len(p.plot_data[0]) == self.n_total
        assert p.group_label is None
        assert p.value_label is None

        # Test an object array that looks 1D but isn't
        x_notreally_1d = np.array([self.x.ravel(),
                                   self.x.ravel()[:int(self.n_total / 2)]],
                                  dtype=object)
        p.establish_variables(data=x_notreally_1d)
        assert len(p.plot_data) == 2
        assert len(p.plot_data[0]) == self.n_total
        assert len(p.plot_data[1]) == self.n_total / 2
        assert p.group_label is None
        assert p.value_label is None

    def test_2d_input_data(self):

        p = cat._CategoricalPlotter()

        x = self.x[:, 0]

        # Test vector data that looks 2D but doesn't really have columns
        p.establish_variables(data=x[:, np.newaxis])
        assert len(p.plot_data) == 1
        assert len(p.plot_data[0]) == self.x.shape[0]
        assert p.group_label is None
        assert p.value_label is None

        # Test vector data that looks 2D but doesn't really have rows
        p.establish_variables(data=x[np.newaxis, :])
        assert len(p.plot_data) == 1
        assert len(p.plot_data[0]) == self.x.shape[0]
        assert p.group_label is None
        assert p.value_label is None

    def test_3d_input_data(self):

        p = cat._CategoricalPlotter()

        # Test that passing actually 3D data raises
        x = np.zeros((5, 5, 5))
        with pytest.raises(ValueError):
            p.establish_variables(data=x)

    def test_list_of_array_input_data(self):

        p = cat._CategoricalPlotter()

        # Test 2D input in list form
        x_list = self.x.T.tolist()
        p.establish_variables(data=x_list)
        assert len(p.plot_data) == 3

        lengths = [len(v_i) for v_i in p.plot_data]
        assert lengths == [self.n_total / 3] * 3

        assert p.group_label is None
        assert p.value_label is None

    def test_wide_array_input_data(self):

        p = cat._CategoricalPlotter()

        # Test 2D input in array form
        p.establish_variables(data=self.x)
        assert np.shape(p.plot_data) == (3, self.n_total / 3)
        npt.assert_array_equal(p.plot_data, self.x.T)

        assert p.group_label is None
        assert p.value_label is None

    def test_single_long_direct_inputs(self):

        p = cat._CategoricalPlotter()

        # Test passing a series to the x variable
        p.establish_variables(x=self.y)
        npt.assert_equal(p.plot_data, [self.y])
        assert p.orient == "y"
        assert p.value_label == "y_data"
        assert p.group_label is None

        # Test passing a series to the y variable
        p.establish_variables(y=self.y)
        npt.assert_equal(p.plot_data, [self.y])
        assert p.orient == "x"
        assert p.value_label == "y_data"
        assert p.group_label is None

        # Test passing an array to the y variable
        p.establish_variables(y=self.y.values)
        npt.assert_equal(p.plot_data, [self.y])
        assert p.orient == "x"
        assert p.group_label is None
        assert p.value_label is None

        # Test array and series with non-default index
        x = pd.Series([1, 1, 1, 1], index=[0, 2, 4, 6])
        y = np.array([1, 2, 3, 4])
        p.establish_variables(x, y)
        assert len(p.plot_data[0]) == 4

    def test_single_long_indirect_inputs(self):

        p = cat._CategoricalPlotter()

        # Test referencing a DataFrame series in the x variable
        p.establish_variables(x="y", data=self.df)
        npt.assert_equal(p.plot_data, [self.y])
        assert p.orient == "y"
        assert p.value_label == "y"
        assert p.group_label is None

        # Test referencing a DataFrame series in the y variable
        p.establish_variables(y="y", data=self.df)
        npt.assert_equal(p.plot_data, [self.y])
        assert p.orient == "x"
        assert p.value_label == "y"
        assert p.group_label is None

    def test_longform_groupby(self):

        p = cat._CategoricalPlotter()

        # Test a vertically oriented grouped and nested plot
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert len(p.plot_data) == 3
        assert len(p.plot_hues) == 3
        assert p.orient == "x"
        assert p.value_label == "y"
        assert p.group_label == "g"
        assert p.hue_title == "h"

        for group, vals in zip(["a", "b", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        for group, hues in zip(["a", "b", "c"], p.plot_hues):
            npt.assert_array_equal(hues, self.h[self.g == group])

        # Test a grouped and nested plot with direct array value data
        p.establish_variables("g", self.y.values, "h", self.df)
        assert p.value_label is None
        assert p.group_label == "g"

        for group, vals in zip(["a", "b", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        # Test a grouped and nested plot with direct array hue data
        p.establish_variables("g", "y", self.h.values, self.df)

        for group, hues in zip(["a", "b", "c"], p.plot_hues):
            npt.assert_array_equal(hues, self.h[self.g == group])

        # Test categorical grouping data
        df = self.df.copy()
        df.g = df.g.astype("category")

        # Test that horizontal orientation is automatically detected
        p.establish_variables("y", "g", hue="h", data=df)
        assert len(p.plot_data) == 3
        assert len(p.plot_hues) == 3
        assert p.orient == "y"
        assert p.value_label == "y"
        assert p.group_label == "g"
        assert p.hue_title == "h"

        for group, vals in zip(["a", "b", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        for group, hues in zip(["a", "b", "c"], p.plot_hues):
            npt.assert_array_equal(hues, self.h[self.g == group])

        # Test grouped data that matches on index
        p1 = cat._CategoricalPlotter()
        p1.establish_variables(self.g, self.y, hue=self.h)
        p2 = cat._CategoricalPlotter()
        p2.establish_variables(self.g, self.y.iloc[::-1], self.h)
        for i, (d1, d2) in enumerate(zip(p1.plot_data, p2.plot_data)):
            assert np.array_equal(d1.sort_index(), d2.sort_index())

    def test_input_validation(self):

        p = cat._CategoricalPlotter()

        kws = dict(x="g", y="y", hue="h", units="u", data=self.df)
        for var in ["x", "y", "hue", "units"]:
            input_kws = kws.copy()
            input_kws[var] = "bad_input"
            with pytest.raises(ValueError):
                p.establish_variables(**input_kws)

    def test_order(self):

        p = cat._CategoricalPlotter()

        # Test inferred order from a wide dataframe input
        p.establish_variables(data=self.x_df)
        assert p.group_names == ["X", "Y", "Z"]

        # Test specified order with a wide dataframe input
        p.establish_variables(data=self.x_df, order=["Y", "Z", "X"])
        assert p.group_names == ["Y", "Z", "X"]

        for group, vals in zip(["Y", "Z", "X"], p.plot_data):
            npt.assert_array_equal(vals, self.x_df[group])

        with pytest.raises(ValueError):
            p.establish_variables(data=self.x, order=[1, 2, 0])

        # Test inferred order from a grouped longform input
        p.establish_variables("g", "y", data=self.df)
        assert p.group_names == ["a", "b", "c"]

        # Test specified order from a grouped longform input
        p.establish_variables("g", "y", data=self.df, order=["b", "a", "c"])
        assert p.group_names == ["b", "a", "c"]

        for group, vals in zip(["b", "a", "c"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        # Test inferred order from a grouped input with categorical groups
        df = self.df.copy()
        df.g = df.g.astype("category")
        df.g = df.g.cat.reorder_categories(["c", "b", "a"])
        p.establish_variables("g", "y", data=df)
        assert p.group_names == ["c", "b", "a"]

        for group, vals in zip(["c", "b", "a"], p.plot_data):
            npt.assert_array_equal(vals, self.y[self.g == group])

        df.g = (df.g.cat.add_categories("d")
                    .cat.reorder_categories(["c", "b", "d", "a"]))
        p.establish_variables("g", "y", data=df)
        assert p.group_names == ["c", "b", "d", "a"]

    def test_hue_order(self):

        p = cat._CategoricalPlotter()

        # Test inferred hue order
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert p.hue_names == ["m", "n"]

        # Test specified hue order
        p.establish_variables("g", "y", hue="h", data=self.df,
                              hue_order=["n", "m"])
        assert p.hue_names == ["n", "m"]

        # Test inferred hue order from a categorical hue input
        df = self.df.copy()
        df.h = df.h.astype("category")
        df.h = df.h.cat.reorder_categories(["n", "m"])
        p.establish_variables("g", "y", hue="h", data=df)
        assert p.hue_names == ["n", "m"]

        df.h = (df.h.cat.add_categories("o")
                    .cat.reorder_categories(["o", "m", "n"]))
        p.establish_variables("g", "y", hue="h", data=df)
        assert p.hue_names == ["o", "m", "n"]

    def test_plot_units(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert p.plot_units is None

        p.establish_variables("g", "y", hue="h", data=self.df, units="u")
        for group, units in zip(["a", "b", "c"], p.plot_units):
            npt.assert_array_equal(units, self.u[self.g == group])

    def test_default_palettes(self):

        p = cat._CategoricalPlotter()

        # Test palette mapping the x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors(None, None, 1)
        assert p.colors == palettes.color_palette(n_colors=3)

        # Test palette mapping the hue position
        p.establish_variables("g", "y", hue="h", data=self.df)
        p.establish_colors(None, None, 1)
        assert p.colors == palettes.color_palette(n_colors=2)

    def test_default_palette_with_many_levels(self):

        with palettes.color_palette(["blue", "red"], 2):
            p = cat._CategoricalPlotter()
            p.establish_variables("g", "y", data=self.df)
            p.establish_colors(None, None, 1)
            npt.assert_array_equal(p.colors,
                                   palettes.husl_palette(3, l=.7))  # noqa

    def test_specific_color(self):

        p = cat._CategoricalPlotter()

        # Test the same color for each x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors("blue", None, 1)
        blue_rgb = mpl.colors.colorConverter.to_rgb("blue")
        assert p.colors == [blue_rgb] * 3

        # Test a color-based blend for the hue mapping
        p.establish_variables("g", "y", hue="h", data=self.df)
        p.establish_colors("#ff0022", None, 1)
        rgba_array = np.array(palettes.light_palette("#ff0022", 2))
        npt.assert_array_almost_equal(p.colors,
                                      rgba_array[:, :3])

    def test_specific_palette(self):

        p = cat._CategoricalPlotter()

        # Test palette mapping the x position
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors(None, "dark", 1)
        assert p.colors == palettes.color_palette("dark", 3)

        # Test that non-None `color` and `hue` raises an error
        p.establish_variables("g", "y", hue="h", data=self.df)
        p.establish_colors(None, "muted", 1)
        assert p.colors == palettes.color_palette("muted", 2)

        # Test that specified palette overrides specified color
        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors("blue", "deep", 1)
        assert p.colors == palettes.color_palette("deep", 3)

    def test_dict_as_palette(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", hue="h", data=self.df)
        pal = {"m": (0, 0, 1), "n": (1, 0, 0)}
        p.establish_colors(None, pal, 1)
        assert p.colors == [(0, 0, 1), (1, 0, 0)]

    def test_palette_desaturation(self):

        p = cat._CategoricalPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.establish_colors((0, 0, 1), None, .5)
        assert p.colors == [(.25, .25, .75)] * 3

        p.establish_colors(None, [(0, 0, 1), (1, 0, 0), "w"], .5)
        assert p.colors == [(.25, .25, .75), (.75, .25, .25), (1, 1, 1)]


class TestCategoricalStatPlotter(CategoricalFixture):

    def test_no_bootstrappig(self):

        p = cat._CategoricalStatPlotter()
        p.establish_variables("g", "y", data=self.df)
        p.estimate_statistic("mean", None, 100, None)
        npt.assert_array_equal(p.confint, np.array([]))

        p.establish_variables("g", "y", hue="h", data=self.df)
        p.estimate_statistic(np.mean, None, 100, None)
        npt.assert_array_equal(p.confint, np.array([[], [], []]))

    def test_single_layer_stats(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y)
        p.estimate_statistic("mean", ("ci", 95), 10000, None)

        assert p.statistic.shape == (3,)
        assert p.confint.shape == (3, 2)

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby(g).mean())

        for ci, (_, grp_y) in zip(p.confint, y.groupby(g)):
            sem = grp_y.std() / np.sqrt(len(grp_y))
            mean = grp_y.mean()
            half_ci = _normal_quantile_func(.975) * sem
            ci_want = mean - half_ci, mean + half_ci
            npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_single_layer_stats_with_units(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 90))
        y = pd.Series(np.random.RandomState(0).randn(270))
        u = pd.Series(np.repeat(np.tile(list("xyz"), 30), 3))
        y[u == "x"] -= 3
        y[u == "y"] += 3

        p.establish_variables(g, y)
        p.estimate_statistic("mean", ("ci", 95), 10000, None)
        stat1, ci1 = p.statistic, p.confint

        p.establish_variables(g, y, units=u)
        p.estimate_statistic("mean", ("ci", 95), 10000, None)
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
        p.estimate_statistic("mean", ("ci", 95), 10000, None)

        assert p.statistic.shape == (4,)
        assert p.confint.shape == (4, 2)

        rows = g == "b"
        mean = y[rows].mean()
        sem = y[rows].std() / np.sqrt(rows.sum())
        half_ci = _normal_quantile_func(.975) * sem
        ci = mean - half_ci, mean + half_ci
        npt.assert_almost_equal(p.statistic[1], mean)
        npt.assert_array_almost_equal(p.confint[1], ci, 2)

        npt.assert_equal(p.statistic[2], np.nan)
        npt.assert_array_equal(p.confint[2], (np.nan, np.nan))

    def test_nested_stats(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        h = pd.Series(np.tile(list("xy"), 150))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y, h)
        p.estimate_statistic("mean", ("ci", 95), 50000, None)

        assert p.statistic.shape == (3, 2)
        assert p.confint.shape == (3, 2, 2)

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby([g, h]).mean().unstack())

        for ci_g, (_, grp_y) in zip(p.confint, y.groupby(g)):
            for ci, hue_y in zip(ci_g, [grp_y.iloc[::2], grp_y.iloc[1::2]]):
                sem = hue_y.std() / np.sqrt(len(hue_y))
                mean = hue_y.mean()
                half_ci = _normal_quantile_func(.975) * sem
                ci_want = mean - half_ci, mean + half_ci
                npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_bootstrap_seed(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        h = pd.Series(np.tile(list("xy"), 150))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y, h)
        p.estimate_statistic("mean", ("ci", 95), 1000, 0)
        confint_1 = p.confint
        p.estimate_statistic("mean", ("ci", 95), 1000, 0)
        confint_2 = p.confint

        npt.assert_array_equal(confint_1, confint_2)

    def test_nested_stats_with_units(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 90))
        h = pd.Series(np.tile(list("xy"), 135))
        u = pd.Series(np.repeat(list("ijkijk"), 45))
        y = pd.Series(np.random.RandomState(0).randn(270))
        y[u == "i"] -= 3
        y[u == "k"] += 3

        p.establish_variables(g, y, h)
        p.estimate_statistic("mean", ("ci", 95), 10000, None)
        stat1, ci1 = p.statistic, p.confint

        p.establish_variables(g, y, h, units=u)
        p.estimate_statistic("mean", ("ci", 95), 10000, None)
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
        p.estimate_statistic("mean", ("ci", 95), 50000, None)

        assert p.statistic.shape == (4, 3)
        assert p.confint.shape == (4, 3, 2)

        rows = (g == "b") & (h == "x")
        mean = y[rows].mean()
        sem = y[rows].std() / np.sqrt(rows.sum())
        half_ci = _normal_quantile_func(.975) * sem
        ci = mean - half_ci, mean + half_ci
        npt.assert_almost_equal(p.statistic[1, 2], mean)
        npt.assert_array_almost_equal(p.confint[1, 2], ci, 2)

        npt.assert_array_equal(p.statistic[:, 0], [np.nan] * 4)
        npt.assert_array_equal(p.statistic[2], [np.nan] * 3)
        npt.assert_array_equal(p.confint[:, 0],
                               np.zeros((4, 2)) * np.nan)
        npt.assert_array_equal(p.confint[2],
                               np.zeros((3, 2)) * np.nan)

    def test_sd_error_bars(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y)
        p.estimate_statistic(np.mean, "sd", None, None)

        assert p.statistic.shape == (3,)
        assert p.confint.shape == (3, 2)

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby(g).mean())

        for ci, (_, grp_y) in zip(p.confint, y.groupby(g)):
            mean = grp_y.mean()
            half_ci = np.std(grp_y)
            ci_want = mean - half_ci, mean + half_ci
            npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_nested_sd_error_bars(self):

        p = cat._CategoricalStatPlotter()

        g = pd.Series(np.repeat(list("abc"), 100))
        h = pd.Series(np.tile(list("xy"), 150))
        y = pd.Series(np.random.RandomState(0).randn(300))

        p.establish_variables(g, y, h)
        p.estimate_statistic(np.mean, "sd", None, None)

        assert p.statistic.shape == (3, 2)
        assert p.confint.shape == (3, 2, 2)

        npt.assert_array_almost_equal(p.statistic,
                                      y.groupby([g, h]).mean().unstack())

        for ci_g, (_, grp_y) in zip(p.confint, y.groupby(g)):
            for ci, hue_y in zip(ci_g, [grp_y.iloc[::2], grp_y.iloc[1::2]]):
                mean = hue_y.mean()
                half_ci = np.std(hue_y)
                ci_want = mean - half_ci, mean + half_ci
                npt.assert_array_almost_equal(ci_want, ci, 2)

    def test_draw_cis(self):

        p = cat._CategoricalStatPlotter()

        # Test vertical CIs
        p.orient = "x"

        f, ax = plt.subplots()
        at_group = [0, 1]
        confints = [(.5, 1.5), (.25, .8)]
        colors = [".2", ".3"]
        p.draw_confints(ax, at_group, confints, colors)

        lines = ax.lines
        for line, at, ci, c in zip(lines, at_group, confints, colors):
            x, y = line.get_xydata().T
            npt.assert_array_equal(x, [at, at])
            npt.assert_array_equal(y, ci)
            assert line.get_color() == c

        plt.close("all")

        # Test horizontal CIs
        p.orient = "y"

        f, ax = plt.subplots()
        p.draw_confints(ax, at_group, confints, colors)

        lines = ax.lines
        for line, at, ci, c in zip(lines, at_group, confints, colors):
            x, y = line.get_xydata().T
            npt.assert_array_equal(x, ci)
            npt.assert_array_equal(y, [at, at])
            assert line.get_color() == c

        plt.close("all")

        # Test vertical CIs with endcaps
        p.orient = "x"

        f, ax = plt.subplots()
        p.draw_confints(ax, at_group, confints, colors, capsize=0.3)
        capline = ax.lines[len(ax.lines) - 1]
        caplinestart = capline.get_xdata()[0]
        caplineend = capline.get_xdata()[1]
        caplinelength = abs(caplineend - caplinestart)
        assert caplinelength == approx(0.3)
        assert len(ax.lines) == 6

        plt.close("all")

        # Test horizontal CIs with endcaps
        p.orient = "y"

        f, ax = plt.subplots()
        p.draw_confints(ax, at_group, confints, colors, capsize=0.3)
        capline = ax.lines[len(ax.lines) - 1]
        caplinestart = capline.get_ydata()[0]
        caplineend = capline.get_ydata()[1]
        caplinelength = abs(caplineend - caplinestart)
        assert caplinelength == approx(0.3)
        assert len(ax.lines) == 6

        # Test extra keyword arguments
        f, ax = plt.subplots()
        p.draw_confints(ax, at_group, confints, colors, lw=4)
        line = ax.lines[0]
        assert line.get_linewidth() == 4

        plt.close("all")

        # Test errwidth is set appropriately
        f, ax = plt.subplots()
        p.draw_confints(ax, at_group, confints, colors, errwidth=2)
        capline = ax.lines[len(ax.lines) - 1]
        assert capline._linewidth == 2
        assert len(ax.lines) == 2

        plt.close("all")


class TestBoxPlotter(CategoricalFixture):

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       orient=None, color=None, palette=None,
                       saturation=.75, width=.8, dodge=True,
                       fliersize=5, linewidth=None)

    def test_nested_width(self):

        kws = self.default_kws.copy()
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert p.nested_width == .4 * .98

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert p.nested_width == .3 * .98

        kws = self.default_kws.copy()
        kws["dodge"] = False
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        assert p.nested_width == .8

    def test_hue_offsets(self):

        p = cat._BoxPlotter(**self.default_kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.2, .2])

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = cat._BoxPlotter(**kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.15, .15])

        p = cat._BoxPlotter(**kws)
        p.establish_variables("h", "y", "g", data=self.df)
        npt.assert_array_almost_equal(p.hue_offsets, [-.2, 0, .2])

    def test_axes_data(self):

        ax = cat.boxplot(x="g", y="y", data=self.df)
        assert len(self.get_box_artists(ax)) == 3

        plt.close("all")

        ax = cat.boxplot(x="g", y="y", hue="h", data=self.df)
        assert len(self.get_box_artists(ax)) == 6

        plt.close("all")

    def test_box_colors(self):

        ax = cat.boxplot(x="g", y="y", data=self.df, saturation=1)
        pal = palettes.color_palette(n_colors=3)
        assert same_color([patch.get_facecolor() for patch in self.get_box_artists(ax)],
                          pal)

        plt.close("all")

        ax = cat.boxplot(x="g", y="y", hue="h", data=self.df, saturation=1)
        pal = palettes.color_palette(n_colors=2)
        assert same_color([patch.get_facecolor() for patch in self.get_box_artists(ax)],
                          pal * 3)

        plt.close("all")

    def test_draw_missing_boxes(self):

        ax = cat.boxplot(x="g", y="y", data=self.df,
                         order=["a", "b", "c", "d"])
        assert len(self.get_box_artists(ax)) == 3

    def test_missing_data(self):

        x = ["a", "a", "b", "b", "c", "c", "d", "d"]
        h = ["x", "y", "x", "y", "x", "y", "x", "y"]
        y = self.rs.randn(8)
        y[-2:] = np.nan

        ax = cat.boxplot(x=x, y=y)
        assert len(self.get_box_artists(ax)) == 3

        plt.close("all")

        y[-1] = 0
        ax = cat.boxplot(x=x, y=y, hue=h)
        assert len(self.get_box_artists(ax)) == 7

        plt.close("all")

    def test_unaligned_index(self):

        f, (ax1, ax2) = plt.subplots(2)
        cat.boxplot(x=self.g, y=self.y, ax=ax1)
        cat.boxplot(x=self.g, y=self.y_perm, ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert np.array_equal(l1.get_xydata(), l2.get_xydata())

        f, (ax1, ax2) = plt.subplots(2)
        hue_order = self.h.unique()
        cat.boxplot(x=self.g, y=self.y, hue=self.h,
                    hue_order=hue_order, ax=ax1)
        cat.boxplot(x=self.g, y=self.y_perm, hue=self.h,
                    hue_order=hue_order, ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert np.array_equal(l1.get_xydata(), l2.get_xydata())

    def test_boxplots(self):

        # Smoke test the high level boxplot options

        cat.boxplot(x="y", data=self.df)
        plt.close("all")

        cat.boxplot(y="y", data=self.df)
        plt.close("all")

        cat.boxplot(x="g", y="y", data=self.df)
        plt.close("all")

        cat.boxplot(x="y", y="g", data=self.df, orient="h")
        plt.close("all")

        cat.boxplot(x="g", y="y", hue="h", data=self.df)
        plt.close("all")

        cat.boxplot(x="g", y="y", hue="h", order=list("nabc"), data=self.df)
        plt.close("all")

        cat.boxplot(x="g", y="y", hue="h", hue_order=list("omn"), data=self.df)
        plt.close("all")

        cat.boxplot(x="y", y="g", hue="h", data=self.df, orient="h")
        plt.close("all")

    def test_axes_annotation(self):

        ax = cat.boxplot(x="g", y="y", data=self.df)
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        assert ax.get_xlim() == (-.5, 2.5)
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])

        plt.close("all")

        ax = cat.boxplot(x="g", y="y", hue="h", data=self.df)
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])
        npt.assert_array_equal([l.get_text() for l in ax.legend_.get_texts()],
                               ["m", "n"])

        plt.close("all")

        ax = cat.boxplot(x="y", y="g", data=self.df, orient="h")
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        assert ax.get_ylim() == (2.5, -.5)
        npt.assert_array_equal(ax.get_yticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_yticklabels()],
                               ["a", "b", "c"])

        plt.close("all")


class TestViolinPlotter(CategoricalFixture):

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       bw="scott", cut=2, scale="area", scale_hue=True,
                       gridsize=100, width=.8, inner="box", split=False,
                       dodge=True, orient=None, linewidth=None,
                       color=None, palette=None, saturation=.75)

    def test_split_error(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="h", y="y", hue="g", data=self.df, split=True))

        with pytest.raises(ValueError):
            cat._ViolinPlotter(**kws)

    def test_no_observations(self):

        p = cat._ViolinPlotter(**self.default_kws)

        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        y[-1] = np.nan
        p.establish_variables(x, y)
        p.estimate_densities("scott", 2, "area", True, 20)

        assert len(p.support[0]) == 20
        assert len(p.support[1]) == 0

        assert len(p.density[0]) == 20
        assert len(p.density[1]) == 1

        assert p.density[1].item() == 1

        p.estimate_densities("scott", 2, "count", True, 20)
        assert p.density[1].item() == 0

        x = ["a"] * 4 + ["b"] * 2
        y = self.rs.randn(6)
        h = ["m", "n"] * 2 + ["m"] * 2

        p.establish_variables(x, y, hue=h)
        p.estimate_densities("scott", 2, "area", True, 20)

        assert len(p.support[1][0]) == 20
        assert len(p.support[1][1]) == 0

        assert len(p.density[1][0]) == 20
        assert len(p.density[1][1]) == 1

        assert p.density[1][1].item() == 1

        p.estimate_densities("scott", 2, "count", False, 20)
        assert p.density[1][1].item() == 0

    def test_single_observation(self):

        p = cat._ViolinPlotter(**self.default_kws)

        x = ["a", "a", "b"]
        y = self.rs.randn(3)
        p.establish_variables(x, y)
        p.estimate_densities("scott", 2, "area", True, 20)

        assert len(p.support[0]) == 20
        assert len(p.support[1]) == 1

        assert len(p.density[0]) == 20
        assert len(p.density[1]) == 1

        assert p.density[1].item() == 1

        p.estimate_densities("scott", 2, "count", True, 20)
        assert p.density[1].item() == .5

        x = ["b"] * 4 + ["a"] * 3
        y = self.rs.randn(7)
        h = (["m", "n"] * 4)[:-1]

        p.establish_variables(x, y, hue=h)
        p.estimate_densities("scott", 2, "area", True, 20)

        assert len(p.support[1][0]) == 20
        assert len(p.support[1][1]) == 1

        assert len(p.density[1][0]) == 20
        assert len(p.density[1][1]) == 1

        assert p.density[1][1].item() == 1

        p.estimate_densities("scott", 2, "count", False, 20)
        assert p.density[1][1].item() == .5

    def test_dwidth(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="g", y="y", data=self.df))

        p = cat._ViolinPlotter(**kws)
        assert p.dwidth == .4

        kws.update(dict(width=.4))
        p = cat._ViolinPlotter(**kws)
        assert p.dwidth == .2

        kws.update(dict(hue="h", width=.8))
        p = cat._ViolinPlotter(**kws)
        assert p.dwidth == .2

        kws.update(dict(split=True))
        p = cat._ViolinPlotter(**kws)
        assert p.dwidth == .4

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
        assert max_after[0] == 1

        before_ratio = max_before[1] / max_before[0]
        after_ratio = max_after[1] / max_after[0]
        assert before_ratio == after_ratio

        # Test nested grouping scaling across all densities
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)],
                   [self.rs.uniform(0, .1, 50), self.rs.uniform(0, .02, 50)]]

        max_before = np.array([[r.max() for r in row] for row in density])
        p.scale_area(density, max_before, False)
        max_after = np.array([[r.max() for r in row] for row in density])
        assert max_after[0, 0] == 1

        before_ratio = max_before[1, 1] / max_before[0, 0]
        after_ratio = max_after[1, 1] / max_after[0, 0]
        assert before_ratio == after_ratio

        # Test nested grouping scaling within hue
        p.hue_names = ["foo", "bar"]
        density = [[self.rs.uniform(0, .8, 50), self.rs.uniform(0, .2, 50)],
                   [self.rs.uniform(0, .1, 50), self.rs.uniform(0, .02, 50)]]

        max_before = np.array([[r.max() for r in row] for row in density])
        p.scale_area(density, max_before, True)
        max_after = np.array([[r.max() for r in row] for row in density])
        assert max_after[0, 0] == 1
        assert max_after[1, 0] == 1

        before_ratio = max_before[1, 1] / max_before[1, 0]
        after_ratio = max_after[1, 1] / max_after[1, 0]
        assert before_ratio == after_ratio

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
        with pytest.raises(ValueError):
            cat._ViolinPlotter(**kws)

    def test_kde_fit(self):

        p = cat._ViolinPlotter(**self.default_kws)
        data = self.y
        data_std = data.std(ddof=1)

        # Test reference rule bandwidth
        kde, bw = p.fit_kde(data, "scott")
        assert kde.factor == kde.scotts_factor()
        assert bw == kde.scotts_factor() * data_std

        # Test numeric scale factor
        kde, bw = p.fit_kde(self.y, .2)
        assert kde.factor == .2
        assert bw == .2 * data_std

    def test_draw_to_density(self):

        p = cat._ViolinPlotter(**self.default_kws)
        # p.dwidth will be 1 for easier testing
        p.width = 2

        # Test vertical plots
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
        p.draw_box_lines(ax, self.y, 0)
        assert len(ax.lines) == 2

        q25, q50, q75 = np.percentile(self.y, [25, 50, 75])
        _, y = ax.lines[1].get_xydata().T
        npt.assert_array_equal(y, [q25, q75])

        _, y = ax.collections[0].get_offsets().T
        assert y == q50

        plt.close("all")

        # Test horizontal plot
        kws = self.default_kws.copy()
        kws.update(dict(x="y", data=self.df, inner=None))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_box_lines(ax, self.y, 0)
        assert len(ax.lines) == 2

        q25, q50, q75 = np.percentile(self.y, [25, 50, 75])
        x, _ = ax.lines[1].get_xydata().T
        npt.assert_array_equal(x, [q25, q75])

        x, _ = ax.collections[0].get_offsets().T
        assert x == q50

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

    def test_validate_inner(self):

        kws = self.default_kws.copy()
        kws.update(dict(inner="bad_inner"))
        with pytest.raises(ValueError):
            cat._ViolinPlotter(**kws)

    def test_draw_violinplots(self):

        kws = self.default_kws.copy()

        # Test single vertical violin
        kws.update(dict(y="y", data=self.df, inner=None,
                        saturation=1, color=(1, 0, 0, 1)))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 1
        npt.assert_array_equal(ax.collections[0].get_facecolors(),
                               [(1, 0, 0, 1)])
        plt.close("all")

        # Test single horizontal violin
        kws.update(dict(x="y", y=None, color=(0, 1, 0, 1)))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 1
        npt.assert_array_equal(ax.collections[0].get_facecolors(),
                               [(0, 1, 0, 1)])
        plt.close("all")

        # Test multiple vertical violins
        kws.update(dict(x="g", y="y", color=None,))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 3
        for violin, color in zip(ax.collections, palettes.color_palette()):
            npt.assert_array_equal(violin.get_facecolors()[0, :-1], color)
        plt.close("all")

        # Test multiple violins with hue nesting
        kws.update(dict(hue="h"))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 6
        for violin, color in zip(ax.collections,
                                 palettes.color_palette(n_colors=2) * 3):
            npt.assert_array_equal(violin.get_facecolors()[0, :-1], color)
        plt.close("all")

        # Test multiple split violins
        kws.update(dict(split=True, palette="muted"))
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 6
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
        assert len(ax.collections) == 1
        assert len(ax.lines) == 0
        plt.close("all")

        # Test nested hue grouping
        x = ["a"] * 4 + ["b"] * 2
        y = self.rs.randn(6)
        h = ["m", "n"] * 2 + ["m"] * 2
        kws.update(x=x, y=y, hue=h)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 3
        assert len(ax.lines) == 0
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
        assert len(ax.collections) == 1
        assert len(ax.lines) == 1
        plt.close("all")

        # Test nested hue grouping
        x = ["b"] * 4 + ["a"] * 3
        y = self.rs.randn(7)
        h = (["m", "n"] * 4)[:-1]
        kws.update(x=x, y=y, hue=h)
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 3
        assert len(ax.lines) == 1
        plt.close("all")

        # Test nested hue grouping with split
        kws["split"] = True
        p = cat._ViolinPlotter(**kws)

        _, ax = plt.subplots()
        p.draw_violins(ax)
        assert len(ax.collections) == 3
        assert len(ax.lines) == 1
        plt.close("all")

    def test_violinplots(self):

        # Smoke test the high level violinplot options

        cat.violinplot(x="y", data=self.df)
        plt.close("all")

        cat.violinplot(y="y", data=self.df)
        plt.close("all")

        cat.violinplot(x="g", y="y", data=self.df)
        plt.close("all")

        cat.violinplot(x="y", y="g", data=self.df, orient="h")
        plt.close("all")

        cat.violinplot(x="g", y="y", hue="h", data=self.df)
        plt.close("all")

        order = list("nabc")
        cat.violinplot(x="g", y="y", hue="h", order=order, data=self.df)
        plt.close("all")

        order = list("omn")
        cat.violinplot(x="g", y="y", hue="h", hue_order=order, data=self.df)
        plt.close("all")

        cat.violinplot(x="y", y="g", hue="h", data=self.df, orient="h")
        plt.close("all")

        for inner in ["box", "quart", "point", "stick", None]:
            cat.violinplot(x="g", y="y", data=self.df, inner=inner)
            plt.close("all")

            cat.violinplot(x="g", y="y", hue="h", data=self.df, inner=inner)
            plt.close("all")

            cat.violinplot(x="g", y="y", hue="h", data=self.df,
                           inner=inner, split=True)
            plt.close("all")

    def test_split_one_each(self, rng):

        x = np.repeat([0, 1], 5)
        y = rng.normal(0, 1, 10)
        ax = cat.violinplot(x=x, y=y, hue=x, split=True, inner="box")
        assert len(ax.lines) == 4


# ====================================================================================
# ====================================================================================


class SharedAxesLevelTests:

    @pytest.fixture
    def common_kws(self):
        return {}

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_labels_long(self, long_df, orient):

        depend = {"x": "y", "y": "x"}[orient]
        kws = {orient: "a", depend: "y", "hue": "b"}

        ax = self.func(long_df, **kws)
        assert getattr(ax, f"get_{orient}label")() == kws[orient]
        assert getattr(ax, f"get_{depend}label")() == kws[depend]

        get_ori_labels = getattr(ax, f"get_{orient}ticklabels")
        ori_labels = [t.get_text() for t in get_ori_labels()]
        ori_levels = categorical_order(long_df[kws[orient]])
        assert ori_labels == ori_levels

        legend = ax.get_legend()
        assert legend.get_title().get_text() == kws["hue"]

        hue_labels = [t.get_text() for t in legend.texts]
        hue_levels = categorical_order(long_df[kws["hue"]])
        assert hue_labels == hue_levels

    def test_labels_wide(self, wide_df):

        wide_df = wide_df.rename_axis("cols", axis=1)
        ax = self.func(wide_df)
        assert ax.get_xlabel() == wide_df.columns.name
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label, level in zip(labels, wide_df.columns):
            assert label == level

    def test_color(self, long_df, common_kws):
        common_kws.update(data=long_df, x="a", y="y")

        ax = plt.figure().subplots()
        self.func(ax=ax, **common_kws)
        assert self.get_last_color(ax) == to_rgba("C0")

        ax = plt.figure().subplots()
        self.func(ax=ax, **common_kws)
        self.func(ax=ax, **common_kws)
        assert self.get_last_color(ax) == to_rgba("C1")

        ax = plt.figure().subplots()
        self.func(color="C2", ax=ax, **common_kws)
        assert self.get_last_color(ax) == to_rgba("C2")

        ax = plt.figure().subplots()
        self.func(color="C3", ax=ax, **common_kws)
        assert self.get_last_color(ax) == to_rgba("C3")

    def test_two_calls(self):

        ax = plt.figure().subplots()
        self.func(x=["a", "b", "c"], y=[1, 2, 3], ax=ax)
        self.func(x=["e", "f"], y=[4, 5], ax=ax)
        assert ax.get_xlim() == (-.5, 4.5)


class SharedScatterTests(SharedAxesLevelTests):
    """Tests functionality common to stripplot and swarmplot."""

    def get_last_color(self, ax):

        colors = ax.collections[-1].get_facecolors()
        unique_colors = np.unique(colors, axis=0)
        assert len(unique_colors) == 1
        return to_rgba(unique_colors.squeeze())

    # ------------------------------------------------------------------------------

    def test_color(self, long_df, common_kws):

        super().test_color(long_df, common_kws)

        ax = plt.figure().subplots()
        self.func(data=long_df, x="a", y="y", facecolor="C4", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C4")

        ax = plt.figure().subplots()
        self.func(data=long_df, x="a", y="y", fc="C5", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C5")

    def test_supplied_color_array(self, long_df):

        cmap = get_colormap("Blues")
        norm = mpl.colors.Normalize()
        colors = cmap(norm(long_df["y"].to_numpy()))

        keys = ["c", "fc", "facecolor", "facecolors"]

        for key in keys:

            ax = plt.figure().subplots()
            self.func(x=long_df["y"], **{key: colors})
            _draw_figure(ax.figure)
            assert_array_equal(ax.collections[0].get_facecolors(), colors)

        ax = plt.figure().subplots()
        self.func(x=long_df["y"], c=long_df["y"], cmap=cmap)
        _draw_figure(ax.figure)
        assert_array_equal(ax.collections[0].get_facecolors(), colors)

    @pytest.mark.parametrize(
        "orient,data_type", [
            ("h", "dataframe"), ("h", "dict"),
            ("v", "dataframe"), ("v", "dict"),
        ]
    )
    def test_wide(self, wide_df, orient, data_type):

        if data_type == "dict":
            wide_df = {k: v.to_numpy() for k, v in wide_df.items()}

        ax = self.func(data=wide_df, orient=orient)
        _draw_figure(ax.figure)
        palette = color_palette()

        cat_idx = 0 if orient == "x" else 1
        val_idx = int(not cat_idx)

        axis_objs = ax.xaxis, ax.yaxis
        cat_axis = axis_objs[cat_idx]

        for i, label in enumerate(cat_axis.get_majorticklabels()):

            key = label.get_text()
            points = ax.collections[i]
            point_pos = points.get_offsets().T
            val_pos = point_pos[val_idx]
            cat_pos = point_pos[cat_idx]

            assert_array_equal(cat_pos.round(), i)
            assert_array_equal(val_pos, wide_df[key])

            for point_color in points.get_facecolors():
                assert tuple(point_color) == to_rgba(palette[i])

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_flat(self, flat_series, orient):

        ax = self.func(data=flat_series, orient=orient)
        _draw_figure(ax.figure)

        cat_idx = ["v", "h"].index(orient)
        val_idx = int(not cat_idx)

        points = ax.collections[0]
        pos = points.get_offsets().T

        assert_array_equal(pos[cat_idx].round(), np.zeros(len(flat_series)))
        assert_array_equal(pos[val_idx], flat_series)

    @pytest.mark.parametrize(
        "variables,orient",
        [
            # Order matters for assigning to x/y
            ({"cat": "a", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "a", "hue": None}, None),
            ({"cat": "a", "val": "y", "hue": "a"}, None),
            ({"val": "y", "cat": "a", "hue": "a"}, None),
            ({"cat": "a", "val": "y", "hue": "b"}, None),
            ({"val": "y", "cat": "a", "hue": "x"}, None),
            ({"cat": "s", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "s", "hue": None}, "h"),
            ({"cat": "a", "val": "b", "hue": None}, None),
            ({"val": "a", "cat": "b", "hue": None}, "h"),
            ({"cat": "a", "val": "t", "hue": None}, None),
            ({"val": "t", "cat": "a", "hue": None}, None),
            ({"cat": "d", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "d", "hue": None}, None),
            ({"cat": "a_cat", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "s_cat", "hue": None}, None),
        ],
    )
    def test_positions(self, long_df, variables, orient):

        cat_var = variables["cat"]
        val_var = variables["val"]
        hue_var = variables["hue"]
        var_names = list(variables.values())
        x_var, y_var, *_ = var_names

        ax = self.func(
            data=long_df, x=x_var, y=y_var, hue=hue_var, orient=orient,
        )

        _draw_figure(ax.figure)

        cat_idx = var_names.index(cat_var)
        val_idx = var_names.index(val_var)

        axis_objs = ax.xaxis, ax.yaxis
        cat_axis = axis_objs[cat_idx]
        val_axis = axis_objs[val_idx]

        cat_data = long_df[cat_var]
        cat_levels = categorical_order(cat_data)

        for i, label in enumerate(cat_levels):

            vals = long_df.loc[cat_data == label, val_var]

            points = ax.collections[i].get_offsets().T
            cat_pos = points[var_names.index(cat_var)]
            val_pos = points[var_names.index(val_var)]

            assert_array_equal(val_pos, val_axis.convert_units(vals))
            assert_array_equal(cat_pos.round(), i)
            assert 0 <= np.ptp(cat_pos) <= .8

            label = pd.Index([label]).astype(str)[0]
            assert cat_axis.get_majorticklabels()[i].get_text() == label

    @pytest.mark.parametrize(
        "variables",
        [
            # Order matters for assigning to x/y
            {"cat": "a", "val": "y", "hue": "b"},
            {"val": "y", "cat": "a", "hue": "c"},
            {"cat": "a", "val": "y", "hue": "f"},
        ],
    )
    def test_positions_dodged(self, long_df, variables):

        cat_var = variables["cat"]
        val_var = variables["val"]
        hue_var = variables["hue"]
        var_names = list(variables.values())
        x_var, y_var, *_ = var_names

        ax = self.func(
            data=long_df, x=x_var, y=y_var, hue=hue_var, dodge=True,
        )

        cat_vals = categorical_order(long_df[cat_var])
        hue_vals = categorical_order(long_df[hue_var])

        n_hue = len(hue_vals)
        offsets = np.linspace(0, .8, n_hue + 1)[:-1]
        offsets -= offsets.mean()
        nest_width = .8 / n_hue

        for i, cat_val in enumerate(cat_vals):
            for j, hue_val in enumerate(hue_vals):
                rows = (long_df[cat_var] == cat_val) & (long_df[hue_var] == hue_val)
                vals = long_df.loc[rows, val_var]

                points = ax.collections[n_hue * i + j].get_offsets().T
                cat_pos = points[var_names.index(cat_var)]
                val_pos = points[var_names.index(val_var)]

                if pd.api.types.is_datetime64_any_dtype(vals):
                    vals = mpl.dates.date2num(vals)

                assert_array_equal(val_pos, vals)

                assert_array_equal(cat_pos.round(), i)
                assert_array_equal((cat_pos - (i + offsets[j])).round() / nest_width, 0)
                assert 0 <= np.ptp(cat_pos) <= nest_width

    @pytest.mark.parametrize("cat_var", ["a", "s", "d"])
    def test_positions_unfixed(self, long_df, cat_var):

        long_df = long_df.sort_values(cat_var)

        kws = dict(size=.001)
        if "stripplot" in str(self.func):  # can't use __name__ with partial
            kws["jitter"] = False

        ax = self.func(data=long_df, x=cat_var, y="y", native_scale=True, **kws)

        for i, (cat_level, cat_data) in enumerate(long_df.groupby(cat_var)):

            points = ax.collections[i].get_offsets().T
            cat_pos = points[0]
            val_pos = points[1]

            assert_array_equal(val_pos, cat_data["y"])

            comp_level = np.squeeze(ax.xaxis.convert_units(cat_level)).item()
            assert_array_equal(cat_pos.round(), comp_level)

    @pytest.mark.parametrize(
        "x_type,order",
        [
            (str, None),
            (str, ["a", "b", "c"]),
            (str, ["c", "a"]),
            (str, ["a", "b", "c", "d"]),
            (int, None),
            (int, [3, 1, 2]),
            (int, [3, 1]),
            (int, [1, 2, 3, 4]),
            (int, ["3", "1", "2"]),
        ]
    )
    def test_order(self, x_type, order):

        if x_type is str:
            x = ["b", "a", "c"]
        else:
            x = [2, 1, 3]
        y = [1, 2, 3]

        ax = self.func(x=x, y=y, order=order)
        _draw_figure(ax.figure)

        if order is None:
            order = x
            if x_type is int:
                order = np.sort(order)

        assert len(ax.collections) == len(order)
        tick_labels = ax.xaxis.get_majorticklabels()

        assert ax.get_xlim()[1] == (len(order) - .5)

        for i, points in enumerate(ax.collections):
            cat = order[i]
            assert tick_labels[i].get_text() == str(cat)

            positions = points.get_offsets()
            if x_type(cat) in x:
                val = y[x.index(x_type(cat))]
                assert positions[0, 1] == val
            else:
                assert not positions.size

    @pytest.mark.parametrize("hue_var", ["a", "b"])
    def test_hue_categorical(self, long_df, hue_var):

        cat_var = "b"

        hue_levels = categorical_order(long_df[hue_var])
        cat_levels = categorical_order(long_df[cat_var])

        pal_name = "muted"
        palette = dict(zip(hue_levels, color_palette(pal_name)))
        ax = self.func(data=long_df, x=cat_var, y="y", hue=hue_var, palette=pal_name)

        for i, level in enumerate(cat_levels):

            sub_df = long_df[long_df[cat_var] == level]
            point_hues = sub_df[hue_var]

            points = ax.collections[i]
            point_colors = points.get_facecolors()

            assert len(point_hues) == len(point_colors)

            for hue, color in zip(point_hues, point_colors):
                assert tuple(color) == to_rgba(palette[hue])

    @pytest.mark.parametrize("hue_var", ["a", "b"])
    def test_hue_dodged(self, long_df, hue_var):

        ax = self.func(data=long_df, x="y", y="a", hue=hue_var, dodge=True)
        colors = color_palette(n_colors=long_df[hue_var].nunique())
        collections = iter(ax.collections)

        # Slightly awkward logic to handle challenges of how the artists work.
        # e.g. there are empty scatter collections but the because facecolors
        # for the empty collections will return the default scatter color
        while colors:
            points = next(collections)
            if points.get_offsets().any():
                face_color = tuple(points.get_facecolors()[0])
                expected_color = to_rgba(colors.pop(0))
                assert face_color == expected_color

    @pytest.mark.parametrize(
        "val_var,val_col,hue_col",
        list(itertools.product(["x", "y"], ["b", "y", "t"], [None, "a"])),
    )
    def test_single(self, long_df, val_var, val_col, hue_col):

        var_kws = {val_var: val_col, "hue": hue_col}
        ax = self.func(data=long_df, **var_kws)
        _draw_figure(ax.figure)

        axis_vars = ["x", "y"]
        val_idx = axis_vars.index(val_var)
        cat_idx = int(not val_idx)
        cat_var = axis_vars[cat_idx]

        cat_axis = getattr(ax, f"{cat_var}axis")
        val_axis = getattr(ax, f"{val_var}axis")

        points = ax.collections[0]
        point_pos = points.get_offsets().T
        cat_pos = point_pos[cat_idx]
        val_pos = point_pos[val_idx]

        assert_array_equal(cat_pos.round(), 0)
        assert cat_pos.max() <= .4
        assert cat_pos.min() >= -.4

        num_vals = val_axis.convert_units(long_df[val_col])
        assert_array_equal(val_pos, num_vals)

        if hue_col is not None:
            palette = dict(zip(
                categorical_order(long_df[hue_col]), color_palette()
            ))

        facecolors = points.get_facecolors()
        for i, color in enumerate(facecolors):
            if hue_col is None:
                assert tuple(color) == to_rgba("C0")
            else:
                hue_level = long_df.loc[i, hue_col]
                expected_color = palette[hue_level]
                assert tuple(color) == to_rgba(expected_color)

        ticklabels = cat_axis.get_majorticklabels()
        assert len(ticklabels) == 1
        assert not ticklabels[0].get_text()

    def test_attributes(self, long_df):

        kwargs = dict(
            size=2,
            linewidth=1,
            edgecolor="C2",
        )

        ax = self.func(x=long_df["y"], **kwargs)
        points, = ax.collections

        assert points.get_sizes().item() == kwargs["size"] ** 2
        assert points.get_linewidths().item() == kwargs["linewidth"]
        assert tuple(points.get_edgecolors().squeeze()) == to_rgba(kwargs["edgecolor"])

    def test_three_points(self):

        x = np.arange(3)
        ax = self.func(x=x)
        for point_color in ax.collections[0].get_facecolor():
            assert tuple(point_color) == to_rgba("C0")

    def test_legend_categorical(self, long_df):

        ax = self.func(data=long_df, x="y", y="a", hue="b")
        legend_texts = [t.get_text() for t in ax.legend_.texts]
        expected = categorical_order(long_df["b"])
        assert legend_texts == expected

    def test_legend_numeric(self, long_df):

        ax = self.func(data=long_df, x="y", y="a", hue="z")
        vals = [float(t.get_text()) for t in ax.legend_.texts]
        assert (vals[1] - vals[0]) == approx(vals[2] - vals[1])

    def test_legend_disabled(self, long_df):

        ax = self.func(data=long_df, x="y", y="a", hue="b", legend=False)
        assert ax.legend_ is None

    def test_palette_from_color_deprecation(self, long_df):

        color = (.9, .4, .5)
        hex_color = mpl.colors.to_hex(color)

        hue_var = "a"
        n_hue = long_df[hue_var].nunique()
        palette = color_palette(f"dark:{hex_color}", n_hue)

        with pytest.warns(FutureWarning, match="Setting a gradient palette"):
            ax = self.func(data=long_df, x="z", hue=hue_var, color=color)

        points = ax.collections[0]
        for point_color in points.get_facecolors():
            assert to_rgb(point_color) in palette

    def test_palette_with_hue_deprecation(self, long_df):
        palette = "Blues"
        with pytest.warns(FutureWarning, match="Passing `palette` without"):
            ax = self.func(data=long_df, x="a", y=long_df["y"], palette=palette)
        strips = ax.collections
        colors = color_palette(palette, len(strips))
        for strip, color in zip(strips, colors):
            assert same_color(strip.get_facecolor()[0], color)

    def test_log_scale(self):

        x = [1, 10, 100, 1000]

        ax = plt.figure().subplots()
        ax.set_xscale("log")
        self.func(x=x)
        vals = ax.collections[0].get_offsets()[:, 0]
        assert_array_equal(x, vals)

        y = [1, 2, 3, 4]

        ax = plt.figure().subplots()
        ax.set_xscale("log")
        self.func(x=x, y=y, native_scale=True)
        for i, point in enumerate(ax.collections):
            val = point.get_offsets()[0, 0]
            assert val == approx(x[i])

        x = y = np.ones(100)

        ax = plt.figure().subplots()
        ax.set_yscale("log")
        self.func(x=x, y=y, orient="h", native_scale=True)
        cat_points = ax.collections[0].get_offsets().copy()[:, 1]
        assert np.ptp(np.log10(cat_points)) <= .8

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="long", x="x", color="C3"),
            dict(data="long", y="y", hue="a", jitter=False),
            dict(data="long", x="a", y="y", hue="z", edgecolor="w", linewidth=.5),
            dict(data="long", x="a_cat", y="y", hue="z"),
            dict(data="long", x="y", y="s", hue="c", orient="h", dodge=True),
            dict(data="long", x="s", y="y", hue="c", native_scale=True),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, kwargs):

        kwargs = kwargs.copy()
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df

        try:
            name = self.func.__name__[:-4]
        except AttributeError:
            name = self.func.func.__name__[:-4]
        if name == "swarm":
            kwargs.pop("jitter", None)

        np.random.seed(0)  # for jitter
        ax = self.func(**kwargs)

        np.random.seed(0)
        g = catplot(**kwargs, kind=name)

        assert_plots_equal(ax, g.ax)

    def test_empty_palette(self):
        self.func(x=[], y=[], hue=[], palette=[])


class TestStripPlot(SharedScatterTests):

    func = staticmethod(stripplot)

    def test_jitter_unfixed(self, long_df):

        ax1, ax2 = plt.figure().subplots(2)
        kws = dict(data=long_df, x="y", orient="h", native_scale=True)

        np.random.seed(0)
        stripplot(**kws, y="s", ax=ax1)

        np.random.seed(0)
        stripplot(**kws, y=long_df["s"] * 2, ax=ax2)

        p1 = ax1.collections[0].get_offsets()[1]
        p2 = ax2.collections[0].get_offsets()[1]

        assert p2.std() > p1.std()

    @pytest.mark.parametrize(
        "orient,jitter",
        itertools.product(["v", "h"], [True, .1]),
    )
    def test_jitter(self, long_df, orient, jitter):

        cat_var, val_var = "a", "y"
        if orient == "x":
            x_var, y_var = cat_var, val_var
            cat_idx, val_idx = 0, 1
        else:
            x_var, y_var = val_var, cat_var
            cat_idx, val_idx = 1, 0

        cat_vals = categorical_order(long_df[cat_var])

        ax = stripplot(
            data=long_df, x=x_var, y=y_var, jitter=jitter,
        )

        if jitter is True:
            jitter_range = .4
        else:
            jitter_range = 2 * jitter

        for i, level in enumerate(cat_vals):

            vals = long_df.loc[long_df[cat_var] == level, val_var]
            points = ax.collections[i].get_offsets().T
            cat_points = points[cat_idx]
            val_points = points[val_idx]

            assert_array_equal(val_points, vals)
            assert np.std(cat_points) > 0
            assert np.ptp(cat_points) <= jitter_range


class TestSwarmPlot(SharedScatterTests):

    func = staticmethod(partial(swarmplot, warn_thresh=1))


class SharedAggTests(SharedAxesLevelTests):

    def test_labels_flat(self):

        ind = pd.Index(["a", "b", "c"], name="x")
        ser = pd.Series([1, 2, 3], ind, name="y")

        ax = self.func(ser)
        assert ax.get_xlabel() == ind.name
        assert ax.get_ylabel() == ser.name
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label, level in zip(labels, ind):
            assert label == level


class TestBarPlot(SharedAggTests):

    func = staticmethod(barplot)

    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    def get_last_color(self, ax):

        colors = [p.get_facecolor() for p in ax.containers[-1]]
        unique_colors = np.unique(colors, axis=0)
        assert len(unique_colors) == 1
        return to_rgba(unique_colors.squeeze())

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_single_var(self, orient):

        vals = pd.Series([1, 3, 10])
        ax = barplot(**{orient: vals})
        bar, = ax.patches
        prop = {"x": "width", "y": "height"}[orient]
        assert getattr(bar, f"get_{prop}")() == approx(vals.mean())

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_wide_df(self, wide_df, orient):

        ax = barplot(wide_df, orient=orient)
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        prop = {"x": "height", "y": "width"}[orient]
        for i, bar in enumerate(ax.patches):
            assert getattr(bar, f"get_{prop}")() == approx(wide_df.iloc[:, i].mean())

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_orient(self, orient):

        keys, vals = ["a", "b", "c"], [1, 2, 3]
        data = dict(zip(keys, vals))
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        prop = {"x": "height", "y": "width"}[orient]
        ax = barplot(data, orient=orient)
        for i, bar in enumerate(ax.patches):
            assert getattr(bar, f"get_{orient}")() == approx(i - 0.4)
            assert getattr(bar, f"get_{prop}")() == approx(vals[i])

    def test_xy_vertical(self):

        x = ["a", "b", "c"]
        y = [1, 3, 2.5]

        ax = barplot(x=x, y=y)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == i
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == 0.8

    def test_xy_horizontal(self):

        x = [1, 3, 2.5]
        y = ["a", "b", "c"]

        ax = barplot(x=x, y=y)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == 0
            assert bar.get_y() + bar.get_height() / 2 == i
            assert bar.get_height() == 0.8
            assert bar.get_width() == x[i]

    def test_hue_redundant(self):

        x = ["a", "b", "c"]
        y = [1, 2, 3]
        hue = ["a", "b", "c"]

        ax = barplot(x=x, y=y, hue=hue, saturation=1)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i}")

    def test_hue_matched(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        hue = ["x", "x", "y"]

        ax = barplot(x=x, y=y, hue=hue, saturation=1)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    def test_hue_matched_by_name(self):

        data = {"x": ["a", "b", "c"], "y": [1, 2, 3]}
        ax = barplot(data, x="x", y="y", hue="x", saturation=1)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == data["y"][i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i}")

    def test_hue_dodged(self):

        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, saturation=1)
        for i, bar in enumerate(ax.patches):
            sign = 1 if i // 2 else -1
            assert (
                bar.get_x() + bar.get_width() / 2
                == approx(i % 2 + sign * 0.8 / 4)
            )
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8 / 2)
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    def test_hue_undodged(self):

        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, saturation=1, dodge=False)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i % 2)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    def test_xy_native_scale(self):

        x = [2, 4, 8]
        y = [1, 2, 3]

        ax = barplot(x=x, y=y, native_scale=True)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(x[i])
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8 * 2)

    def test_dateimte_native_scale_axis(self):

        x = pd.date_range("2010-01-01", periods=20, freq="m")
        y = np.arange(20)
        ax = barplot(x=x, y=y, native_scale=True)
        assert "Date" in ax.xaxis.get_major_locator().__class__.__name__
        day = "2003-02-28"
        assert_array_equal(ax.xaxis.convert_units([day]), mpl.dates.date2num([day]))

    def test_hue_native_scale_dodged(self):

        x = [2, 4, 2, 4]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, native_scale=True)
        for i, bar in enumerate(ax.patches):
            for i, bar in enumerate(ax.patches):
                sign = 1 if i // 2 else -1
                assert (
                    bar.get_x() + bar.get_width() / 2
                    == approx(x[i % 2] + sign * 0.8 / 4)
                )
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8 * 2 / 2)

    def test_estimate_default(self, long_df):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].mean()

        ax = barplot(long_df, x=agg_var, y=val_var, errorbar=None)
        order = categorical_order(long_df[agg_var])
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_estimate_string(self, long_df):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].median()

        ax = barplot(long_df, x=agg_var, y=val_var, estimator="median", errorbar=None)
        order = categorical_order(long_df[agg_var])
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_estimate_func(self, long_df):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].median()

        ax = barplot(long_df, x=agg_var, y=val_var, estimator=np.median, errorbar=None)
        order = categorical_order(long_df[agg_var])
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_errorbar(self, long_df):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].agg(["mean", "std"])

        ax = barplot(long_df, x=agg_var, y=val_var, errorbar="sd")
        order = categorical_order(long_df[agg_var])
        for i, line in enumerate(ax.lines):
            row = agg_df.loc[order[i]]
            lo, hi = line.get_ydata()
            assert lo == approx(row["mean"] - row["std"])
            assert hi == approx(row["mean"] + row["std"])

    def test_width(self):

        width = .5
        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = barplot(x=x, y=y, width=width)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_width() == width

    def test_width_native_scale(self):

        width = .5
        x, y = [4, 6, 10], [1, 2, 3]
        ax = barplot(x=x, y=y, width=width, native_scale=True)
        for i, bar in enumerate(ax.patches):
            assert bar.get_width() == (width * 2)

    def test_legend_numeric_auto(self, long_df):

        ax = barplot(long_df, x="x", y="y", hue="x")
        assert len(ax.get_legend().texts) <= 6

    def test_legend_numeric_full(self, long_df):

        ax = barplot(long_df, x="x", y="y", hue="x", legend="full")
        labels = [t.get_text() for t in ax.get_legend().texts]
        levels = [str(x) for x in sorted(long_df["x"].unique())]
        assert labels == levels

    def test_legend_disabled(self, long_df):

        ax = barplot(long_df, x="x", y="y", hue="b", legend=False)
        assert ax.get_legend() is None

    def test_hue_implied_by_palette(self):

        x = ["a", "b", "c"]
        y = [1, 2, 3]
        palette = "Set1"
        colors = color_palette(palette, len(x))
        msg = "Passing `palette` without assigning `hue` is deprecated."
        with pytest.warns(FutureWarning, match=msg):
            ax = barplot(x=x, y=y, saturation=1, palette=palette)
        for i, bar in enumerate(ax.patches):
            assert same_color(bar.get_facecolor(), colors[i])


class TestBarPlotter(CategoricalFixture):

    default_kws = dict(
        data=None, x=None, y=None, hue=None, units=None,
        estimator="mean", errorbar=("ci", 95), n_boot=100, seed=None,
        order=None, hue_order=None,
        orient=None, color=None, palette=None,
        saturation=.75, width=0.8,
        errcolor=".26", errwidth=None,
        capsize=None, dodge=True
    )

    def test_nested_width(self):

        ax = cat.barplot(data=self.df, x="g", y="y", hue="h")
        for bar in ax.patches:
            assert bar.get_width() == approx(.8 / 2)
        ax.clear()

        ax = cat.barplot(data=self.df, x="g", y="y", hue="g", width=.5)
        for bar in ax.patches:
            assert bar.get_width() == approx(.5)
        ax.clear()

        ax = cat.barplot(data=self.df, x="g", y="y", hue="g", dodge=False)
        for bar in ax.patches:
            assert bar.get_width() == approx(.8)
        ax.clear()

    def test_draw_vertical_bars(self):

        kws = self.default_kws.copy()
        kws.update(x="g", y="y", data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        assert len(ax.patches) == len(p.plot_data)
        assert len(ax.lines) == len(p.plot_data)

        for bar, color in zip(ax.patches, p.colors):
            assert bar.get_facecolor()[:-1] == color

        positions = np.arange(len(p.plot_data)) - p.width / 2
        for bar, pos, stat in zip(ax.patches, positions, p.statistic):
            assert bar.get_x() == pos
            assert bar.get_width() == p.width
            assert bar.get_y() == 0
            assert bar.get_height() == stat

    def test_draw_horizontal_bars(self):

        kws = self.default_kws.copy()
        kws.update(x="y", y="g", orient="h", data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        assert len(ax.patches) == len(p.plot_data)
        assert len(ax.lines) == len(p.plot_data)

        for bar, color in zip(ax.patches, p.colors):
            assert bar.get_facecolor()[:-1] == color

        positions = np.arange(len(p.plot_data)) - p.width / 2
        for bar, pos, stat in zip(ax.patches, positions, p.statistic):
            assert bar.get_y() == pos
            assert bar.get_height() == p.width
            assert bar.get_x() == 0
            assert bar.get_width() == stat

    def test_draw_nested_vertical_bars(self):

        kws = self.default_kws.copy()
        kws.update(x="g", y="y", hue="h", data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        n_groups, n_hues = len(p.plot_data), len(p.hue_names)
        assert len(ax.patches) == n_groups * n_hues
        assert len(ax.lines) == n_groups * n_hues

        for bar in ax.patches[:n_groups]:
            assert bar.get_facecolor()[:-1] == p.colors[0]
        for bar in ax.patches[n_groups:]:
            assert bar.get_facecolor()[:-1] == p.colors[1]

        positions = np.arange(len(p.plot_data))
        for bar, pos in zip(ax.patches[:n_groups], positions):
            assert bar.get_x() == approx(pos - p.width / 2)
            assert bar.get_width() == approx(p.nested_width)

        for bar, stat in zip(ax.patches, p.statistic.T.flat):
            assert bar.get_y() == approx(0)
            assert bar.get_height() == approx(stat)

    def test_draw_nested_horizontal_bars(self):

        kws = self.default_kws.copy()
        kws.update(x="y", y="g", hue="h", orient="h", data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        n_groups, n_hues = len(p.plot_data), len(p.hue_names)
        assert len(ax.patches) == n_groups * n_hues
        assert len(ax.lines) == n_groups * n_hues

        for bar in ax.patches[:n_groups]:
            assert bar.get_facecolor()[:-1] == p.colors[0]
        for bar in ax.patches[n_groups:]:
            assert bar.get_facecolor()[:-1] == p.colors[1]

        positions = np.arange(len(p.plot_data))
        for bar, pos in zip(ax.patches[:n_groups], positions):
            assert bar.get_y() == approx(pos - p.width / 2)
            assert bar.get_height() == approx(p.nested_width)

        for bar, stat in zip(ax.patches, p.statistic.T.flat):
            assert bar.get_x() == approx(0)
            assert bar.get_width() == approx(stat)

    def test_draw_missing_bars(self):

        kws = self.default_kws.copy()

        order = list("abcd")
        kws.update(x="g", y="y", order=order, data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        assert len(ax.patches) == len(order)
        assert len(ax.lines) == len(order)

        plt.close("all")

        hue_order = list("mno")
        kws.update(x="g", y="y", hue="h", hue_order=hue_order, data=self.df)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        assert len(ax.patches) == len(p.plot_data) * len(hue_order)
        assert len(ax.lines) == len(p.plot_data) * len(hue_order)

        plt.close("all")

    def test_unaligned_index(self):

        f, (ax1, ax2) = plt.subplots(2)
        cat.barplot(x=self.g, y=self.y, errorbar="sd", ax=ax1)
        cat.barplot(x=self.g, y=self.y_perm, errorbar="sd", ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert approx(l1.get_xydata()) == l2.get_xydata()
        for p1, p2 in zip(ax1.patches, ax2.patches):
            assert approx(p1.get_xy()) == p2.get_xy()
            assert approx(p1.get_height()) == p2.get_height()
            assert approx(p1.get_width()) == p2.get_width()

        f, (ax1, ax2) = plt.subplots(2)
        hue_order = self.h.unique()
        cat.barplot(x=self.g, y=self.y, hue=self.h,
                    hue_order=hue_order, errorbar="sd", ax=ax1)
        cat.barplot(x=self.g, y=self.y_perm, hue=self.h,
                    hue_order=hue_order, errorbar="sd", ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert approx(l1.get_xydata()) == l2.get_xydata()
        for p1, p2 in zip(ax1.patches, ax2.patches):
            assert approx(p1.get_xy()) == p2.get_xy()
            assert approx(p1.get_height()) == p2.get_height()
            assert approx(p1.get_width()) == p2.get_width()

    def test_barplot_colors(self):

        # Test unnested palette colors
        kws = self.default_kws.copy()
        kws.update(x="g", y="y", data=self.df,
                   saturation=1, palette="muted")
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        palette = palettes.color_palette("muted", len(self.g.unique()))
        for patch, pal_color in zip(ax.patches, palette):
            assert patch.get_facecolor()[:-1] == pal_color

        plt.close("all")

        # Test single color
        color = (.2, .2, .3, 1)
        kws = self.default_kws.copy()
        kws.update(x="g", y="y", data=self.df,
                   saturation=1, color=color)
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        for patch in ax.patches:
            assert patch.get_facecolor() == color

        plt.close("all")

        # Test nested palette colors
        kws = self.default_kws.copy()
        kws.update(x="g", y="y", hue="h", data=self.df,
                   saturation=1, palette="Set2")
        p = cat._BarPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_bars(ax, {})

        palette = palettes.color_palette("Set2", len(self.h.unique()))
        for patch in ax.patches[:len(self.g.unique())]:
            assert patch.get_facecolor()[:-1] == palette[0]
        for patch in ax.patches[len(self.g.unique()):]:
            assert patch.get_facecolor()[:-1] == palette[1]

        plt.close("all")

    def test_simple_barplots(self):

        ax = cat.barplot(x="g", y="y", data=self.df)
        assert len(ax.patches) == len(self.g.unique())
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        plt.close("all")

        ax = cat.barplot(x="y", y="g", orient="h", data=self.df)
        assert len(ax.patches) == len(self.g.unique())
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        plt.close("all")

        ax = cat.barplot(x="g", y="y", hue="h", data=self.df)
        assert len(ax.patches) == len(self.g.unique()) * len(self.h.unique())
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        plt.close("all")

        ax = cat.barplot(x="y", y="g", hue="h", orient="h", data=self.df)
        assert len(ax.patches) == len(self.g.unique()) * len(self.h.unique())
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        plt.close("all")

    def test_errorbar(self, long_df):

        ax = cat.barplot(data=long_df, x="a", y="y", errorbar=("sd", 2))
        order = categorical_order(long_df["a"])

        for i, line in enumerate(ax.lines):
            sub_df = long_df.loc[long_df["a"] == order[i], "y"]
            mean = sub_df.mean()
            sd = sub_df.std()
            expected = mean - 2 * sd, mean + 2 * sd
            assert_array_equal(line.get_ydata(), expected)


class TestPointPlotter(CategoricalFixture):

    default_kws = dict(
        x=None, y=None, hue=None, data=None,
        estimator="mean", errorbar=("ci", 95),
        n_boot=100, units=None, seed=None,
        order=None, hue_order=None,
        markers="o", linestyles="-", dodge=0,
        join=True, scale=1, orient=None,
        color=None, palette=None,
        errwidth=None, capsize=None, label=None,

    )

    def test_different_defualt_colors(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="g", y="y", data=self.df))
        p = cat._PointPlotter(**kws)
        color = palettes.color_palette()[0]
        npt.assert_array_equal(p.colors, [color, color, color])

    def test_hue_offsets(self):

        kws = self.default_kws.copy()
        kws.update(dict(x="g", y="y", hue="h", data=self.df))

        p = cat._PointPlotter(**kws)
        npt.assert_array_equal(p.hue_offsets, [0, 0])

        kws.update(dict(dodge=.5))

        p = cat._PointPlotter(**kws)
        npt.assert_array_equal(p.hue_offsets, [-.25, .25])

        kws.update(dict(x="h", hue="g", dodge=0))

        p = cat._PointPlotter(**kws)
        npt.assert_array_equal(p.hue_offsets, [0, 0, 0])

        kws.update(dict(dodge=.3))

        p = cat._PointPlotter(**kws)
        npt.assert_array_equal(p.hue_offsets, [-.15, 0, .15])

    def test_draw_vertical_points(self):

        kws = self.default_kws.copy()
        kws.update(x="g", y="y", data=self.df)
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        assert len(ax.collections) == 1
        assert len(ax.lines) == len(p.plot_data) + 1
        points = ax.collections[0]
        assert len(points.get_offsets()) == len(p.plot_data)

        x, y = points.get_offsets().T
        npt.assert_array_equal(x, np.arange(len(p.plot_data)))
        npt.assert_array_equal(y, p.statistic)

        for got_color, want_color in zip(points.get_facecolors(),
                                         p.colors):
            npt.assert_array_equal(got_color[:-1], want_color)

    def test_draw_horizontal_points(self):

        kws = self.default_kws.copy()
        kws.update(x="y", y="g", orient="h", data=self.df)
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        assert len(ax.collections) == 1
        assert len(ax.lines) == len(p.plot_data) + 1
        points = ax.collections[0]
        assert len(points.get_offsets()) == len(p.plot_data)

        x, y = points.get_offsets().T
        npt.assert_array_equal(x, p.statistic)
        npt.assert_array_equal(y, np.arange(len(p.plot_data)))

        for got_color, want_color in zip(points.get_facecolors(),
                                         p.colors):
            npt.assert_array_equal(got_color[:-1], want_color)

    def test_draw_vertical_nested_points(self):

        kws = self.default_kws.copy()
        kws.update(x="g", y="y", hue="h", data=self.df)
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        assert len(ax.collections) == 2
        assert len(ax.lines) == len(p.plot_data) * len(p.hue_names) + len(p.hue_names)

        for points, numbers, color in zip(ax.collections,
                                          p.statistic.T,
                                          p.colors):

            assert len(points.get_offsets()) == len(p.plot_data)

            x, y = points.get_offsets().T
            npt.assert_array_equal(x, np.arange(len(p.plot_data)))
            npt.assert_array_equal(y, numbers)

            for got_color in points.get_facecolors():
                npt.assert_array_equal(got_color[:-1], color)

    def test_draw_horizontal_nested_points(self):

        kws = self.default_kws.copy()
        kws.update(x="y", y="g", hue="h", orient="h", data=self.df)
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        assert len(ax.collections) == 2
        assert len(ax.lines) == len(p.plot_data) * len(p.hue_names) + len(p.hue_names)

        for points, numbers, color in zip(ax.collections,
                                          p.statistic.T,
                                          p.colors):

            assert len(points.get_offsets()) == len(p.plot_data)

            x, y = points.get_offsets().T
            npt.assert_array_equal(x, numbers)
            npt.assert_array_equal(y, np.arange(len(p.plot_data)))

            for got_color in points.get_facecolors():
                npt.assert_array_equal(got_color[:-1], color)

    def test_draw_missing_points(self):

        kws = self.default_kws.copy()
        df = self.df.copy()

        kws.update(x="g", y="y", hue="h", hue_order=["x", "y"], data=df)
        p = cat._PointPlotter(**kws)
        f, ax = plt.subplots()
        p.draw_points(ax)

        df.loc[df["h"] == "m", "y"] = np.nan
        kws.update(x="g", y="y", hue="h", data=df)
        p = cat._PointPlotter(**kws)
        f, ax = plt.subplots()
        p.draw_points(ax)

    def test_unaligned_index(self):

        f, (ax1, ax2) = plt.subplots(2)
        cat.pointplot(x=self.g, y=self.y, errorbar="sd", ax=ax1)
        cat.pointplot(x=self.g, y=self.y_perm, errorbar="sd", ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert approx(l1.get_xydata()) == l2.get_xydata()
        for p1, p2 in zip(ax1.collections, ax2.collections):
            assert approx(p1.get_offsets()) == p2.get_offsets()

        f, (ax1, ax2) = plt.subplots(2)
        hue_order = self.h.unique()
        cat.pointplot(x=self.g, y=self.y, hue=self.h,
                      hue_order=hue_order, errorbar="sd", ax=ax1)
        cat.pointplot(x=self.g, y=self.y_perm, hue=self.h,
                      hue_order=hue_order, errorbar="sd", ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert approx(l1.get_xydata()) == l2.get_xydata()
        for p1, p2 in zip(ax1.collections, ax2.collections):
            assert approx(p1.get_offsets()) == p2.get_offsets()

    def test_pointplot_colors(self):

        # Test a single-color unnested plot
        color = (.2, .2, .3, 1)
        kws = self.default_kws.copy()
        kws.update(x="g", y="y", data=self.df, color=color)
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        for line in ax.lines:
            assert line.get_color() == color[:-1]

        for got_color in ax.collections[0].get_facecolors():
            npt.assert_array_equal(rgb2hex(got_color), rgb2hex(color))

        plt.close("all")

        # Test a multi-color unnested plot
        palette = palettes.color_palette("Set1", 3)
        kws.update(x="g", y="y", data=self.df, palette="Set1")
        p = cat._PointPlotter(**kws)

        assert not p.join

        f, ax = plt.subplots()
        p.draw_points(ax)

        for line, pal_color in zip(ax.lines, palette):
            npt.assert_array_equal(line.get_color(), pal_color)

        for point_color, pal_color in zip(ax.collections[0].get_facecolors(),
                                          palette):
            npt.assert_array_equal(rgb2hex(point_color), rgb2hex(pal_color))

        plt.close("all")

        # Test a multi-colored nested plot
        palette = palettes.color_palette("dark", 2)
        kws.update(x="g", y="y", hue="h", data=self.df, palette="dark")
        p = cat._PointPlotter(**kws)

        f, ax = plt.subplots()
        p.draw_points(ax)

        for line in ax.lines[:(len(p.plot_data) + 1)]:
            assert line.get_color() == palette[0]
        for line in ax.lines[(len(p.plot_data) + 1):]:
            assert line.get_color() == palette[1]

        for i, pal_color in enumerate(palette):
            for point_color in ax.collections[i].get_facecolors():
                npt.assert_array_equal(point_color[:-1], pal_color)

        plt.close("all")

    def test_simple_pointplots(self):

        ax = cat.pointplot(x="g", y="y", data=self.df)
        assert len(ax.collections) == 1
        assert len(ax.lines) == len(self.g.unique()) + 1
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        plt.close("all")

        ax = cat.pointplot(x="y", y="g", orient="h", data=self.df)
        assert len(ax.collections) == 1
        assert len(ax.lines) == len(self.g.unique()) + 1
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        plt.close("all")

        ax = cat.pointplot(x="g", y="y", hue="h", data=self.df)
        assert len(ax.collections) == len(self.h.unique())
        assert len(ax.lines) == (
            len(self.g.unique()) * len(self.h.unique()) + len(self.h.unique())
        )
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        plt.close("all")

        ax = cat.pointplot(x="y", y="g", hue="h", orient="h", data=self.df)
        assert len(ax.collections) == len(self.h.unique())
        assert len(ax.lines) == (
            len(self.g.unique()) * len(self.h.unique()) + len(self.h.unique())
        )
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        plt.close("all")

    def test_errorbar(self, long_df):

        ax = cat.pointplot(
            data=long_df, x="a", y="y", errorbar=("sd", 2), join=False
        )
        order = categorical_order(long_df["a"])

        for i, line in enumerate(ax.lines):
            sub_df = long_df.loc[long_df["a"] == order[i], "y"]
            mean = sub_df.mean()
            sd = sub_df.std()
            expected = mean - 2 * sd, mean + 2 * sd
            assert_array_equal(line.get_ydata(), expected)

    def test_on_facetgrid(self, long_df):

        g = FacetGrid(long_df, hue="a")
        g.map(pointplot, "a", "y")
        g.add_legend()

        order = categorical_order(long_df["a"])
        legend_texts = [t.get_text() for t in g.legend.texts]
        assert legend_texts == order


class TestCountPlot(CategoricalFixture):

    def test_plot_elements(self):

        ax = cat.countplot(x="g", data=self.df)
        assert len(ax.patches) == self.g.unique().size
        for p in ax.patches:
            assert p.get_y() == 0
            assert p.get_height() == self.g.size / self.g.unique().size
        plt.close("all")

        ax = cat.countplot(y="g", data=self.df)
        assert len(ax.patches) == self.g.unique().size
        for p in ax.patches:
            assert p.get_x() == 0
            assert p.get_width() == self.g.size / self.g.unique().size
        plt.close("all")

        ax = cat.countplot(x="g", hue="h", data=self.df)
        assert len(ax.patches) == self.g.unique().size * self.h.unique().size
        plt.close("all")

        ax = cat.countplot(y="g", hue="h", data=self.df)
        assert len(ax.patches) == self.g.unique().size * self.h.unique().size
        plt.close("all")

    def test_input_error(self):

        with pytest.raises(ValueError):
            cat.countplot(x="g", y="h", data=self.df)


class TestCatPlot(CategoricalFixture):

    def test_facet_organization(self):

        g = cat.catplot(x="g", y="y", data=self.df)
        assert g.axes.shape == (1, 1)

        g = cat.catplot(x="g", y="y", col="h", data=self.df)
        assert g.axes.shape == (1, 2)

        g = cat.catplot(x="g", y="y", row="h", data=self.df)
        assert g.axes.shape == (2, 1)

        g = cat.catplot(x="g", y="y", col="u", row="h", data=self.df)
        assert g.axes.shape == (2, 3)

    def test_plot_elements(self):

        g = cat.catplot(x="g", y="y", data=self.df, kind="point")
        assert len(g.ax.collections) == 1
        want_lines = self.g.unique().size + 1
        assert len(g.ax.lines) == want_lines

        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="point")
        want_collections = self.h.unique().size
        assert len(g.ax.collections) == want_collections
        want_lines = (self.g.unique().size + 1) * self.h.unique().size
        assert len(g.ax.lines) == want_lines

        g = cat.catplot(x="g", y="y", data=self.df, kind="bar")
        want_elements = self.g.unique().size
        assert len(g.ax.patches) == want_elements
        assert len(g.ax.lines) == want_elements

        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="bar")
        want_elements = self.g.unique().size * self.h.unique().size
        assert len(g.ax.patches) == want_elements
        assert len(g.ax.lines) == want_elements

        g = cat.catplot(x="g", data=self.df, kind="count")
        want_elements = self.g.unique().size
        assert len(g.ax.patches) == want_elements
        assert len(g.ax.lines) == 0

        g = cat.catplot(x="g", hue="h", data=self.df, kind="count")
        want_elements = self.g.unique().size * self.h.unique().size
        assert len(g.ax.patches) == want_elements
        assert len(g.ax.lines) == 0

        g = cat.catplot(y="y", data=self.df, kind="box")
        want_artists = 1
        assert len(self.get_box_artists(g.ax)) == want_artists

        g = cat.catplot(x="g", y="y", data=self.df, kind="box")
        want_artists = self.g.unique().size
        assert len(self.get_box_artists(g.ax)) == want_artists

        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="box")
        want_artists = self.g.unique().size * self.h.unique().size
        assert len(self.get_box_artists(g.ax)) == want_artists

        g = cat.catplot(x="g", y="y", data=self.df,
                        kind="violin", inner=None)
        want_elements = self.g.unique().size
        assert len(g.ax.collections) == want_elements

        g = cat.catplot(x="g", y="y", hue="h", data=self.df,
                        kind="violin", inner=None)
        want_elements = self.g.unique().size * self.h.unique().size
        assert len(g.ax.collections) == want_elements

        g = cat.catplot(x="g", y="y", data=self.df, kind="strip")
        want_elements = self.g.unique().size
        assert len(g.ax.collections) == want_elements
        for strip in g.ax.collections:
            assert same_color(strip.get_facecolors(), "C0")

        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="strip")
        want_elements = self.g.unique().size + self.h.unique().size
        assert len(g.ax.collections) == want_elements

    def test_bad_plot_kind_error(self):

        with pytest.raises(ValueError):
            cat.catplot(x="g", y="y", data=self.df, kind="not_a_kind")

    def test_count_x_and_y(self):

        with pytest.raises(ValueError):
            cat.catplot(x="g", y="y", data=self.df, kind="count")

    def test_plot_colors(self):

        ax = cat.barplot(x="g", y="y", data=self.df)
        g = cat.catplot(x="g", y="y", data=self.df, kind="bar")
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        plt.close("all")

        ax = cat.barplot(x="g", y="y", data=self.df, color="purple")
        g = cat.catplot(x="g", y="y", data=self.df,
                        kind="bar", color="purple")
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        plt.close("all")

        ax = cat.barplot(x="g", y="y", data=self.df, palette="Set2", hue="h")
        g = cat.catplot(x="g", y="y", data=self.df,
                        kind="bar", palette="Set2", hue="h")
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        plt.close("all")

        ax = cat.pointplot(x="g", y="y", data=self.df)
        g = cat.catplot(x="g", y="y", data=self.df)
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        plt.close("all")

        ax = cat.pointplot(x="g", y="y", data=self.df, color="purple")
        g = cat.catplot(x="g", y="y", data=self.df, color="purple")
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        plt.close("all")

        ax = cat.pointplot(x="g", y="y", data=self.df, palette="Set2", hue="h")
        g = cat.catplot(x="g", y="y", data=self.df, palette="Set2", hue="h")
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        plt.close("all")

    def test_ax_kwarg_removal(self):

        f, ax = plt.subplots()
        with pytest.warns(UserWarning, match="catplot is a figure-level"):
            g = cat.catplot(x="g", y="y", data=self.df, ax=ax)
        assert len(ax.collections) == 0
        assert len(g.ax.collections) > 0

    def test_share_xy(self):

        # Test default behavior works
        g = cat.catplot(x="g", y="y", col="g", data=self.df, sharex=True)
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        g = cat.catplot(x="y", y="g", col="g", data=self.df, sharey=True)
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        # Test unsharing workscol
        with pytest.warns(UserWarning):
            g = cat.catplot(
                x="g", y="y", col="g", data=self.df, sharex=False, kind="bar",
            )
            for ax in g.axes.flat:
                assert len(ax.patches) == 1

        with pytest.warns(UserWarning):
            g = cat.catplot(
                x="y", y="g", col="g", data=self.df, sharey=False, kind="bar",
            )
            for ax in g.axes.flat:
                assert len(ax.patches) == 1

        # Make sure no warning is raised if color is provided on unshared plot
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = cat.catplot(
                x="g", y="y", col="g", data=self.df, sharex=False, color="b"
            )
        for ax in g.axes.flat:
            assert ax.get_xlim() == (-.5, .5)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = cat.catplot(
                x="y", y="g", col="g", data=self.df, sharey=False, color="r"
            )
        for ax in g.axes.flat:
            assert ax.get_ylim() == (.5, -.5)

        # Make sure order is used if given, regardless of sharex value
        order = self.df.g.unique()
        g = cat.catplot(x="g", y="y", col="g", data=self.df, sharex=False, order=order)
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        g = cat.catplot(x="y", y="g", col="g", data=self.df, sharey=False, order=order)
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

    @pytest.mark.parametrize("var", ["col", "row"])
    def test_array_faceter(self, long_df, var):

        g1 = catplot(data=long_df, x="y", **{var: "a"})
        g2 = catplot(data=long_df, x="y", **{var: long_df["a"].to_numpy()})

        for ax1, ax2 in zip(g1.axes.flat, g2.axes.flat):
            assert_plots_equal(ax1, ax2)


class TestBoxenPlotter(CategoricalFixture):

    default_kws = dict(x=None, y=None, hue=None, data=None,
                       order=None, hue_order=None,
                       orient=None, color=None, palette=None,
                       saturation=.75, width=.8, dodge=True,
                       k_depth='tukey', linewidth=None,
                       scale='exponential', outlier_prop=0.007,
                       trust_alpha=0.05, showfliers=True)

    def ispatch(self, c):

        return isinstance(c, mpl.collections.PatchCollection)

    def ispath(self, c):

        return isinstance(c, mpl.collections.PathCollection)

    def edge_calc(self, n, data):

        q = np.asanyarray([0.5 ** n, 1 - 0.5 ** n]) * 100
        q = list(np.unique(q))
        return np.percentile(data, q)

    def test_box_ends_finite(self):

        p = cat._LVPlotter(**self.default_kws)
        p.establish_variables("g", "y", data=self.df)
        box_ends = []
        k_vals = []
        for s in p.plot_data:
            b, k = p._lv_box_ends(s)
            box_ends.append(b)
            k_vals.append(k)

        # Check that all the box ends are finite and are within
        # the bounds of the data
        b_e = map(lambda a: np.all(np.isfinite(a)), box_ends)
        assert np.sum(list(b_e)) == len(box_ends)

        def within(t):
            a, d = t
            return ((np.ravel(a) <= d.max())
                    & (np.ravel(a) >= d.min())).all()

        b_w = map(within, zip(box_ends, p.plot_data))
        assert np.sum(list(b_w)) == len(box_ends)

        k_f = map(lambda k: (k > 0.) & np.isfinite(k), k_vals)
        assert np.sum(list(k_f)) == len(k_vals)

    def test_box_ends_correct_tukey(self):

        n = 100
        linear_data = np.arange(n)
        expected_k = max(int(np.log2(n)) - 3, 1)
        expected_edges = [self.edge_calc(i, linear_data)
                          for i in range(expected_k + 1, 1, -1)]

        p = cat._LVPlotter(**self.default_kws)
        calc_edges, calc_k = p._lv_box_ends(linear_data)

        npt.assert_array_equal(expected_edges, calc_edges)
        assert expected_k == calc_k

    def test_box_ends_correct_proportion(self):

        n = 100
        linear_data = np.arange(n)
        expected_k = int(np.log2(n)) - int(np.log2(n * 0.007)) + 1
        expected_edges = [self.edge_calc(i, linear_data)
                          for i in range(expected_k + 1, 1, -1)]

        kws = self.default_kws.copy()
        kws["k_depth"] = "proportion"
        p = cat._LVPlotter(**kws)
        calc_edges, calc_k = p._lv_box_ends(linear_data)

        npt.assert_array_equal(expected_edges, calc_edges)
        assert expected_k == calc_k

    @pytest.mark.parametrize(
        "n,exp_k",
        [(491, 6), (492, 7), (983, 7), (984, 8), (1966, 8), (1967, 9)],
    )
    def test_box_ends_correct_trustworthy(self, n, exp_k):

        linear_data = np.arange(n)
        kws = self.default_kws.copy()
        kws["k_depth"] = "trustworthy"
        p = cat._LVPlotter(**kws)
        _, calc_k = p._lv_box_ends(linear_data)

        assert exp_k == calc_k

    def test_outliers(self):

        n = 100
        outlier_data = np.append(np.arange(n - 1), 2 * n)
        expected_k = max(int(np.log2(n)) - 3, 1)
        expected_edges = [self.edge_calc(i, outlier_data)
                          for i in range(expected_k + 1, 1, -1)]

        p = cat._LVPlotter(**self.default_kws)
        calc_edges, calc_k = p._lv_box_ends(outlier_data)

        npt.assert_array_equal(calc_edges, expected_edges)
        assert calc_k == expected_k

        out_calc = p._lv_outliers(outlier_data, calc_k)
        out_exp = p._lv_outliers(outlier_data, expected_k)

        npt.assert_equal(out_calc, out_exp)

    def test_showfliers(self):

        ax = cat.boxenplot(x="g", y="y", data=self.df, k_depth="proportion",
                           showfliers=True)
        ax_collections = list(filter(self.ispath, ax.collections))
        for c in ax_collections:
            assert len(c.get_offsets()) == 2

        # Test that all data points are in the plot
        assert ax.get_ylim()[0] < self.df["y"].min()
        assert ax.get_ylim()[1] > self.df["y"].max()

        plt.close("all")

        ax = cat.boxenplot(x="g", y="y", data=self.df, showfliers=False)
        assert len(list(filter(self.ispath, ax.collections))) == 0

        plt.close("all")

    def test_invalid_depths(self):

        kws = self.default_kws.copy()

        # Make sure illegal depth raises
        kws["k_depth"] = "nosuchdepth"
        with pytest.raises(ValueError):
            cat._LVPlotter(**kws)

        # Make sure illegal outlier_prop raises
        kws["k_depth"] = "proportion"
        for p in (-13, 37):
            kws["outlier_prop"] = p
            with pytest.raises(ValueError):
                cat._LVPlotter(**kws)

        kws["k_depth"] = "trustworthy"
        for alpha in (-13, 37):
            kws["trust_alpha"] = alpha
            with pytest.raises(ValueError):
                cat._LVPlotter(**kws)

    @pytest.mark.parametrize("power", [1, 3, 7, 11, 13, 17])
    def test_valid_depths(self, power):

        x = np.random.standard_t(10, 2 ** power)

        valid_depths = ["proportion", "tukey", "trustworthy", "full"]
        kws = self.default_kws.copy()

        for depth in valid_depths + [4]:
            kws["k_depth"] = depth
            box_ends, k = cat._LVPlotter(**kws)._lv_box_ends(x)

            if depth == "full":
                assert k == int(np.log2(len(x))) + 1

    def test_valid_scales(self):

        valid_scales = ["linear", "exponential", "area"]
        kws = self.default_kws.copy()

        for scale in valid_scales + ["unknown_scale"]:
            kws["scale"] = scale
            if scale not in valid_scales:
                with pytest.raises(ValueError):
                    cat._LVPlotter(**kws)
            else:
                cat._LVPlotter(**kws)

    def test_hue_offsets(self):

        p = cat._LVPlotter(**self.default_kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.2, .2])

        kws = self.default_kws.copy()
        kws["width"] = .6
        p = cat._LVPlotter(**kws)
        p.establish_variables("g", "y", hue="h", data=self.df)
        npt.assert_array_equal(p.hue_offsets, [-.15, .15])

        p = cat._LVPlotter(**kws)
        p.establish_variables("h", "y", "g", data=self.df)
        npt.assert_array_almost_equal(p.hue_offsets, [-.2, 0, .2])

    def test_axes_data(self):

        ax = cat.boxenplot(x="g", y="y", data=self.df)
        patches = filter(self.ispatch, ax.collections)
        assert len(list(patches)) == 3

        plt.close("all")

        ax = cat.boxenplot(x="g", y="y", hue="h", data=self.df)
        patches = filter(self.ispatch, ax.collections)
        assert len(list(patches)) == 6

        plt.close("all")

    def test_box_colors(self):

        pal = palettes.color_palette()

        ax = cat.boxenplot(
            x="g", y="y", data=self.df, saturation=1, showfliers=False
        )
        ax.figure.canvas.draw()
        for i, box in enumerate(ax.collections):
            assert same_color(box.get_facecolor()[0], pal[i])

        plt.close("all")

        ax = cat.boxenplot(
            x="g", y="y", hue="h", data=self.df, saturation=1, showfliers=False
        )
        ax.figure.canvas.draw()
        for i, box in enumerate(ax.collections):
            assert same_color(box.get_facecolor()[0], pal[i % 2])

        plt.close("all")

    def test_draw_missing_boxes(self):

        ax = cat.boxenplot(x="g", y="y", data=self.df,
                           order=["a", "b", "c", "d"])

        patches = filter(self.ispatch, ax.collections)
        assert len(list(patches)) == 3
        plt.close("all")

    def test_unaligned_index(self):

        f, (ax1, ax2) = plt.subplots(2)
        cat.boxenplot(x=self.g, y=self.y, ax=ax1)
        cat.boxenplot(x=self.g, y=self.y_perm, ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert np.array_equal(l1.get_xydata(), l2.get_xydata())

        f, (ax1, ax2) = plt.subplots(2)
        hue_order = self.h.unique()
        cat.boxenplot(x=self.g, y=self.y, hue=self.h,
                      hue_order=hue_order, ax=ax1)
        cat.boxenplot(x=self.g, y=self.y_perm, hue=self.h,
                      hue_order=hue_order, ax=ax2)
        for l1, l2 in zip(ax1.lines, ax2.lines):
            assert np.array_equal(l1.get_xydata(), l2.get_xydata())

    def test_missing_data(self):

        x = ["a", "a", "b", "b", "c", "c", "d", "d"]
        h = ["x", "y", "x", "y", "x", "y", "x", "y"]
        y = self.rs.randn(8)
        y[-2:] = np.nan

        ax = cat.boxenplot(x=x, y=y)
        assert len(ax.lines) == 3

        plt.close("all")

        y[-1] = 0
        ax = cat.boxenplot(x=x, y=y, hue=h)
        assert len(ax.lines) == 7

        plt.close("all")

    def test_boxenplots(self):

        # Smoke test the high level boxenplot options

        cat.boxenplot(x="y", data=self.df)
        plt.close("all")

        cat.boxenplot(y="y", data=self.df)
        plt.close("all")

        cat.boxenplot(x="g", y="y", data=self.df)
        plt.close("all")

        cat.boxenplot(x="y", y="g", data=self.df, orient="h")
        plt.close("all")

        cat.boxenplot(x="g", y="y", hue="h", data=self.df)
        plt.close("all")

        for scale in ("linear", "area", "exponential"):
            cat.boxenplot(x="g", y="y", hue="h", scale=scale, data=self.df)
            plt.close("all")

        for depth in ("proportion", "tukey", "trustworthy"):
            cat.boxenplot(x="g", y="y", hue="h", k_depth=depth, data=self.df)
            plt.close("all")

        order = list("nabc")
        cat.boxenplot(x="g", y="y", hue="h", order=order, data=self.df)
        plt.close("all")

        order = list("omn")
        cat.boxenplot(x="g", y="y", hue="h", hue_order=order, data=self.df)
        plt.close("all")

        cat.boxenplot(x="y", y="g", hue="h", data=self.df, orient="h")
        plt.close("all")

        cat.boxenplot(x="y", y="g", hue="h", data=self.df, orient="h",
                      palette="Set2")
        plt.close("all")

        cat.boxenplot(x="y", y="g", hue="h", data=self.df,
                      orient="h", color="b")
        plt.close("all")

    def test_axes_annotation(self):

        ax = cat.boxenplot(x="g", y="y", data=self.df)
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        assert ax.get_xlim() == (-.5, 2.5)
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])

        plt.close("all")

        ax = cat.boxenplot(x="g", y="y", hue="h", data=self.df)
        assert ax.get_xlabel() == "g"
        assert ax.get_ylabel() == "y"
        npt.assert_array_equal(ax.get_xticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_xticklabels()],
                               ["a", "b", "c"])
        npt.assert_array_equal([l.get_text() for l in ax.legend_.get_texts()],
                               ["m", "n"])

        plt.close("all")

        ax = cat.boxenplot(x="y", y="g", data=self.df, orient="h")
        assert ax.get_xlabel() == "y"
        assert ax.get_ylabel() == "g"
        assert ax.get_ylim() == (2.5, -.5)
        npt.assert_array_equal(ax.get_yticks(), [0, 1, 2])
        npt.assert_array_equal([l.get_text() for l in ax.get_yticklabels()],
                               ["a", "b", "c"])

        plt.close("all")

    @pytest.mark.parametrize("size", ["large", "medium", "small", 22, 12])
    def test_legend_titlesize(self, size):

        rc_ctx = {"legend.title_fontsize": size}
        exp = mpl.font_manager.FontProperties(size=size).get_size()

        with plt.rc_context(rc=rc_ctx):
            ax = cat.boxenplot(x="g", y="y", hue="h", data=self.df)
            obs = ax.get_legend().get_title().get_fontproperties().get_size()
            assert obs == exp

        plt.close("all")

    @pytest.mark.skipif(
        _version_predates(pd, "1.2"),
        reason="Test requires pandas>=1.2")
    def test_Float64_input(self):
        data = pd.DataFrame(
            {"x": np.random.choice(["a", "b"], 20), "y": np.random.random(20)}
        )
        data['y'] = data['y'].astype(pd.Float64Dtype())
        _ = cat.boxenplot(x="x", y="y", data=data)

        plt.close("all")

    def test_line_kws(self):
        line_kws = {'linewidth': 5, 'color': 'purple',
                    'linestyle': '-.'}

        ax = cat.boxenplot(data=self.df, y='y', line_kws=line_kws)

        median_line = ax.lines[0]

        assert median_line.get_linewidth() == line_kws['linewidth']
        assert median_line.get_linestyle() == line_kws['linestyle']
        assert median_line.get_color() == line_kws['color']

        plt.close("all")

    def test_flier_kws(self):
        flier_kws = {
            'marker': 'v',
            'color': np.array([[1, 0, 0, 1]]),
            's': 5,
        }

        ax = cat.boxenplot(data=self.df, y='y', x='g', flier_kws=flier_kws)

        outliers_scatter = ax.findobj(mpl.collections.PathCollection)[0]

        # The number of vertices for a triangle is 3, the length of Path
        # collection objects is defined as n + 1 vertices.
        assert len(outliers_scatter.get_paths()[0]) == 4
        assert len(outliers_scatter.get_paths()[-1]) == 4

        assert (outliers_scatter.get_facecolor() == flier_kws['color']).all()

        assert np.unique(outliers_scatter.get_sizes()) == flier_kws['s']

        plt.close("all")

    def test_box_kws(self):

        box_kws = {'linewidth': 5, 'edgecolor': np.array([[0, 1, 0, 1]])}

        ax = cat.boxenplot(data=self.df, y='y', x='g',
                           box_kws=box_kws)

        boxes = ax.findobj(mpl.collections.PatchCollection)[0]

        # The number of vertices for a triangle is 3, the length of Path
        # collection objects is defined as n + 1 vertices.
        assert len(boxes.get_paths()[0]) == 5
        assert len(boxes.get_paths()[-1]) == 5

        assert np.unique(boxes.get_linewidth() == box_kws['linewidth'])

        plt.close("all")


class TestBeeswarm:

    def test_could_overlap(self):

        p = Beeswarm()
        neighbors = p.could_overlap(
            (1, 1, .5),
            [(0, 0, .5),
             (1, .1, .2),
             (.5, .5, .5)]
        )
        assert_array_equal(neighbors, [(.5, .5, .5)])

    def test_position_candidates(self):

        p = Beeswarm()
        xy_i = (0, 1, .5)
        neighbors = [(0, 1, .5), (0, 1.5, .5)]
        candidates = p.position_candidates(xy_i, neighbors)
        dx1 = 1.05
        dx2 = np.sqrt(1 - .5 ** 2) * 1.05
        assert_array_equal(
            candidates,
            [(0, 1, .5), (-dx1, 1, .5), (dx1, 1, .5), (dx2, 1, .5), (-dx2, 1, .5)]
        )

    def test_find_first_non_overlapping_candidate(self):

        p = Beeswarm()
        candidates = [(.5, 1, .5), (1, 1, .5), (1.5, 1, .5)]
        neighbors = np.array([(0, 1, .5)])

        first = p.first_non_overlapping_candidate(candidates, neighbors)
        assert_array_equal(first, (1, 1, .5))

    def test_beeswarm(self, long_df):

        p = Beeswarm()
        data = long_df["y"]
        d = data.diff().mean() * 1.5
        x = np.zeros(data.size)
        y = np.sort(data)
        r = np.full_like(y, d)
        orig_xyr = np.c_[x, y, r]
        swarm = p.beeswarm(orig_xyr)[:, :2]
        dmat = np.sqrt(np.sum(np.square(swarm[:, np.newaxis] - swarm), axis=-1))
        triu = dmat[np.triu_indices_from(dmat, 1)]
        assert_array_less(d, triu)
        assert_array_equal(y, swarm[:, 1])

    def test_add_gutters(self):

        p = Beeswarm(width=1)

        points = np.zeros(10)
        assert_array_equal(points, p.add_gutters(points, 0))

        points = np.array([0, -1, .4, .8])
        msg = r"50.0% of the points cannot be placed.+$"
        with pytest.warns(UserWarning, match=msg):
            new_points = p.add_gutters(points, 0)
        assert_array_equal(new_points, np.array([0, -.5, .4, .5]))
