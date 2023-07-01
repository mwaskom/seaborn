import itertools
from functools import partial

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import same_color, to_rgb, to_rgba

import pytest
from pytest import approx
import numpy.testing as npt
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_array_almost_equal,
)

from seaborn import categorical as cat
from seaborn import palettes

from seaborn.utils import _version_predates, desaturate
from seaborn._oldcore import categorical_order
from seaborn.categorical import (
    _CategoricalPlotterNew,
    Beeswarm,
    BoxPlotContainer,
    catplot,
    barplot,
    boxplot,
    boxenplot,
    countplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
)
from seaborn.palettes import color_palette
from seaborn.utils import _draw_figure
from seaborn._compat import get_colormap, get_legend_handles
from seaborn._testing import assert_plots_equal


PLOT_FUNCS = [
    catplot,
    barplot,
    boxplot,
    boxenplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
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


# ====================================================================================
# ====================================================================================


class SharedAxesLevelTests:

    def orient_indices(self, orient):
        pos_idx = ["x", "y"].index(orient)
        val_idx = ["y", "x"].index(orient)
        return pos_idx, val_idx

    @pytest.fixture
    def common_kws(self):
        return {}

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_labels_long(self, long_df, orient):

        depend = {"x": "y", "y": "x"}[orient]
        kws = {orient: "a", depend: "y", "hue": "b"}

        ax = self.func(long_df, **kws)

        # To populate texts; only needed on older matplotlibs
        _draw_figure(ax.figure)

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

        # To populate texts; only needed on older matplotlibs
        _draw_figure(ax.figure)

        assert ax.get_xlabel() == wide_df.columns.name
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label, level in zip(labels, wide_df.columns):
            assert label == level

    def test_labels_hue_order(self, long_df):

        hue_var = "b"
        hue_order = categorical_order(long_df[hue_var])[::-1]
        ax = self.func(long_df, x="a", y="y", hue=hue_var, hue_order=hue_order)
        legend = ax.get_legend()
        hue_labels = [t.get_text() for t in legend.texts]
        assert hue_labels == hue_order

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

    def test_redundant_hue_legend(self, long_df):

        ax = self.func(long_df, x="a", y="y", hue="a")
        assert ax.get_legend() is None
        ax.clear()

        self.func(long_df, x="a", y="y", hue="a", legend=True)
        assert ax.get_legend() is not None


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
            ("y", "dataframe"), ("y", "dict"),
            ("x", "dataframe"), ("x", "dict"),
        ]
    )
    def test_wide(self, wide_df, orient, data_type):

        if data_type == "dict":
            wide_df = {k: v.to_numpy() for k, v in wide_df.items()}

        ax = self.func(data=wide_df, orient=orient)
        _draw_figure(ax.figure)

        cat_idx = 0 if orient in "vx" else 1
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
                assert tuple(point_color) == to_rgba("C0")

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


class SharedAggTests(SharedAxesLevelTests):

    def test_labels_flat(self):

        ind = pd.Index(["a", "b", "c"], name="x")
        ser = pd.Series([1, 2, 3], ind, name="y")

        ax = self.func(ser)

        # To populate texts; only needed on older matplotlibs
        _draw_figure(ax.figure)

        assert ax.get_xlabel() == ind.name
        assert ax.get_ylabel() == ser.name
        labels = [t.get_text() for t in ax.get_xticklabels()]
        for label, level in zip(labels, ind):
            assert label == level


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


class TestBoxPlot(SharedAxesLevelTests):

    func = staticmethod(boxplot)

    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    def get_last_color(self, ax):

        colors = [b.get_facecolor() for b in ax.containers[-1].boxes]
        unique_colors = np.unique(colors, axis=0)
        assert len(unique_colors) == 1
        return to_rgba(unique_colors.squeeze())

    def get_box_verts(self, box):

        path = box.get_path()
        visible_codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO]
        visible = np.isin(path.codes, visible_codes)
        return path.vertices[visible].T

    def check_box(self, bxp, data, orient, pos, width=0.8):

        pos_idx, val_idx = self.orient_indices(orient)

        p25, p50, p75 = np.percentile(data, [25, 50, 75])

        box = self.get_box_verts(bxp.box)
        assert box[val_idx].min() == p25
        assert box[val_idx].max() == p75
        assert box[pos_idx].min() == approx(pos - width / 2)
        assert box[pos_idx].max() == approx(pos + width / 2)

        med = bxp.median.get_xydata().T
        assert tuple(med[val_idx]) == (p50, p50)
        assert np.allclose(med[pos_idx], (pos - width / 2, pos + width / 2))

    def check_whiskers(self, bxp, data, orient, pos, capsize=0.4, whis=1.5):

        pos_idx, val_idx = self.orient_indices(orient)

        whis_lo = bxp.whiskers[0].get_xydata().T
        whis_hi = bxp.whiskers[1].get_xydata().T
        caps_lo = bxp.caps[0].get_xydata().T
        caps_hi = bxp.caps[1].get_xydata().T
        fliers = bxp.fliers.get_xydata().T

        p25, p75 = np.percentile(data, [25, 75])
        iqr = p75 - p25

        adj_lo = data[data >= (p25 - iqr * whis)].min()
        adj_hi = data[data <= (p75 + iqr * whis)].max()

        assert whis_lo[val_idx].max() == p25
        assert whis_lo[val_idx].min() == approx(adj_lo)
        assert np.allclose(whis_lo[pos_idx], (pos, pos))
        assert np.allclose(caps_lo[val_idx], (adj_lo, adj_lo))
        assert np.allclose(caps_lo[pos_idx], (pos - capsize / 2, pos + capsize / 2))

        assert whis_hi[val_idx].min() == p75
        assert whis_hi[val_idx].max() == approx(adj_hi)
        assert np.allclose(whis_hi[pos_idx], (pos, pos))
        assert np.allclose(caps_hi[val_idx], (adj_hi, adj_hi))
        assert np.allclose(caps_hi[pos_idx], (pos - capsize / 2, pos + capsize / 2))

        flier_data = data[(data < adj_lo) | (data > adj_hi)]
        assert sorted(fliers[val_idx]) == sorted(flier_data)
        assert np.allclose(fliers[pos_idx], pos)

    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    def test_single_var(self, long_df, orient, col):

        var = {"x": "y", "y": "x"}[orient]
        ax = boxplot(long_df, **{var: col})
        bxp = ax.containers[0][0]
        self.check_box(bxp, long_df[col], orient, 0)
        self.check_whiskers(bxp, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    def test_vector_data(self, long_df, orient, col):

        ax = boxplot(long_df[col], orient=orient)
        orient = "x" if orient is None else orient
        bxp = ax.containers[0][0]
        self.check_box(bxp, long_df[col], orient, 0)
        self.check_whiskers(bxp, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_wide_data(self, wide_df, orient):

        orient = {"h": "y", "v": "x"}[orient]
        ax = boxplot(wide_df, orient=orient)
        for i, bxp in enumerate(ax.containers):
            col = wide_df.columns[i]
            self.check_box(bxp[i], wide_df[col], orient, i)
            self.check_whiskers(bxp[i], wide_df[col], orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = boxplot(long_df, **{orient: "a", value: "z"})
        bxp, = ax.containers
        levels = categorical_order(long_df["a"])
        for i, level in enumerate(levels):
            data = long_df.loc[long_df["a"] == level, "z"]
            self.check_box(bxp[i], data, orient, i)
            self.check_whiskers(bxp[i], data, orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_hue_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = boxplot(long_df, hue="c", **{orient: "a", value: "z"})
        for i, hue_level in enumerate(categorical_order(long_df["c"])):
            bxp = ax.containers[i]
            for j, level in enumerate(categorical_order(long_df["a"])):
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = j + [-.2, +.2][i]
                width, capsize = 0.4, 0.2
                self.check_box(bxp[j], data, orient, pos, width)
                self.check_whiskers(bxp[j], data, orient, pos, capsize)

    def test_hue_not_dodged(self, long_df):

        levels = categorical_order(long_df["b"])
        hue = long_df["b"].isin(levels[:2])
        ax = boxplot(long_df, x="b", y="z", hue=hue)
        bxps = ax.containers
        for i, level in enumerate(levels):
            idx = int(i < 2)
            data = long_df.loc[long_df["b"] == level, "z"]
            self.check_box(bxps[idx][i % 2], data, "x", i)
            self.check_whiskers(bxps[idx][i % 2], data, "x", i)

    def test_dodge_native_scale(self, long_df):

        centers = categorical_order(long_df["s"])
        hue_levels = categorical_order(long_df["c"])
        spacing = min(np.diff(centers))
        width = 0.8 * spacing / len(hue_levels)
        offset = width / len(hue_levels)
        ax = boxplot(long_df, x="s", y="z", hue="c", native_scale=True)
        for i, hue_level in enumerate(hue_levels):
            bxp = ax.containers[i]
            for j, center in enumerate(centers):
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = center + [-offset, +offset][i]
                self.check_box(bxp[j], data, "x", pos, width)
                self.check_whiskers(bxp[j], data, "x", pos, width / 2)

    def test_dodge_native_scale_log(self, long_df):

        pos = 10 ** long_df["s"]
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        boxplot(long_df, x=pos, y="z", hue="c", native_scale=True, ax=ax)
        widths = []
        for bxp in ax.containers:
            for box in bxp.boxes:
                coords = np.log10(box.get_path().vertices.T[0])
                widths.append(np.ptp(coords))
        assert np.std(widths) == approx(0)

    def test_color(self, long_df):

        color = "#123456"
        ax = boxplot(long_df, x="a", y="y", color=color, saturation=1)
        for box in ax.containers[0].boxes:
            assert same_color(box.get_facecolor(), color)

    def test_hue_colors(self, long_df):

        ax = boxplot(long_df, x="a", y="y", hue="b", saturation=1)
        for i, bxp in enumerate(ax.containers):
            for box in bxp.boxes:
                assert same_color(box.get_facecolor(), f"C{i}")

    def test_linecolor(self, long_df):

        color = "#778815"
        ax = boxplot(long_df, x="a", y="y", linecolor=color)
        bxp = ax.containers[0]
        for line in [*bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert same_color(line.get_color(), color)
        for box in bxp.boxes:
            assert same_color(box.get_edgecolor(), color)
        for flier in bxp.fliers:
            assert same_color(flier.get_markeredgecolor(), color)

    def test_saturation(self, long_df):

        color = "#8912b0"
        ax = boxplot(long_df["x"], color=color, saturation=.5)
        box = ax.containers[0].boxes[0]
        assert np.allclose(box.get_facecolor()[:3], desaturate(color, 0.5))

    def test_linewidth(self, long_df):

        width = 5
        ax = boxplot(long_df, x="a", y="y", linewidth=width)
        bxp = ax.containers[0]
        for line in [*bxp.boxes, *bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert line.get_linewidth() == width

    def test_fill(self, long_df):

        color = "#459900"
        ax = boxplot(x=long_df["z"], fill=False, color=color)
        bxp = ax.containers[0]
        assert isinstance(bxp.boxes[0], mpl.lines.Line2D)
        for line in [*bxp.boxes, *bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert same_color(line.get_color(), color)

    @pytest.mark.parametrize("notch_param", ["notch", "shownotches"])
    def test_notch(self, long_df, notch_param):

        ax = boxplot(x=long_df["z"], **{notch_param: True})
        verts = ax.containers[0].boxes[0].get_path().vertices
        assert len(verts) == 12

    def test_whis(self, long_df):

        data = long_df["z"]
        ax = boxplot(x=data, whis=2)
        bxp = ax.containers[0][0]
        self.check_whiskers(bxp, data, "y", 0, whis=2)

    def test_gap(self, long_df):

        ax = boxplot(long_df, x="a", y="z", hue="c", gap=.1)
        for i, hue_level in enumerate(categorical_order(long_df["c"])):
            bxp = ax.containers[i]
            for j, level in enumerate(categorical_order(long_df["a"])):
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = j + [-.2, +.2][i]
                width = 0.9 * 0.4
                self.check_box(bxp[j], data, "x", pos, width)

    def test_prop_dicts(self, long_df):

        prop_dicts = dict(
            boxprops=dict(linewidth=3),
            medianprops=dict(color=".1"),
            whiskerprops=dict(linestyle="--"),
            capprops=dict(solid_capstyle="butt"),
            flierprops=dict(marker="s"),
        )
        attr_map = dict(box="boxes", flier="fliers")
        ax = boxplot(long_df, x="a", y="z", hue="c", **prop_dicts)
        for bxp in ax.containers:
            for element in ["box", "median", "whisker", "cap", "flier"]:
                attr = attr_map.get(element, f"{element}s")
                for artist in getattr(bxp, attr):
                    for k, v in prop_dicts[f"{element}props"].items():
                        assert plt.getp(artist, k) == v

    def test_showfliers(self, long_df):

        ax = boxplot(long_df["x"], showfliers=False)
        assert not ax.containers[0].fliers

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s"),
            dict(data="null", x="a", y="y", hue="a"),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),
            dict(data="null", x="a", y="y", whis=1, showfliers=False),
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),
            dict(data="null", x="a", y="y", shownotches=True, showcaps=False),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = boxplot(**kwargs)
        g = catplot(**kwargs, kind="box")

        assert_plots_equal(ax, g.ax)


class TestBoxenPlot(SharedAxesLevelTests):

    func = staticmethod(boxenplot)

    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    def get_last_color(self, ax):

        fcs = ax.collections[-2].get_facecolors()
        return to_rgba(fcs[len(fcs) // 2])

    def get_box_width(self, path, orient="x"):

        verts = path.vertices.T
        idx = ["y", "x"].index(orient)
        return np.ptp(verts[idx])

    def check_boxen(self, patches, data, orient, pos, width=0.8):

        pos_idx, val_idx = self.orient_indices(orient)
        verts = np.stack([v.vertices for v in patches.get_paths()], 1).T

        assert verts[pos_idx].min().round(4) >= np.round(pos - width / 2, 4)
        assert verts[pos_idx].max().round(4) <= np.round(pos + width / 2, 4)
        assert np.in1d(
            np.percentile(data, [25, 75]).round(4), verts[val_idx].round(4).flat
        ).all()
        assert_array_equal(verts[val_idx, 1:, 0], verts[val_idx, :-1, 2])

    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    def test_single_var(self, long_df, orient, col):

        var = {"x": "y", "y": "x"}[orient]
        ax = boxenplot(long_df, **{var: col})
        patches = ax.collections[0]
        self.check_boxen(patches, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    def test_vector_data(self, long_df, orient, col):

        orient = "x" if orient is None else orient
        ax = boxenplot(long_df[col], orient=orient)
        patches = ax.collections[0]
        self.check_boxen(patches, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_wide_data(self, wide_df, orient):

        orient = {"h": "y", "v": "x"}[orient]
        ax = boxenplot(wide_df, orient=orient)
        collections = ax.findobj(mpl.collections.PatchCollection)
        for i, patches in enumerate(collections):
            col = wide_df.columns[i]
            self.check_boxen(patches, wide_df[col], orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = boxenplot(long_df, **{orient: "a", value: "z"})
        levels = categorical_order(long_df["a"])
        collections = ax.findobj(mpl.collections.PatchCollection)
        for i, level in enumerate(levels):
            data = long_df.loc[long_df["a"] == level, "z"]
            self.check_boxen(collections[i], data, orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_hue_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = boxenplot(long_df, hue="c", **{orient: "a", value: "z"})
        collections = iter(ax.findobj(mpl.collections.PatchCollection))
        for i, level in enumerate(categorical_order(long_df["a"])):
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = i + [-.2, +.2][j]
                width = 0.4
                self.check_boxen(next(collections), data, orient, pos, width)

    def test_dodge_native_scale(self, long_df):

        centers = categorical_order(long_df["s"])
        hue_levels = categorical_order(long_df["c"])
        spacing = min(np.diff(centers))
        width = 0.8 * spacing / len(hue_levels)
        offset = width / len(hue_levels)
        ax = boxenplot(long_df, x="s", y="z", hue="c", native_scale=True)
        collections = iter(ax.findobj(mpl.collections.PatchCollection))
        for center in centers:
            for i, hue_level in enumerate(hue_levels):
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = center + [-offset, +offset][i]
                self.check_boxen(next(collections), data, "x", pos, width)

    def test_color(self, long_df):

        color = "#123456"
        ax = boxenplot(long_df, x="a", y="y", color=color, saturation=1)
        collections = ax.findobj(mpl.collections.PatchCollection)
        for patches in collections:
            fcs = patches.get_facecolors()
            assert same_color(fcs[len(fcs) // 2], color)

    def test_hue_colors(self, long_df):

        ax = boxenplot(long_df, x="a", y="y", hue="b", saturation=1)
        n_levels = long_df["b"].nunique()
        collections = ax.findobj(mpl.collections.PatchCollection)
        for i, patches in enumerate(collections):
            fcs = patches.get_facecolors()
            assert same_color(fcs[len(fcs) // 2], f"C{i % n_levels}")

    def test_linecolor(self, long_df):

        color = "#669913"
        ax = boxenplot(long_df, x="a", y="y", linecolor=color)
        for patches in ax.findobj(mpl.collections.PatchCollection):
            assert same_color(patches.get_edgecolor(), color)

    def test_linewidth(self, long_df):

        width = 5
        ax = boxenplot(long_df, x="a", y="y", linewidth=width)
        for patches in ax.findobj(mpl.collections.PatchCollection):
            assert patches.get_linewidth() == width

    def test_saturation(self, long_df):

        color = "#8912b0"
        ax = boxenplot(long_df["x"], color=color, saturation=.5)
        fcs = ax.collections[0].get_facecolors()
        assert np.allclose(fcs[len(fcs) // 2, :3], desaturate(color, 0.5))

    def test_gap(self, long_df):

        ax1, ax2 = mpl.figure.Figure().subplots(2)
        boxenplot(long_df, x="a", y="y", hue="s", ax=ax1)
        boxenplot(long_df, x="a", y="y", hue="s", gap=.2, ax=ax2)
        c1 = ax1.findobj(mpl.collections.PatchCollection)
        c2 = ax2.findobj(mpl.collections.PatchCollection)
        for p1, p2 in zip(c1, c2):
            w1 = np.ptp(p1.get_paths()[0].vertices[:, 0])
            w2 = np.ptp(p2.get_paths()[0].vertices[:, 0])
            assert (w2 / w1) == pytest.approx(0.8)

    def test_fill(self, long_df):

        ax = boxenplot(long_df, x="a", y="y", hue="s", fill=False)
        for c in ax.findobj(mpl.collections.PatchCollection):
            assert not c.get_facecolors().size

    def test_k_depth_int(self, rng):

        x = rng.normal(0, 1, 10_000)
        ax = boxenplot(x, k_depth=(k := 8))
        assert len(ax.collections[0].get_paths()) == (k * 2 - 1)

    def test_trust_alpha(self, rng):

        x = rng.normal(0, 1, 10_000)
        ax = boxenplot(x, k_depth="trustworthy", trust_alpha=.1)
        boxenplot(x, k_depth="trustworthy", trust_alpha=.001, ax=ax)
        cs = ax.findobj(mpl.collections.PatchCollection)
        assert len(cs[0].get_paths()) > len(cs[1].get_paths())

    def test_outlier_prop(self, rng):

        x = rng.normal(0, 1, 10_000)
        ax = boxenplot(x, k_depth="proportion", outlier_prop=.001)
        boxenplot(x, k_depth="proportion", outlier_prop=.1, ax=ax)
        cs = ax.findobj(mpl.collections.PatchCollection)
        assert len(cs[0].get_paths()) > len(cs[1].get_paths())

    def test_exponential_width_method(self, rng):

        x = rng.normal(0, 1, 10_000)
        ax = boxenplot(x=x, width_method="exponential")
        c = ax.findobj(mpl.collections.PatchCollection)[0]
        ws = [self.get_box_width(p) for p in c.get_paths()]
        assert (ws[1] / ws[0]) == pytest.approx(ws[2] / ws[1])

    def test_linear_width_method(self, rng):

        x = rng.normal(0, 1, 10_000)
        ax = boxenplot(x=x, width_method="linear")
        c = ax.findobj(mpl.collections.PatchCollection)[0]
        ws = [self.get_box_width(p) for p in c.get_paths()]
        assert (ws[1] - ws[0]) == pytest.approx(ws[2] - ws[1])

    def test_area_width_method(self, rng):

        x = rng.uniform(0, 1, 10_000)
        ax = boxenplot(x=x, width_method="area", k_depth=2)
        ps = ax.findobj(mpl.collections.PatchCollection)[0].get_paths()
        ws = [self.get_box_width(p) for p in ps]
        assert np.greater(ws, 0.7).all()

    def test_box_kws(self, long_df):

        ax = boxenplot(long_df, x="a", y="y", box_kws={"linewidth": (lw := 7.1)})
        for c in ax.findobj(mpl.collections.PatchCollection):
            assert c.get_linewidths() == lw

    def test_line_kws(self, long_df):

        ax = boxenplot(long_df, x="a", y="y", line_kws={"linewidth": (lw := 6.2)})
        for line in ax.lines:
            assert line.get_linewidth() == lw

    def test_flier_kws(self, long_df):

        ax = boxenplot(long_df, x="a", y="y", flier_kws={"marker": (marker := "X")})
        expected = mpl.markers.MarkerStyle(marker).get_path().vertices
        for c in ax.findobj(mpl.collections.PathCollection):
            assert_array_equal(c.get_paths()[0].vertices, expected)

    def test_k_depth_checks(self, long_df):

        with pytest.raises(ValueError, match="The value for `k_depth`"):
            boxenplot(x=long_df["y"], k_depth="auto")

        with pytest.raises(TypeError, match="The `k_depth` parameter"):
            boxenplot(x=long_df["y"], k_depth=(1, 2))

    def test_width_method_check(self, long_df):

        with pytest.raises(ValueError, match="The value for `width_method`"):
            boxenplot(x=long_df["y"], width_method="uniform")

    def test_scale_deprecation(self, long_df):

        with pytest.warns(FutureWarning, match="The `scale` parameter has been"):
            boxenplot(x=long_df["y"], scale="linear")

        with pytest.warns(FutureWarning, match=".+result for 'area' will appear"):
            boxenplot(x=long_df["y"], scale="area")

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s", showfliers=False),
            dict(data="null", x="a", y="y", hue="a", saturation=.5),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),
            dict(data="long", x="a", y="y", k_depth="trustworthy", trust_alpha=.1),
            dict(data="long", x="a", y="y", k_depth="proportion", outlier_prop=.1),
            dict(data="long", x="a", y="z", width_method="area"),
            dict(data="long", x="a", y="z", box_kws={"alpha": .2}, alpha=.4)
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = boxenplot(**kwargs)
        g = catplot(**kwargs, kind="boxen")

        assert_plots_equal(ax, g.ax)


class TestViolinPlot(SharedAxesLevelTests):

    func = staticmethod(violinplot)

    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    def get_last_color(self, ax):

        color = ax.collections[-1].get_facecolor()
        return to_rgba(color)

    def violin_width(self, poly, orient="x"):

        idx, _ = self.orient_indices(orient)
        return np.ptp(poly.get_paths()[0].vertices[:, idx])

    def check_violin(self, poly, data, orient, pos, width=0.8):

        pos_idx, val_idx = self.orient_indices(orient)
        verts = poly.get_paths()[0].vertices.T

        assert verts[pos_idx].min() >= (pos - width / 2)
        assert verts[pos_idx].max() <= (pos + width / 2)
        # Assumes violin was computed with cut=0
        assert verts[val_idx].min() == approx(data.min())
        assert verts[val_idx].max() == approx(data.max())

    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    def test_single_var(self, long_df, orient, col):

        var = {"x": "y", "y": "x"}[orient]
        ax = violinplot(long_df, **{var: col}, cut=0)
        poly = ax.collections[0]
        self.check_violin(poly, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    def test_vector_data(self, long_df, orient, col):

        orient = "x" if orient is None else orient
        ax = violinplot(long_df[col], cut=0, orient=orient)
        poly = ax.collections[0]
        self.check_violin(poly, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_wide_data(self, wide_df, orient):

        orient = {"h": "y", "v": "x"}[orient]
        ax = violinplot(wide_df, cut=0, orient=orient)
        for i, poly in enumerate(ax.collections):
            col = wide_df.columns[i]
            self.check_violin(poly, wide_df[col], orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = violinplot(long_df, **{orient: "a", value: "z"}, cut=0)
        levels = categorical_order(long_df["a"])
        for i, level in enumerate(levels):
            data = long_df.loc[long_df["a"] == level, "z"]
            self.check_violin(ax.collections[i], data, orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_hue_grouped(self, long_df, orient):

        value = {"x": "y", "y": "x"}[orient]
        ax = violinplot(long_df, hue="c", **{orient: "a", value: "z"}, cut=0)
        polys = iter(ax.collections)
        for i, level in enumerate(categorical_order(long_df["a"])):
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = i + [-.2, +.2][j]
                width = 0.4
                self.check_violin(next(polys), data, orient, pos, width)

    def test_hue_not_dodged(self, long_df):

        levels = categorical_order(long_df["b"])
        hue = long_df["b"].isin(levels[:2])
        ax = violinplot(long_df, x="b", y="z", hue=hue, cut=0)
        for i, level in enumerate(levels):
            poly = ax.collections[i]
            data = long_df.loc[long_df["b"] == level, "z"]
            self.check_violin(poly, data, "x", i)

    def test_dodge_native_scale(self, long_df):

        centers = categorical_order(long_df["s"])
        hue_levels = categorical_order(long_df["c"])
        spacing = min(np.diff(centers))
        width = 0.8 * spacing / len(hue_levels)
        offset = width / len(hue_levels)
        ax = violinplot(long_df, x="s", y="z", hue="c", native_scale=True, cut=0)
        violins = iter(ax.collections)
        for center in centers:
            for i, hue_level in enumerate(hue_levels):
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = center + [-offset, +offset][i]
                poly = next(violins)
                self.check_violin(poly, data, "x", pos, width)

    def test_dodge_native_scale_log(self, long_df):

        pos = 10 ** long_df["s"]
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        variables = dict(x=pos, y="z", hue="c")
        violinplot(long_df, **variables, native_scale=True, density_norm="width", ax=ax)
        widths = []
        n_violins = long_df["s"].nunique() * long_df["c"].nunique()
        for poly in ax.collections[:n_violins]:
            verts = poly.get_paths()[0].vertices[:, 0]
            coords = np.log10(verts)
            widths.append(np.ptp(coords))
        assert np.std(widths) == approx(0)

    def test_color(self, long_df):

        color = "#123456"
        ax = violinplot(long_df, x="a", y="y", color=color, saturation=1)
        for poly in ax.collections:
            assert same_color(poly.get_facecolor(), color)

    def test_hue_colors(self, long_df):

        ax = violinplot(long_df, x="a", y="y", hue="b", saturation=1)
        n_levels = long_df["b"].nunique()
        for i, poly in enumerate(ax.collections):
            assert same_color(poly.get_facecolor(), f"C{i % n_levels}")

    @pytest.mark.parametrize("inner", ["box", "quart", "stick", "point"])
    def test_linecolor(self, long_df, inner):

        color = "#669913"
        ax = violinplot(long_df, x="a", y="y", linecolor=color, inner=inner)
        for poly in ax.findobj(mpl.collections.PolyCollection):
            assert same_color(poly.get_edgecolor(), color)
        for lines in ax.findobj(mpl.collections.LineCollection):
            assert same_color(lines.get_color(), color)
        for line in ax.lines:
            assert same_color(line.get_color(), color)

    def test_linewidth(self, long_df):

        width = 5
        ax = violinplot(long_df, x="a", y="y", linewidth=width)
        poly = ax.collections[0]
        assert poly.get_linewidth() == width

    def test_saturation(self, long_df):

        color = "#8912b0"
        ax = violinplot(long_df["x"], color=color, saturation=.5)
        poly = ax.collections[0]
        assert np.allclose(poly.get_facecolors()[0, :3], desaturate(color, 0.5))

    @pytest.mark.parametrize("inner", ["box", "quart", "stick", "point"])
    def test_fill(self, long_df, inner):

        color = "#459900"
        ax = violinplot(x=long_df["z"], fill=False, color=color, inner=inner)
        for poly in ax.findobj(mpl.collections.PolyCollection):
            assert poly.get_facecolor().size == 0
            assert same_color(poly.get_edgecolor(), color)
        for lines in ax.findobj(mpl.collections.LineCollection):
            assert same_color(lines.get_color(), color)
        for line in ax.lines:
            assert same_color(line.get_color(), color)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_box(self, long_df, orient):

        pos_idx, val_idx = self.orient_indices(orient)
        ax = violinplot(long_df["y"], orient=orient)
        stats = mpl.cbook.boxplot_stats(long_df["y"])[0]

        whiskers = ax.lines[0].get_xydata()
        assert whiskers[0, val_idx] == stats["whislo"]
        assert whiskers[1, val_idx] == stats["whishi"]
        assert whiskers[:, pos_idx].tolist() == [0, 0]

        box = ax.lines[1].get_xydata()
        assert box[0, val_idx] == stats["q1"]
        assert box[1, val_idx] == stats["q3"]
        assert box[:, pos_idx].tolist() == [0, 0]

        median = ax.lines[2].get_xydata()
        assert median[0, val_idx] == stats["med"]
        assert median[0, pos_idx] == 0

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_quartiles(self, long_df, orient):

        pos_idx, val_idx = self.orient_indices(orient)
        ax = violinplot(long_df["y"], orient=orient, inner="quart")
        quartiles = np.percentile(long_df["y"], [25, 50, 75])

        for q, line in zip(quartiles, ax.lines):
            pts = line.get_xydata()
            for pt in pts:
                assert pt[val_idx] == q
            assert pts[0, pos_idx] == -pts[1, pos_idx]

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_stick(self, long_df, orient):

        pos_idx, val_idx = self.orient_indices(orient)
        ax = violinplot(long_df["y"], orient=orient, inner="stick")
        for i, pts in enumerate(ax.collections[1].get_segments()):
            for pt in pts:
                assert pt[val_idx] == long_df["y"].iloc[i]
            assert pts[0, pos_idx] == -pts[1, pos_idx]

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_points(self, long_df, orient):

        pos_idx, val_idx = self.orient_indices(orient)
        ax = violinplot(long_df["y"], orient=orient, inner="points")
        points = ax.collections[1]
        for i, pt in enumerate(points.get_offsets()):
            assert pt[val_idx] == long_df["y"].iloc[i]
            assert pt[pos_idx] == 0

    def test_split_single(self, long_df):

        ax = violinplot(long_df, x="a", y="z", split=True, cut=0)
        levels = categorical_order(long_df["a"])
        for i, level in enumerate(levels):
            data = long_df.loc[long_df["a"] == level, "z"]
            self.check_violin(ax.collections[i], data, "x", i)
            verts = ax.collections[i].get_paths()[0].vertices
            assert np.isclose(verts[:, 0], i + .4).sum() >= 100

    def test_split_multi(self, long_df):

        ax = violinplot(long_df, x="a", y="z", hue="c", split=True, cut=0)
        polys = iter(ax.collections)
        for i, level in enumerate(categorical_order(long_df["a"])):
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                pos = i + [-.2, +.2][j]
                poly = next(polys)
                self.check_violin(poly, data, "x", pos, width=0.4)
                verts = poly.get_paths()[0].vertices
                assert np.isclose(verts[:, 0], i).sum() >= 100

    def test_density_norm_area(self, long_df):

        y = long_df["y"].to_numpy()
        ax = violinplot([y, y * 5])
        widths = []
        for poly in ax.collections:
            widths.append(self.violin_width(poly))
        assert widths[0] / widths[1] == approx(5)

    def test_density_norm_count(self, long_df):

        y = long_df["y"].to_numpy()
        ax = violinplot([np.repeat(y, 3), y], density_norm="count")
        widths = []
        for poly in ax.collections:
            widths.append(self.violin_width(poly))
        assert widths[0] / widths[1] == approx(3)

    def test_density_norm_width(self, long_df):

        ax = violinplot(long_df, x="a", y="y", density_norm="width")
        for poly in ax.collections:
            assert self.violin_width(poly) == approx(0.8)

    def test_common_norm(self, long_df):

        ax = violinplot(long_df, x="a", y="y", hue="c", common_norm=True, legend=False)
        widths = []
        for poly in ax.collections:
            widths.append(self.violin_width(poly))
        assert sum(w > 0.3999 for w in widths) == 1

    def test_scale_deprecation(self, long_df):

        with pytest.warns(FutureWarning, match=r".+Pass `density_norm='count'`"):
            violinplot(long_df, x="a", y="y", hue="b", scale="count")

    def test_scale_hue_deprecation(self, long_df):

        with pytest.warns(FutureWarning, match=r".+Pass `common_norm=True`"):
            violinplot(long_df, x="a", y="y", hue="b", scale_hue=False)

    def test_bw_adjust(self, long_df):

        ax = violinplot(long_df["y"], bw_adjust=.2)
        violinplot(long_df["y"], bw_adjust=2)
        kde1 = ax.collections[0].get_paths()[0].vertices[:100, 0]
        kde2 = ax.collections[1].get_paths()[0].vertices[:100, 0]
        assert np.std(np.diff(kde1)) > np.std(np.diff(kde2))

    def test_bw_deprecation(self, long_df):

        with pytest.warns(FutureWarning, match=r".*Setting `bw_method='silverman'`"):
            violinplot(long_df["y"], bw="silverman")

    def test_gap(self, long_df):

        ax = violinplot(long_df, y="y", hue="c", gap=.2)
        a = ax.collections[0].get_paths()[0].vertices[:, 0].max()
        b = ax.collections[1].get_paths()[0].vertices[:, 0].min()
        assert (b - a) == approx(0.2 * 0.8 / 2)

    def test_inner_kws(self, long_df):

        kws = {"linewidth": 3}
        ax = violinplot(long_df, x="a", y="y", inner="stick", inner_kws=kws)
        for line in ax.lines:
            assert line.get_linewidth() == kws["linewidth"]

    def test_box_inner_kws(self, long_df):

        kws = {"box_width": 10, "whis_width": 2, "marker": "x"}
        ax = violinplot(long_df, x="a", y="y", inner_kws=kws)
        for line in ax.lines[::3]:
            assert line.get_linewidth() == kws["whis_width"]
        for line in ax.lines[1::3]:
            assert line.get_linewidth() == kws["box_width"]
        for line in ax.lines[2::3]:
            assert line.get_marker() == kws["marker"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y", split=True),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s", split=True),
            dict(data="null", x="a", y="y", hue="a"),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),
            dict(data="long", x="a", y="y", inner="stick"),
            dict(data="long", x="a", y="y", inner="points"),
            dict(data="long", x="a", y="y", hue="b", inner="quartiles", split=True),
            dict(data="long", x="a", y="y", density_norm="count", common_norm=True),
            dict(data="long", x="a", y="y", bw=2),
            dict(data="long", x="a", y="y", bw_adjust=2),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = violinplot(**kwargs)
        g = catplot(**kwargs, kind="violin")

        assert_plots_equal(ax, g.ax)


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
    def test_vector_orient(self, orient):

        keys, vals = ["a", "b", "c"], [1, 2, 3]
        data = dict(zip(keys, vals))
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        prop = {"x": "height", "y": "width"}[orient]
        ax = barplot(data, orient=orient)
        for i, bar in enumerate(ax.patches):
            assert getattr(bar, f"get_{orient}")() == approx(i - 0.4)
            assert getattr(bar, f"get_{prop}")() == approx(vals[i])

    def test_xy_vertical(self):

        x, y = ["a", "b", "c"], [1, 3, 2.5]

        ax = barplot(x=x, y=y)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == approx(0)
            assert bar.get_height() == approx(y[i])
            assert bar.get_width() == approx(0.8)

    def test_xy_horizontal(self):

        x, y = [1, 3, 2.5], ["a", "b", "c"]

        ax = barplot(x=x, y=y)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == approx(0)
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)
            assert bar.get_width() == approx(x[i])

    def test_xy_with_na_grouper(self):

        x, y = ["a", None, "b"], [1, 2, 3]
        ax = barplot(x=x, y=y)
        _draw_figure(ax.figure)  # For matplotlib<3.5
        assert ax.get_xticks() == [0, 1]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
        assert ax.patches[0].get_height() == 1
        assert ax.patches[1].get_height() == 3

    def test_xy_with_na_value(self):

        x, y = ["a", "b", "c"], [1, None, 3]
        ax = barplot(x=x, y=y)
        _draw_figure(ax.figure)  # For matplotlib<3.5
        assert ax.get_xticks() == [0, 1, 2]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
        assert ax.patches[0].get_height() == 1
        assert ax.patches[1].get_height() == 3

    def test_hue_redundant(self):

        x, y = ["a", "b", "c"], [1, 2, 3]

        ax = barplot(x=x, y=y, hue=x, saturation=1)
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

    def test_gap(self):

        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, gap=.25)
        for i, bar in enumerate(ax.patches):
            assert bar.get_width() == approx(0.8 / 2 * .75)

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

    def test_hue_order(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        hue_order = ["c", "b", "a"]

        ax = barplot(x=x, y=y, hue=x, hue_order=hue_order, saturation=1)
        for i, bar in enumerate(ax.patches):
            assert same_color(bar.get_facecolor(), f"C{i}")
            assert bar.get_x() + bar.get_width() / 2 == approx(2 - i)

    def test_hue_norm(self):

        x, y = [1, 2, 3, 4], [1, 2, 3, 4]

        ax = barplot(x=x, y=y, hue=x, hue_norm=(2, 3))
        colors = [bar.get_facecolor() for bar in ax.patches]
        assert colors[0] == colors[1]
        assert colors[1] != colors[2]
        assert colors[2] == colors[3]

    def test_fill(self):

        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, fill=False)
        for i, bar in enumerate(ax.patches):
            assert same_color(bar.get_edgecolor(), f"C{i // 2}")
            assert same_color(bar.get_facecolor(), (0, 0, 0, 0))

    def test_xy_native_scale(self):

        x, y = [2, 4, 8], [1, 2, 3]

        ax = barplot(x=x, y=y, native_scale=True)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(x[i])
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8 * 2)

    def test_xy_native_scale_log_transform(self):

        x, y = [1, 10, 100], [1, 2, 3]

        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        barplot(x=x, y=y, native_scale=True, ax=ax)
        for i, bar in enumerate(ax.patches):
            x0, x1 = np.log10([bar.get_x(), bar.get_x() + bar.get_width()])
            center = 10 ** (x0 + (x1 - x0) / 2)
            assert center == approx(x[i])
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
        assert ax.patches[1].get_width() > ax.patches[0].get_width()

    def test_datetime_native_scale_axis(self):

        x = pd.date_range("2010-01-01", periods=20, freq="m")
        y = np.arange(20)
        ax = barplot(x=x, y=y, native_scale=True)
        assert "Date" in ax.xaxis.get_major_locator().__class__.__name__
        day = "2003-02-28"
        assert_array_equal(ax.xaxis.convert_units([day]), mpl.dates.date2num([day]))

    def test_native_scale_dodged(self):

        x, y = [2, 4, 2, 4], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = barplot(x=x, y=y, hue=hue, native_scale=True)

        for x_i, bar in zip(x[:2], ax.patches[:2]):
            assert bar.get_x() + bar.get_width() == approx(x_i)
        for x_i, bar in zip(x[2:], ax.patches[2:]):
            assert bar.get_x() == approx(x_i)

    def test_native_scale_log_transform_dodged(self):

        x, y = [1, 100, 1, 100], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        barplot(x=x, y=y, hue=hue, native_scale=True, ax=ax)

        for x_i, bar in zip(x[:2], ax.patches[:2]):
            assert bar.get_x() + bar.get_width() == approx(x_i)
        for x_i, bar in zip(x[2:], ax.patches[2:]):
            assert bar.get_x() == approx(x_i)

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

    def test_estimate_log_transform(self, long_df):

        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        barplot(x=long_df["z"], ax=ax)
        bar, = ax.patches
        assert bar.get_width() == 10 ** np.log10(long_df["z"]).mean()

    def test_errorbars(self, long_df):

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
        for bar in ax.patches:
            assert bar.get_width() == (width * 2)

    def test_width_spaced_categories(self):

        ax = barplot(x=["a", "b", "c"], y=[4, 5, 6])
        barplot(x=["a", "c"], y=[1, 3], ax=ax)
        for bar in ax.patches:
            assert bar.get_width() == pytest.approx(0.8)

    def test_saturation_color(self):

        color = (.1, .9, .2)
        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = barplot(x=x, y=y)
        for bar in ax.patches:
            assert np.var(bar.get_facecolor()[:3]) < np.var(color)

    def test_saturation_palette(self):

        palette = color_palette("viridis", 3)
        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = barplot(x=x, y=y, hue=x, palette=palette)
        for i, bar in enumerate(ax.patches):
            assert np.var(bar.get_facecolor()[:3]) < np.var(palette[i])

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

    def test_error_caps(self):

        x, y = ["a", "b", "c"] * 2, [1, 2, 3, 4, 5, 6]
        ax = barplot(x=x, y=y, capsize=.8, errorbar="pi")

        assert len(ax.patches) == len(ax.lines)
        for bar, error in zip(ax.patches, ax.lines):
            pos = error.get_xdata()
            assert len(pos) == 8
            assert np.nanmin(pos) == approx(bar.get_x())
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())

    def test_error_caps_native_scale(self):

        x, y = [2, 4, 20] * 2, [1, 2, 3, 4, 5, 6]
        ax = barplot(x=x, y=y, capsize=.8, native_scale=True, errorbar="pi")

        assert len(ax.patches) == len(ax.lines)
        for bar, error in zip(ax.patches, ax.lines):
            pos = error.get_xdata()
            assert len(pos) == 8
            assert np.nanmin(pos) == approx(bar.get_x())
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())

    def test_error_caps_native_scale_log_transform(self):

        x, y = [1, 10, 1000] * 2, [1, 2, 3, 4, 5, 6]
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        barplot(x=x, y=y, capsize=.8, native_scale=True, errorbar="pi", ax=ax)

        assert len(ax.patches) == len(ax.lines)
        for bar, error in zip(ax.patches, ax.lines):
            pos = error.get_xdata()
            assert len(pos) == 8
            assert np.nanmin(pos) == approx(bar.get_x())
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())

    def test_bar_kwargs(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        kwargs = dict(linewidth=3, facecolor=(.5, .4, .3, .2), rasterized=True)
        ax = barplot(x=x, y=y, **kwargs)
        for bar in ax.patches:
            assert bar.get_linewidth() == kwargs["linewidth"]
            assert bar.get_facecolor() == kwargs["facecolor"]
            assert bar.get_rasterized() == kwargs["rasterized"]

    @pytest.mark.parametrize("fill", [True, False])
    def test_err_kws(self, fill):

        x, y = ["a", "b", "c"], [1, 2, 3]
        err_kws = dict(color=(1, 1, .5, .5), linewidth=5)
        ax = barplot(x=x, y=y, fill=fill, err_kws=err_kws)
        for line in ax.lines:
            assert line.get_color() == err_kws["color"]
            assert line.get_linewidth() == err_kws["linewidth"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s"),
            dict(data="long", x="a", y="y", units="c"),
            dict(data="null", x="a", y="y", hue="a", gap=.1, fill=False),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="long", x="a", y="y", errorbar=("pi", 50)),
            dict(data="long", x="a", y="y", errorbar=None),
            dict(data="long", x="a", y="y", capsize=.3, err_kws=dict(c="k")),
            dict(data="long", x="a", y="y", color="blue", ec="green", alpha=.5),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        kwargs = kwargs.copy()
        kwargs["seed"] = 0
        kwargs["n_boot"] = 10

        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = barplot(**kwargs)
        g = catplot(**kwargs, kind="bar")

        assert_plots_equal(ax, g.ax)

    def test_errwidth_deprecation(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        val = 5
        with pytest.warns(FutureWarning, match="\n\nThe `errwidth` parameter"):
            ax = barplot(x=x, y=y, errwidth=val)
        for line in ax.lines:
            assert line.get_linewidth() == val

    def test_errcolor_deprecation(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        val = (1, .7, .4, .8)
        with pytest.warns(FutureWarning, match="\n\nThe `errcolor` parameter"):
            ax = barplot(x=x, y=y, errcolor=val)
        for line in ax.lines:
            assert line.get_color() == val

    def test_capsize_as_none_deprecation(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        with pytest.warns(FutureWarning, match="\n\nPassing `capsize=None`"):
            ax = barplot(x=x, y=y, capsize=None)
        for line in ax.lines:
            assert len(line.get_xdata()) == 2

    def test_hue_implied_by_palette_deprecation(self):

        x = ["a", "b", "c"]
        y = [1, 2, 3]
        palette = "Set1"
        colors = color_palette(palette, len(x))
        msg = "Passing `palette` without assigning `hue` is deprecated."
        with pytest.warns(FutureWarning, match=msg):
            ax = barplot(x=x, y=y, saturation=1, palette=palette)
        for i, bar in enumerate(ax.patches):
            assert same_color(bar.get_facecolor(), colors[i])


class TestPointPlot(SharedAggTests):

    func = staticmethod(pointplot)

    def get_last_color(self, ax):

        color = ax.lines[-1].get_color()
        return to_rgba(color)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_single_var(self, orient):

        vals = pd.Series([1, 3, 10])
        ax = pointplot(**{orient: vals})
        line = ax.lines[0]
        assert getattr(line, f"get_{orient}data")() == approx(vals.mean())

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_wide_df(self, wide_df, orient):

        ax = pointplot(wide_df, orient=orient)
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        depend = {"x": "y", "y": "x"}[orient]
        line = ax.lines[0]
        assert_array_equal(
            getattr(line, f"get_{orient}data")(),
            np.arange(len(wide_df.columns)),
        )
        assert_array_almost_equal(
            getattr(line, f"get_{depend}data")(),
            wide_df.mean(axis=0),
        )

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_vector_orient(self, orient):

        keys, vals = ["a", "b", "c"], [1, 2, 3]
        data = dict(zip(keys, vals))
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        depend = {"x": "y", "y": "x"}[orient]
        ax = pointplot(data, orient=orient)
        line = ax.lines[0]
        assert_array_equal(
            getattr(line, f"get_{orient}data")(),
            np.arange(len(keys)),
        )
        assert_array_equal(getattr(line, f"get_{depend}data")(), vals)

    def test_xy_vertical(self):

        x, y = ["a", "b", "c"], [1, 3, 2.5]
        ax = pointplot(x=x, y=y)
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i, y[i])

    def test_xy_horizontal(self):

        x, y = [1, 3, 2.5], ["a", "b", "c"]
        ax = pointplot(x=x, y=y)
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (x[i], i)

    def test_xy_with_na_grouper(self):

        x, y = ["a", None, "b"], [1, 2, 3]
        ax = pointplot(x=x, y=y)
        _draw_figure(ax.figure)  # For matplotlib<3.5
        assert ax.get_xticks() == [0, 1]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
        assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        assert_array_equal(ax.lines[0].get_ydata(), [1, 3])

    def test_xy_with_na_value(self):

        x, y = ["a", "b", "c"], [1, np.nan, 3]
        ax = pointplot(x=x, y=y)
        _draw_figure(ax.figure)  # For matplotlib<3.5
        assert ax.get_xticks() == [0, 1, 2]
        assert [t.get_text() for t in ax.get_xticklabels()] == x
        assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
        assert_array_equal(ax.lines[0].get_ydata(), y)

    def test_hue(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        ax = pointplot(x=x, y=y, hue=hue, errorbar=None)
        for i, line in enumerate(ax.lines[:2]):
            assert_array_equal(line.get_ydata(), y[i::2])
            assert same_color(line.get_color(), f"C{i}")

    def test_xy_native_scale(self):

        x, y = [2, 4, 8], [1, 2, 3]

        ax = pointplot(x=x, y=y, native_scale=True)
        line = ax.lines[0]
        assert_array_equal(line.get_xdata(), x)
        assert_array_equal(line.get_ydata(), y)

    @pytest.mark.parametrize("estimator", ["mean", np.mean])
    def test_estimate(self, long_df, estimator):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].agg(estimator)

        ax = pointplot(long_df, x=agg_var, y=val_var, errorbar=None)
        order = categorical_order(long_df[agg_var])
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == approx((i, agg_df[order[i]]))

    def test_estimate_log_transform(self, long_df):

        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        pointplot(x=long_df["z"], ax=ax)
        val, = ax.lines[0].get_xdata()
        assert val == 10 ** np.log10(long_df["z"]).mean()

    def test_errorbars(self, long_df):

        agg_var, val_var = "a", "y"
        agg_df = long_df.groupby(agg_var)[val_var].agg(["mean", "std"])

        ax = pointplot(long_df, x=agg_var, y=val_var, errorbar="sd")
        order = categorical_order(long_df[agg_var])
        for i, line in enumerate(ax.lines[1:]):
            row = agg_df.loc[order[i]]
            lo, hi = line.get_ydata()
            assert lo == approx(row["mean"] - row["std"])
            assert hi == approx(row["mean"] + row["std"])

    def test_marker_linestyle(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = pointplot(x=x, y=y, marker="s", linestyle="--")
        line = ax.lines[0]
        assert line.get_marker() == "s"
        assert line.get_linestyle() == "--"

    def test_markers_linestyles_single(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = pointplot(x=x, y=y, markers="s", linestyles="--")
        line = ax.lines[0]
        assert line.get_marker() == "s"
        assert line.get_linestyle() == "--"

    def test_markers_linestyles_mapped(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        markers = ["d", "s"]
        linestyles = ["--", ":"]
        ax = pointplot(
            x=x, y=y, hue=hue,
            markers=markers, linestyles=linestyles,
            errorbar=None,
        )
        for i, line in enumerate(ax.lines[:2]):
            assert line.get_marker() == markers[i]
            assert line.get_linestyle() == linestyles[i]

    def test_dodge_boolean(self):

        x, y = ["a", "b", "a", "b"], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        ax = pointplot(x=x, y=y, hue=hue, dodge=True, errorbar=None)
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i - .025, y[i])
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == (i + .025, y[2 + i])

    def test_dodge_float(self):

        x, y = ["a", "b", "a", "b"], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        ax = pointplot(x=x, y=y, hue=hue, dodge=.2, errorbar=None)
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i - .1, y[i])
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == (i + .1, y[2 + i])

    def test_dodge_log_scale(self):

        x, y = [10, 1000, 10, 1000], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")
        pointplot(x=x, y=y, hue=hue, dodge=.2, native_scale=True, errorbar=None, ax=ax)
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == approx((10 ** (np.log10(x[i]) - .2), y[i]))
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == approx((10 ** (np.log10(x[2 + i]) + .2), y[2 + i]))

    def test_err_kws(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        err_kws = dict(color=(.2, .5, .3), linewidth=10)
        ax = pointplot(x=x, y=y, errorbar=("pi", 100), err_kws=err_kws)
        for line in ax.lines[1:]:
            assert same_color(line.get_color(), err_kws["color"])
            assert line.get_linewidth() == err_kws["linewidth"]

    def test_err_kws_inherited(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        kws = dict(color=(.2, .5, .3), linewidth=10)
        ax = pointplot(x=x, y=y, errorbar=("pi", 100), **kws)
        for line in ax.lines[1:]:
            assert same_color(line.get_color(), kws["color"])
            assert line.get_linewidth() == kws["linewidth"]

    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Legend handle missing marker property"
    )
    def test_legend_contents(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        ax = pointplot(x=x, y=y, hue=hue)
        _draw_figure(ax.figure)
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.texts] == ["x", "y"]
        for i, handle in enumerate(get_legend_handles(legend)):
            assert handle.get_marker() == "o"
            assert handle.get_linestyle() == "-"
            assert same_color(handle.get_color(), f"C{i}")

    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Legend handle missing marker property"
    )
    def test_legend_set_props(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        kws = dict(marker="s", linewidth=1)
        ax = pointplot(x=x, y=y, hue=hue, **kws)
        legend = ax.get_legend()
        for i, handle in enumerate(get_legend_handles(legend)):
            assert handle.get_marker() == kws["marker"]
            assert handle.get_linewidth() == kws["linewidth"]

    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Legend handle missing marker property"
    )
    def test_legend_synced_props(self):

        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        kws = dict(markers=["s", "d"], linestyles=["--", ":"])
        ax = pointplot(x=x, y=y, hue=hue, **kws)
        legend = ax.get_legend()
        for i, handle in enumerate(get_legend_handles(legend)):
            assert handle.get_marker() == kws["markers"][i]
            assert handle.get_linestyle() == kws["linestyles"][i]

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s"),
            dict(data="long", x="a", y="y", units="c"),
            dict(data="null", x="a", y="y", hue="a"),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="long", x="a", y="y", errorbar=("pi", 50)),
            dict(data="long", x="a", y="y", errorbar=None),
            dict(data="null", x="a", y="y", hue="a", dodge=True),
            dict(data="null", x="a", y="y", hue="a", dodge=.2),
            dict(data="long", x="a", y="y", capsize=.3, err_kws=dict(c="k")),
            dict(data="long", x="a", y="y", color="blue", marker="s"),
            dict(data="long", x="a", y="y", hue="a", markers=["s", "d", "p"]),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        kwargs = kwargs.copy()
        kwargs["seed"] = 0
        kwargs["n_boot"] = 10

        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = pointplot(**kwargs)
        g = catplot(**kwargs, kind="point")

        assert_plots_equal(ax, g.ax)

    def test_legend_disabled(self, long_df):

        ax = pointplot(long_df, x="x", y="y", hue="b", legend=False)
        assert ax.get_legend() is None

    def test_join_deprecation(self):

        with pytest.warns(UserWarning, match="The `join` parameter"):
            ax = pointplot(x=["a", "b", "c"], y=[1, 2, 3], join=False)
        assert ax.lines[0].get_linestyle().lower() == "none"

    def test_scale_deprecation(self):

        x, y = ["a", "b", "c"], [1, 2, 3]
        ax = pointplot(x=x, y=y, errorbar=None)
        with pytest.warns(UserWarning, match="The `scale` parameter"):
            pointplot(x=x, y=y, errorbar=None, scale=2)
        l1, l2 = ax.lines
        assert l2.get_linewidth() == 2 * l1.get_linewidth()
        assert l2.get_markersize() > l1.get_markersize()

    def test_layered_plot_clipping(self):

        x, y = ['a'], [4]
        pointplot(x=x, y=y)
        x, y = ['b'], [5]
        ax = pointplot(x=x, y=y)
        y_range = ax.viewLim.intervaly
        assert y_range[0] < 4 and y_range[1] > 5


class TestCountPlot:

    def test_empty(self):

        ax = countplot()
        assert not ax.patches

        ax = countplot(x=[])
        assert not ax.patches

    def test_labels_long(self, long_df):

        fig = mpl.figure.Figure()
        axs = fig.subplots(2)
        countplot(long_df, x="a", ax=axs[0])
        countplot(long_df, x="b", stat="percent", ax=axs[1])

        # To populate texts; only needed on older matplotlibs
        _draw_figure(fig)

        assert axs[0].get_xlabel() == "a"
        assert axs[1].get_xlabel() == "b"
        assert axs[0].get_ylabel() == "count"
        assert axs[1].get_ylabel() == "percent"

    def test_wide_data(self, wide_df):

        ax = countplot(wide_df)
        assert len(ax.patches) == len(wide_df.columns)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == len(wide_df)
            assert bar.get_width() == approx(0.8)

    def test_flat_series(self):

        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        ax = countplot(vals)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == 0
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)
            assert bar.get_width() == counts[i]

    def test_x_series(self):

        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        ax = countplot(x=vals)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == counts[i]
            assert bar.get_width() == approx(0.8)

    def test_y_series(self):

        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        ax = countplot(y=vals)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == 0
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)
            assert bar.get_width() == counts[i]

    def test_hue_redundant(self):

        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])

        ax = countplot(x=vals, hue=vals, saturation=1)
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == counts[i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i}")

    def test_hue_dodged(self):

        vals = ["a", "a", "a", "b", "b", "b"]
        hue = ["x", "y", "y", "x", "x", "x"]
        counts = [1, 3, 2, 0]

        ax = countplot(x=vals, hue=hue, saturation=1)
        for i, bar in enumerate(ax.patches):
            sign = 1 if i // 2 else -1
            assert (
                bar.get_x() + bar.get_width() / 2
                == approx(i % 2 + sign * 0.8 / 4)
            )
            assert bar.get_y() == 0
            assert bar.get_height() == counts[i]
            assert bar.get_width() == approx(0.8 / 2)
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    @pytest.mark.parametrize("stat", ["percent", "probability", "proportion"])
    def test_stat(self, long_df, stat):

        col = "a"
        order = categorical_order(long_df[col])
        expected = long_df[col].value_counts(normalize=True)
        if stat == "percent":
            expected *= 100
        ax = countplot(long_df, x=col, stat=stat)
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(expected[order[i]])

    def test_xy_error(self, long_df):

        with pytest.raises(TypeError, match="Cannot pass values for both"):
            countplot(long_df, x="a", y="b")

    def test_legend_numeric_auto(self, long_df):

        ax = countplot(long_df, x="x", hue="x")
        assert len(ax.get_legend().texts) <= 6

    def test_legend_disabled(self, long_df):

        ax = countplot(long_df, x="x", hue="b", legend=False)
        assert ax.get_legend() is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a"),
            dict(data=None, x="a"),
            dict(data="long", y="b"),
            dict(data="long", x="a", hue="a"),
            dict(data=None, x="a", hue="a"),
            dict(data="long", x="a", hue="b"),
            dict(data=None, x="s", hue="a"),
            dict(data="long", x="a", hue="s"),
            dict(data="null", x="a", hue="a"),
            dict(data="long", x="s", hue="a", native_scale=True),
            dict(data="long", x="d", hue="a", native_scale=True),
            dict(data="long", x="a", stat="percent"),
            dict(data="long", x="a", hue="b", stat="proportion"),
            dict(data="long", x="a", color="blue", ec="green", alpha=.5),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):

        kwargs = kwargs.copy()
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        ax = countplot(**kwargs)
        g = catplot(**kwargs, kind="count")

        assert_plots_equal(ax, g.ax)


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
        want_lines = 1 + self.g.unique().size
        assert len(g.ax.lines) == want_lines

        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="point")
        want_lines = (
            len(self.g.unique()) * len(self.h.unique()) + 2 * len(self.h.unique())
        )
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
        assert len(g.ax.collections) == want_elements + self.h.unique().size

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

        # Test unsharing works
        g = cat.catplot(
            x="g", y="y", col="g", data=self.df, sharex=False, kind="bar",
        )
        for ax in g.axes.flat:
            assert len(ax.patches) == 1

        g = cat.catplot(
            x="y", y="g", col="g", data=self.df, sharey=False, kind="bar",
        )
        for ax in g.axes.flat:
            assert len(ax.patches) == 1

        g = cat.catplot(
            x="g", y="y", col="g", data=self.df, sharex=False, color="b"
        )
        for ax in g.axes.flat:
            assert ax.get_xlim() == (-.5, .5)

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


class TestBoxPlotContainer:

    @pytest.fixture
    def container(self, wide_array):

        ax = mpl.figure.Figure().subplots()
        artist_dict = ax.boxplot(wide_array)
        return BoxPlotContainer(artist_dict)

    def test_repr(self, container, wide_array):

        n = wide_array.shape[1]
        assert str(container) == f"<BoxPlotContainer object with {n} boxes>"

    def test_iteration(self, container):
        for artist_tuple in container:
            for attr in ["box", "median", "whiskers", "caps", "fliers", "mean"]:
                assert hasattr(artist_tuple, attr)

    def test_label(self, container):

        label = "a box plot"
        container.set_label(label)
        assert container.get_label() == label

    def test_children(self, container):

        children = container.get_children()
        for child in children:
            assert isinstance(child, mpl.artist.Artist)
