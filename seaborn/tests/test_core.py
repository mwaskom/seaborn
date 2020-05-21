import numpy as np
import pandas as pd
import matplotlib as mpl
from cycler import cycler

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from ..core import (
    HueMapping,
    _VectorPlotter,
    variable_type,
    infer_orient,
    unique_dashes,
    unique_markers,
    categorical_order,
    get_color_cycle,
    remove_na,
)

from ..palettes import color_palette


class TestHueMapping:

    def test_hue_map_null(self, long_df, null_series):

        p = _VectorPlotter(variables=dict(hue=null_series))
        m = HueMapping(p)
        assert m.levels == [None]
        assert m.palette is None
        assert m.map_type is None
        assert m.cmap is None
        assert m.lookup_table == {}

    def test_hue_map_categorical(self, wide_df, long_df):

        p = _VectorPlotter(data=wide_df)
        m = HueMapping(p)
        assert m.levels == wide_df.columns.tolist()
        assert m.map_type == "categorical"
        assert m.cmap is None

        # Test named palette
        palette = "Blues"
        expected_colors = color_palette(palette, wide_df.shape[1])
        expected_lookup_table = dict(zip(wide_df.columns, expected_colors))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == expected_lookup_table

        # Test list palette
        palette = color_palette("Reds", wide_df.shape[1])
        expected_lookup_table = dict(zip(wide_df.columns, palette))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == expected_lookup_table

        # Test dict palette
        colors = color_palette("Set1", 8)
        palette = dict(zip(wide_df.columns, colors))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == palette

        # Test dict with missing keys
        palette = dict(zip(wide_df.columns[:-1], colors))
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # Test dict with missing keys
        palette = dict(zip(wide_df.columns[:-1], colors))
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # Test list with wrong number of colors
        palette = colors[:-1]
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # Test hue order
        hue_order = ["a", "c", "d"]
        m = HueMapping(p, order=hue_order)
        assert m.levels == hue_order

        # Test long data
        p = _VectorPlotter(data=long_df, variables=dict(x="x", y="y", hue="a"))
        m = HueMapping(p)
        assert m.levels == categorical_order(long_df.a)
        assert m.map_type == "categorical"
        assert m.cmap is None

        # Test default palette
        m = HueMapping(p)
        hue_levels = categorical_order(long_df.a)
        expected_colors = color_palette(n_colors=len(hue_levels))
        expected_lookup_table = dict(zip(hue_levels, expected_colors))
        assert m.lookup_table == expected_lookup_table

        # Test default palette with many levels
        x = y = np.arange(26)
        hue = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        p = _VectorPlotter(variables=dict(x=x, y=y, hue=hue))
        m = HueMapping(p)
        expected_colors = color_palette("husl", n_colors=len(hue))
        expected_lookup_table = dict(zip(hue, expected_colors))
        assert m.lookup_table == expected_lookup_table

        # Test binary data
        p = _VectorPlotter(data=long_df, variables=dict(x="x", y="y", hue="c"))
        m = HueMapping(p)
        assert m.levels == [0, 1]
        assert m.map_type == "categorical"

        for val in [0, 1]:
            p = _VectorPlotter(
                data=long_df[long_df["c"] == val],
                variables=dict(x="x", y="y", hue="c"),
            )
            m = HueMapping(p)
            assert m.levels == [val]
            assert m.map_type == "categorical"

        # Test Timestamp data
        p = _VectorPlotter(data=long_df, variables=dict(x="x", y="y", hue="t"))
        m = HueMapping(p)
        assert m.levels == [pd.Timestamp('2005-02-25')]
        assert m.map_type == "datetime"

        # Test numeric data with category type
        p = _VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="s_cat")
        )
        m = HueMapping(p)
        assert m.levels == categorical_order(long_df["s_cat"])
        assert m.map_type == "categorical"
        assert m.cmap is None

        # Test categorical palette specified for numeric data
        p = _VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="s")
        )
        palette = "deep"
        levels = categorical_order(long_df["s"])
        expected_colors = color_palette(palette, n_colors=len(levels))
        expected_lookup_table = dict(zip(levels, expected_colors))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == expected_lookup_table
        assert m.map_type == "categorical"

    def test_hue_map_numeric(self, long_df):

        # Test default colormap
        p = _VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="s")
        )
        hue_levels = list(np.sort(long_df["s"].unique()))
        m = HueMapping(p)
        assert m.levels == hue_levels
        assert m.map_type == "numeric"
        assert m.cmap.name == "seaborn_cubehelix"

        # Test named colormap
        palette = "Purples"
        m = HueMapping(p, palette=palette)
        assert m.cmap is mpl.cm.get_cmap(palette)

        # Test colormap object
        palette = mpl.cm.get_cmap("Greens")
        m = HueMapping(p, palette=palette)
        assert m.cmap is mpl.cm.get_cmap(palette)

        # Test cubehelix shorthand
        palette = "ch:2,0,light=.2"
        m = HueMapping(p, palette=palette)
        assert isinstance(m.cmap, mpl.colors.ListedColormap)

        # Test default hue limits
        m = HueMapping(p)
        data_range = p.plot_data["hue"].min(), p.plot_data["hue"].max()
        assert m.limits == data_range

        # Test specified hue limits
        hue_norm = 1, 4
        m = HueMapping(p, norm=hue_norm)
        assert m.limits == hue_norm
        assert isinstance(m.norm, mpl.colors.Normalize)
        assert m.norm.vmin == hue_norm[0]
        assert m.norm.vmax == hue_norm[1]

        # Test Normalize object
        hue_norm = mpl.colors.PowerNorm(2, vmin=1, vmax=10)
        m = HueMapping(p, norm=hue_norm)
        assert m.limits == (hue_norm.vmin, hue_norm.vmax)
        assert m.norm is hue_norm

        # Test default colormap values
        hmin, hmax = p.plot_data["hue"].min(), p.plot_data["hue"].max()
        m = HueMapping(p)
        assert m.lookup_table[hmin] == pytest.approx(m.cmap(0.0))
        assert m.lookup_table[hmax] == pytest.approx(m.cmap(1.0))

        # Test specified colormap values
        hue_norm = hmin - 1, hmax - 1
        m = HueMapping(p, norm=hue_norm)
        norm_min = (hmin - hue_norm[0]) / (hue_norm[1] - hue_norm[0])
        assert m.lookup_table[hmin] == pytest.approx(m.cmap(norm_min))
        assert m.lookup_table[hmax] == pytest.approx(m.cmap(1.0))

        # Test list of colors
        hue_levels = list(np.sort(long_df["s"].unique()))
        palette = color_palette("Blues", len(hue_levels))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == dict(zip(hue_levels, palette))

        palette = color_palette("Blues", len(hue_levels) + 1)
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # Test dictionary of colors
        palette = dict(zip(hue_levels, color_palette("Reds")))
        m = HueMapping(p, palette=palette)
        assert m.lookup_table == palette

        palette.pop(hue_levels[0])
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # Test invalid palette
        with pytest.raises(ValueError):
            HueMapping(p, palette="not a valid palette")

        # Test bad norm argument
        with pytest.raises(ValueError):
            HueMapping(p, norm="not a norm")


class TestVectorPlotter:

    def test_flat_variables(self, flat_data):

        p = _VectorPlotter()
        p.establish_variables(data=flat_data)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y"]
        assert len(p.plot_data) == len(flat_data)

        try:
            expected_x = flat_data.index
            expected_x_name = flat_data.index.name
        except AttributeError:
            expected_x = np.arange(len(flat_data))
            expected_x_name = None

        x = p.plot_data["x"]
        assert_array_equal(x, expected_x)

        expected_y = flat_data
        expected_y_name = getattr(flat_data, "name", None)

        y = p.plot_data["y"]
        assert_array_equal(y, expected_y)

        assert p.variables["x"] == expected_x_name
        assert p.variables["y"] == expected_y_name


class TestCoreFunc:

    def test_unique_dashes(self):

        n = 24
        dashes = unique_dashes(n)

        assert len(dashes) == n
        assert len(set(dashes)) == n
        assert dashes[0] == ""
        for spec in dashes[1:]:
            assert isinstance(spec, tuple)
            assert not len(spec) % 2

    def test_unique_markers(self):

        n = 24
        markers = unique_markers(n)

        assert len(markers) == n
        assert len(set(markers)) == n
        for m in markers:
            assert mpl.markers.MarkerStyle(m).is_filled()

    def test_variable_type(self):

        s = pd.Series([1., 2., 3.])
        assert variable_type(s) == "numeric"
        assert variable_type(s.astype(int)) == "numeric"
        assert variable_type(s.astype(object)) == "numeric"
        # assert variable_type(s.to_numpy()) == "numeric"
        assert variable_type(s.values) == "numeric"
        # assert variable_type(s.to_list()) == "numeric"
        assert variable_type(s.tolist()) == "numeric"

        s = pd.Series([1, 2, 3, np.nan], dtype=object)
        assert variable_type(s) == "numeric"

        s = pd.Series([np.nan, np.nan])
        # s = pd.Series([pd.NA, pd.NA])
        assert variable_type(s) == "numeric"

        s = pd.Series(["1", "2", "3"])
        assert variable_type(s) == "categorical"
        # assert variable_type(s.to_numpy()) == "categorical"
        assert variable_type(s.values) == "categorical"
        # assert variable_type(s.to_list()) == "categorical"
        assert variable_type(s.tolist()) == "categorical"

        s = pd.Series([True, False, False])
        assert variable_type(s) == "numeric"
        assert variable_type(s, boolean_type="categorical") == "categorical"

        s = pd.Series([pd.Timestamp(1), pd.Timestamp(2)])
        assert variable_type(s) == "datetime"
        assert variable_type(s.astype(object)) == "datetime"
        # assert variable_type(s.to_numpy()) == "datetime"
        assert variable_type(s.values) == "datetime"
        # assert variable_type(s.to_list()) == "datetime"
        assert variable_type(s.tolist()) == "datetime"

    def test_infer_orient(self):

        nums = pd.Series(np.arange(6))
        cats = pd.Series(["a", "b"] * 3)

        assert infer_orient(cats, nums) == "v"
        assert infer_orient(nums, cats) == "h"

        assert infer_orient(nums, None) == "h"
        with pytest.warns(UserWarning, match="Vertical .+ `x`"):
            assert infer_orient(nums, None, "v") == "h"

        assert infer_orient(None, nums) == "v"
        with pytest.warns(UserWarning, match="Horizontal .+ `y`"):
            assert infer_orient(None, nums, "h") == "v"

        infer_orient(cats, None, require_numeric=False) == "h"
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, None)

        infer_orient(cats, None, require_numeric=False) == "v"
        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(None, cats)

        assert infer_orient(nums, nums, "vert") == "v"
        assert infer_orient(nums, nums, "hori") == "h"

        assert infer_orient(cats, cats, "h", require_numeric=False) == "h"
        assert infer_orient(cats, cats, "v", require_numeric=False) == "v"
        assert infer_orient(cats, cats, require_numeric=False) == "v"

        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(cats, cats, "v")
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, cats, "h")
        with pytest.raises(TypeError, match="Neither"):
            infer_orient(cats, cats)

    def test_categorical_order(self):

        x = ["a", "c", "c", "b", "a", "d"]
        y = [3, 2, 5, 1, 4]
        order = ["a", "b", "c", "d"]

        out = categorical_order(x)
        assert out == ["a", "c", "b", "d"]

        out = categorical_order(x, order)
        assert out == order

        out = categorical_order(x, ["b", "a"])
        assert out == ["b", "a"]

        out = categorical_order(np.array(x))
        assert out == ["a", "c", "b", "d"]

        out = categorical_order(pd.Series(x))
        assert out == ["a", "c", "b", "d"]

        out = categorical_order(y)
        assert out == [1, 2, 3, 4, 5]

        out = categorical_order(np.array(y))
        assert out == [1, 2, 3, 4, 5]

        out = categorical_order(pd.Series(y))
        assert out == [1, 2, 3, 4, 5]

        x = pd.Categorical(x, order)
        out = categorical_order(x)
        assert out == list(x.categories)

        x = pd.Series(x)
        out = categorical_order(x)
        assert out == list(x.cat.categories)

        out = categorical_order(x, ["b", "a"])
        assert out == ["b", "a"]

        x = ["a", np.nan, "c", "c", "b", "a", "d"]
        out = categorical_order(x)
        assert out == ["a", "c", "b", "d"]

    @pytest.mark.parametrize(
        "cycler,result",
        [
            (cycler(color=["y"]), ["y"]),
            (cycler(color=["k"]), ["k"]),
            (cycler(color=["k", "y"]), ["k", "y"]),
            (cycler(color=["y", "k"]), ["y", "k"]),
            (cycler(color=["b", "r"]), ["b", "r"]),
            (cycler(color=["r", "b"]), ["r", "b"]),
            (cycler(lw=[1, 2]), [".15"]),  # no color in cycle
        ],
    )
    def test_get_color_cycle(self, cycler, result):
        with mpl.rc_context(rc={"axes.prop_cycle": cycler}):
            assert get_color_cycle() == result

    def test_remove_na(self):

        a_array = np.array([1, 2, np.nan, 3])
        a_array_rm = remove_na(a_array)
        assert_array_equal(a_array_rm, np.array([1, 2, 3]))

        a_series = pd.Series([1, 2, np.nan, 3])
        a_series_rm = remove_na(a_series)
        assert_series_equal(a_series_rm, pd.Series([1., 2, 3], [0, 1, 3]))
