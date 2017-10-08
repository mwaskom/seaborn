from __future__ import division
import numpy as np
import pandas as pd
import matplotlib as mpl
import pytest
from .. import basic
from ..palettes import color_palette
from ..utils import categorical_order


class TestBasicPlotter(object):

    @pytest.fixture
    def wide_df(self):

        columns = list("abc")
        index = np.arange(10, 50, 2)
        values = np.random.randn(len(index), len(columns))
        return pd.DataFrame(values, index=index, columns=columns)

    @pytest.fixture
    def wide_array(self):

        return np.random.randn(20, 3)

    @pytest.fixture
    def flat_array(self):

        return np.random.randn(20)

    @pytest.fixture
    def wide_list(self):

        return [np.random.randn(20), np.random.randn(10)]

    @pytest.fixture
    def long_df(self):

        n = 100
        rs = np.random.RandomState()
        return pd.DataFrame(dict(
            x=rs.randint(0, 20, n),
            y=rs.randn(n),
            a=np.take(list("abc"), rs.randint(0, 3, n)),
            b=np.take(list("mnop"), rs.randint(0, 4, n)),
            s=np.take([2, 4, 8], rs.randint(0, 3, n)),
        ))

    @pytest.fixture
    def null_column(self):

        return pd.Series(index=pd.RangeIndex(0, 20))

    def test_wide_df_variables(self, wide_df):

        p = basic._BasicPlotter()
        p.establish_variables(data=wide_df)
        assert p.input_format == "wide"
        assert len(p.plot_data) == np.product(wide_df.shape)

        x = p.plot_data["x"]
        expected_x = np.tile(wide_df.index, wide_df.shape[1])
        assert np.array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = wide_df.values.ravel(order="f")
        assert np.array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(wide_df.columns, wide_df.shape[0])
        assert np.array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert np.array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

    def test_wide_df_variables_check(self, wide_df):

        p = basic._BasicPlotter()
        wide_df = wide_df.copy()
        wide_df.loc[:, "not_numeric"] = "a"
        with pytest.raises(ValueError):
            p.establish_variables(data=wide_df)

    def test_wide_array_variables(self, wide_array):

        p = basic._BasicPlotter()
        p.establish_variables(data=wide_array)
        assert p.input_format == "wide"
        assert len(p.plot_data) == np.product(wide_array.shape)

        nrow, ncol = wide_array.shape

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(nrow), ncol)
        assert np.array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = wide_array.ravel(order="f")
        assert np.array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(np.arange(ncol), nrow)
        assert np.array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert np.array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

    def test_flat_array_variables(self, flat_array):

        p = basic._BasicPlotter()
        p.establish_variables(data=flat_array)
        assert p.input_format == "wide"
        assert len(p.plot_data) == np.product(flat_array.shape)

        x = p.plot_data["x"]
        expected_x = np.arange(flat_array.shape[0])
        assert np.array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = flat_array
        assert np.array_equal(y, expected_y)

        assert p.plot_data["hue"].isnull().all()
        assert p.plot_data["style"].isnull().all()
        assert p.plot_data["size"].isnull().all()

    def test_wide_list_variables(self, wide_list):

        p = basic._BasicPlotter()
        p.establish_variables(data=wide_list)
        assert p.input_format == "wide"
        assert len(p.plot_data) == sum(len(l) for l in wide_list)

        x = p.plot_data["x"]
        expected_x = np.concatenate([np.arange(len(l)) for l in wide_list])
        assert np.array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = np.concatenate(wide_list)
        assert np.array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.concatenate([
            np.ones_like(l) * i for i, l in enumerate(wide_list)
        ])
        assert np.array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert np.array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

    def test_long_df(self, long_df):

        p = basic._BasicPlotter()
        p.establish_variables(x="x", y="y", data=long_df)
        assert p.input_format == "long"

        assert np.array_equal(p.plot_data["x"], long_df["x"])
        assert np.array_equal(p.plot_data["y"], long_df["y"])
        for col in ["hue", "style", "size"]:
            assert p.plot_data[col].isnull().all()

        p.establish_variables(x=long_df.x, y="y", data=long_df)
        assert np.array_equal(p.plot_data["x"], long_df["x"])
        assert np.array_equal(p.plot_data["y"], long_df["y"])

        p.establish_variables(x="x", y=long_df.y, data=long_df)
        assert np.array_equal(p.plot_data["x"], long_df["x"])
        assert np.array_equal(p.plot_data["y"], long_df["y"])

        p.establish_variables(x="x", y="y", hue="a", data=long_df)
        assert np.array_equal(p.plot_data["hue"], long_df["a"])
        for col in ["style", "size"]:
            assert p.plot_data[col].isnull().all()

        p.establish_variables(x="x", y="y", hue="a", style="a", data=long_df)
        assert np.array_equal(p.plot_data["hue"], long_df["a"])
        assert np.array_equal(p.plot_data["style"], long_df["a"])
        assert p.plot_data["size"].isnull().all()

        p.establish_variables(x="x", y="y", hue="a", style="b", data=long_df)
        assert np.array_equal(p.plot_data["hue"], long_df["a"])
        assert np.array_equal(p.plot_data["style"], long_df["b"])
        assert p.plot_data["size"].isnull().all()

        p.establish_variables(x="x", y="y", size="y", data=long_df)
        assert np.array_equal(p.plot_data["size"], long_df["y"])

    def test_bad_input(self, long_df):

        p = basic._BasicPlotter()

        with pytest.raises(ValueError):
            p.establish_variables(x=long_df.x)

        with pytest.raises(ValueError):
            p.establish_variables(y=long_df.y)

        with pytest.raises(ValueError):
            p.establish_variables(x="not_in_df", data=long_df)

        with pytest.raises(ValueError):
            p.establish_variables(x="x", y="not_in_df", data=long_df)

        with pytest.raises(ValueError):
            p.establish_variables(x="x", y="not_in_df", data=long_df)

        with pytest.raises(ValueError):
            p.establish_variables(data=np.array([[], []]))


class TestLinePlotter(TestBasicPlotter):

    def test_parse_hue_null(self, wide_df, null_column):

        p = basic._LinePlotter(data=wide_df)
        p.parse_hue(null_column, "Blues", None, None)
        assert p.hue_levels == [None]
        assert p.palette == {}
        assert p.hue_type is None
        assert p.cmap is None

    def test_parse_hue_categorical(self, wide_df, long_df):

        p = basic._LinePlotter(data=wide_df)
        assert p.hue_levels == wide_df.columns.tolist()
        assert p.hue_type is "categorical"
        assert p.cmap is None

        # Test named palette
        palette = "Blues"
        expected_colors = color_palette(palette, wide_df.shape[1])
        expected_palette = dict(zip(wide_df.columns, expected_colors))
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.palette == expected_palette

        # Test list palette
        palette = color_palette("Reds", wide_df.shape[1] + 2)
        p.parse_hue(p.plot_data.hue, palette, None, None)
        expected_palette = dict(zip(wide_df.columns, palette))
        assert p.palette == expected_palette

        # Test dict palette
        colors = color_palette("Set1", 8)
        palette = dict(zip(wide_df.columns, colors))
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.palette == palette

        # Test dict with missing keys
        palette = dict(zip(wide_df.columns[:-1], colors))
        with pytest.raises(ValueError):
            p.parse_hue(p.plot_data.hue, palette, None, None)

        # Test hue order
        hue_order = ["a", "c", "d"]
        p.parse_hue(p.plot_data.hue, None, hue_order, None)
        assert p.hue_levels == hue_order

        # Test long data
        p = basic._LinePlotter(x="x", y="y", hue="a", data=long_df)
        assert p.hue_levels == categorical_order(long_df.a)
        assert p.hue_type is "categorical"
        assert p.cmap is None

        # Test default palette
        p.parse_hue(p.plot_data.hue, None, None, None)
        hue_levels = categorical_order(long_df.a)
        expected_colors = color_palette(n_colors=len(hue_levels))
        expected_palette = dict(zip(hue_levels, expected_colors))
        assert p.palette == expected_palette

        # Test default palette with many levels
        levels = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        p.parse_hue(levels, None, None, None)
        expected_colors = color_palette("husl", n_colors=len(levels))
        expected_palette = dict(zip(levels, expected_colors))
        assert p.palette == expected_palette

    def test_parse_hue_numeric(self, long_df):

        p = basic._LinePlotter(x="x", y="y", hue="s", data=long_df)
        hue_levels = list(np.sort(long_df.s.unique()))
        assert p.hue_levels == hue_levels
        assert p.hue_type is "numeric"
        assert p.cmap is mpl.cm.get_cmap(mpl.rcParams["image.cmap"])

        # Test named colormap
        palette = "Purples"
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.cmap is mpl.cm.get_cmap(palette)

        # Test colormap object
        palette = mpl.cm.get_cmap("Greens")
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.cmap is palette

        # Test default hue limits
        p.parse_hue(p.plot_data.hue, None, None, None)
        assert p.hue_limits == (p.plot_data.hue.min(), p.plot_data.hue.max())

        # Test specified hue limits
        hue_limits = 1, 4
        p.parse_hue(p.plot_data.hue, None, None, hue_limits)
        assert p.hue_limits == hue_limits

        # Test default colormap values
        min, max = p.plot_data.hue.min(), p.plot_data.hue.max()
        p.parse_hue(p.plot_data.hue, None, None, None)
        assert p.palette[min] == pytest.approx(p.cmap(0.0))
        assert p.palette[max] == pytest.approx(p.cmap(1.0))

        # Test specified colormap values
        hue_limits = min - 1, max - 1
        p.parse_hue(p.plot_data.hue, None, None, hue_limits)
        norm_min = (min - hue_limits[0]) / (hue_limits[1] - hue_limits[0])
        assert p.palette[min] == pytest.approx(p.cmap(norm_min))
        assert p.palette[max] == pytest.approx(p.cmap(1.0))

        # Test list of colors
        hue_levels = list(np.sort(long_df.s.unique()))
        palette = color_palette("Blues", len(hue_levels))
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.palette == dict(zip(hue_levels, palette))

        palette = color_palette("Blues", len(hue_levels) + 1)
        with pytest.raises(ValueError):
            p.parse_hue(p.plot_data.hue, palette, None, None)

        # Test dictionary of colors
        palette = dict(zip(hue_levels, color_palette("Reds")))
        p.parse_hue(p.plot_data.hue, palette, None, None)
        assert p.palette == palette

        palette.pop(hue_levels[0])
        with pytest.raises(ValueError):
            p.parse_hue(p.plot_data.hue, palette, None, None)

        # Test invalid palette
        palette = "not_a_valid_palette"
        with pytest.raises(ValueError):
            p.parse_hue(p.plot_data.hue, palette, None, None)
