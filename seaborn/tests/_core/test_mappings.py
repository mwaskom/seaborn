import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize, same_color

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._compat import MarkerStyle
from seaborn._core.rules import categorical_order
from seaborn._core.scales import (
    CategoricalScale,
    DateTimeScale,
    NumericScale,
    get_default_scale,
)
from seaborn._core.mappings import (
    BooleanSemantic,
    ColorSemantic,
    MarkerSemantic,
    LineStyleSemantic,
    WidthSemantic,
    EdgeWidthSemantic,
    LineWidthSemantic,
)
from seaborn.palettes import color_palette


class TestColor:

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

    @pytest.fixture
    def num_scale(self, num_vector):
        norm = Normalize()
        norm.autoscale(num_vector)
        scale = get_default_scale(num_vector)
        return scale

    @pytest.fixture
    def cat_vector(self, long_df):
        return long_df["a"]

    @pytest.fixture
    def cat_order(self, cat_vector):
        return categorical_order(cat_vector)

    @pytest.fixture
    def dt_num_vector(self, long_df):
        return long_df["t"]

    @pytest.fixture
    def dt_cat_vector(self, long_df):
        return long_df["d"]

    def test_categorical_default_palette(self, cat_vector, cat_order):

        expected = dict(zip(cat_order, color_palette()))
        scale = get_default_scale(cat_vector)
        m = ColorSemantic().setup(cat_vector, scale)

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_default_palette_large(self):

        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        scale = get_default_scale(vector)
        n_colors = len(vector)
        expected = dict(zip(vector, color_palette("husl", n_colors)))
        m = ColorSemantic().setup(vector, scale)

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_named_palette(self, cat_vector, cat_order):

        palette = "Blues"
        scale = get_default_scale(cat_vector)
        m = ColorSemantic(palette=palette).setup(cat_vector, scale)

        colors = color_palette(palette, len(cat_order))
        expected = dict(zip(cat_order, colors))
        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_list_palette(self, cat_vector, cat_order):

        palette = color_palette("Reds", len(cat_order))
        scale = get_default_scale(cat_vector)
        m = ColorSemantic(palette=palette).setup(cat_vector, scale)

        expected = dict(zip(cat_order, palette))
        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_list_palette(self, num_vector, num_order):

        palette = color_palette("Reds", len(num_order))
        scale = get_default_scale(num_vector)
        m = ColorSemantic(palette=palette).setup(num_vector, scale)

        expected = dict(zip(num_order, palette))
        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_dict_palette(self, cat_vector, cat_order):

        palette = dict(zip(cat_order, color_palette("Greens")))
        scale = get_default_scale(cat_vector)
        m = ColorSemantic(palette=palette).setup(cat_vector, scale)
        assert m.mapping == palette

        for level, color in palette.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_dict_palette(self, num_vector, num_order):

        palette = dict(zip(num_order, color_palette("Greens")))
        scale = get_default_scale(num_vector)
        m = ColorSemantic(palette=palette).setup(num_vector, scale)
        assert m.mapping == palette

        for level, color in palette.items():
            assert same_color(m(level), color)

    def test_categorical_dict_with_missing_keys(self, cat_vector, cat_order):

        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        scale = get_default_scale(cat_vector)
        with pytest.raises(ValueError):
            ColorSemantic(palette=palette).setup(cat_vector, scale)

    def test_categorical_list_too_short(self, cat_vector, cat_order):

        n = len(cat_order) - 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has fewer values \({n}\) than needed \({n + 1}\)"
        m = ColorSemantic(palette=palette, variable="edgecolor")
        scale = get_default_scale(cat_vector)
        with pytest.warns(UserWarning, match=msg):
            m.setup(cat_vector, scale)

    @pytest.mark.xfail(reason="Need decision on new behavior")
    def test_categorical_list_too_long(self, cat_vector, cat_order):

        n = len(cat_order) + 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has more values \({n}\) than needed \({n - 1}\)"
        m = ColorSemantic(palette=palette, variable="edgecolor")
        with pytest.warns(UserWarning, match=msg):
            m.setup(cat_vector)

    def test_categorical_with_ordered_scale(self, cat_vector):

        cat_order = list(cat_vector.unique()[::-1])
        scale = CategoricalScale(LinearScale("color"), cat_order, format)

        palette = "deep"
        colors = color_palette(palette, len(cat_order))

        m = ColorSemantic(palette=palette).setup(cat_vector, scale)

        expected = dict(zip(cat_order, colors))

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_scale(self, num_vector, num_order):

        scale = CategoricalScale(LinearScale("color"), num_order, format)
        scale.type_declared = True

        palette = "deep"
        colors = color_palette(palette, len(num_order))

        m = ColorSemantic(palette=palette).setup(num_vector, scale)

        expected = dict(zip(num_order, colors))

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_ordered_scale(self, num_vector):

        order = num_vector.unique()
        if order[0] < order[1]:
            order[[0, 1]] = order[[1, 0]]
        order = list(order)

        scale = CategoricalScale(LinearScale("color"), order, format)

        palette = "deep"
        colors = color_palette(palette, len(order))

        m = ColorSemantic(palette=palette).setup(num_vector, scale)

        expected = dict(zip(order, colors))

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_with_ordered_categories(self, cat_vector, cat_order):

        new_order = list(reversed(cat_order))
        new_vector = cat_vector.astype("category").cat.set_categories(new_order)
        scale = get_default_scale(new_vector)

        expected = dict(zip(new_order, color_palette()))

        m = ColorSemantic().setup(new_vector, scale)

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_categories(self, num_vector):

        new_vector = num_vector.astype("category")
        new_order = categorical_order(new_vector)
        scale = get_default_scale(new_vector)

        expected = dict(zip(new_order, color_palette()))

        m = ColorSemantic().setup(new_vector, scale)

        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_implied_by_palette(self, num_vector, num_order):

        palette = "bright"
        expected = dict(zip(num_order, color_palette(palette)))
        scale = get_default_scale(num_vector)
        m = ColorSemantic(palette=palette).setup(num_vector, scale)
        for level, color in expected.items():
            assert same_color(m(level), color)

    def test_categorical_from_binary_data(self):

        vector = pd.Series([1, 0, 0, 0, 1, 1, 1])
        scale = get_default_scale(vector)
        expected_palette = dict(zip([0, 1], color_palette()))
        m = ColorSemantic().setup(vector, scale)

        for level, color in expected_palette.items():
            assert same_color(m(level), color)

        first_color, *_ = color_palette()

        for val in [0, 1]:
            x = pd.Series([val] * 4)
            scale = get_default_scale(x)
            m = ColorSemantic().setup(x, scale)
            assert same_color(m(val), first_color)

    def test_categorical_multi_lookup(self):

        x = pd.Series(["a", "b", "c"])
        colors = color_palette(n_colors=len(x))
        scale = get_default_scale(x)
        m = ColorSemantic().setup(x, scale)
        assert_series_equal(m(x), pd.Series(colors))

    def test_categorical_multi_lookup_categorical(self):

        x = pd.Series(["a", "b", "c"]).astype("category")
        colors = color_palette(n_colors=len(x))
        scale = get_default_scale(x)
        m = ColorSemantic().setup(x, scale)
        assert_series_equal(m(x), pd.Series(colors))

    def test_numeric_default_palette(self, num_vector, num_order, num_scale):

        m = ColorSemantic().setup(num_vector, num_scale)
        expected_cmap = color_palette("ch:", as_cmap=True)
        for level in num_order:
            assert same_color(m(level), expected_cmap(num_scale.norm(level)))

    def test_numeric_named_palette(self, num_vector, num_order, num_scale):

        palette = "viridis"
        m = ColorSemantic(palette=palette).setup(num_vector, num_scale)
        expected_cmap = color_palette(palette, as_cmap=True)
        for level in num_order:
            assert same_color(m(level), expected_cmap(num_scale.norm(level)))

    def test_numeric_colormap_palette(self, num_vector, num_order, num_scale):

        cmap = color_palette("rocket", as_cmap=True)
        m = ColorSemantic(palette=cmap).setup(num_vector, num_scale)
        for level in num_order:
            assert same_color(m(level), cmap(num_scale.norm(level)))

    def test_numeric_norm_limits(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        cmap = color_palette("rocket", as_cmap=True)
        scale = NumericScale(LinearScale("color"), norm=lims)
        norm = Normalize(*lims)
        m = ColorSemantic(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert same_color(m(level), cmap(norm(level)))

    def test_numeric_norm_object(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        norm = Normalize(*lims)
        cmap = color_palette("rocket", as_cmap=True)
        scale = NumericScale(LinearScale("color"), norm=lims)
        m = ColorSemantic(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert same_color(m(level), cmap(norm(level)))

    def test_numeric_dict_palette_with_norm(self, num_vector, num_order, num_scale):

        palette = dict(zip(num_order, color_palette()))
        m = ColorSemantic(palette=palette).setup(num_vector, num_scale)
        for level, color in palette.items():
            assert same_color(m(level), color)

    def test_numeric_multi_lookup(self, num_vector, num_scale):

        cmap = color_palette("mako", as_cmap=True)
        m = ColorSemantic(palette=cmap).setup(num_vector, num_scale)
        expected_colors = cmap(num_scale.norm(num_vector.to_numpy()))[:, :3]
        assert_array_equal(m(num_vector.to_numpy()), expected_colors)

    def test_datetime_default_palette(self, dt_num_vector):

        scale = get_default_scale(dt_num_vector)
        m = ColorSemantic().setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - dt_num_vector.min()
        normed = tmp / tmp.max()

        expected_cmap = color_palette("ch:", as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)

    def test_datetime_specified_palette(self, dt_num_vector):

        palette = "mako"
        scale = get_default_scale(dt_num_vector)
        m = ColorSemantic(palette=palette).setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - dt_num_vector.min()
        normed = tmp / tmp.max()

        expected_cmap = color_palette(palette, as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)

    @pytest.mark.xfail(reason="No support for norms in datetime scale yet")
    def test_datetime_norm_limits(self, dt_num_vector):

        norm = (
            dt_num_vector.min() - pd.Timedelta(2, "m"),
            dt_num_vector.max() - pd.Timedelta(1, "m"),
        )
        palette = "mako"

        scale = DateTimeScale(LinearScale("color"), norm=norm)
        m = ColorSemantic(palette=palette).setup(dt_num_vector, scale)
        mapped = m(dt_num_vector)

        tmp = dt_num_vector - norm[0]
        normed = tmp / norm[1]

        expected_cmap = color_palette(palette, as_cmap=True)
        expected = expected_cmap(normed)

        assert len(mapped) == len(expected)
        for have, want in zip(mapped, expected):
            assert same_color(have, want)

    def test_bad_palette(self, num_vector, num_scale):

        with pytest.raises(ValueError):
            ColorSemantic(palette="not_a_palette").setup(num_vector, num_scale)


class DiscreteBase:

    def test_none_provided(self):

        keys = pd.Series(["a", "b", "c"])
        scale = get_default_scale(keys)
        m = self.semantic().setup(keys, scale)

        defaults = self.semantic()._default_values(len(keys))

        for key, want in zip(keys, defaults):
            self.assert_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(defaults)
        for have, want in zip(mapped, defaults):
            self.assert_equal(have, want)

    def _test_provided_list(self, values):

        keys = pd.Series(["a", "b", "c", "d"])
        scale = get_default_scale(keys)
        m = self.semantic(values).setup(keys, scale)

        for key, want in zip(keys, values):
            self.assert_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(values)
        for have, want in zip(mapped, values):
            self.assert_equal(have, want)

    def _test_provided_dict(self, values):

        keys = pd.Series(["a", "b", "c", "d"])
        scale = get_default_scale(keys)
        mapping = dict(zip(keys, values))
        m = self.semantic(mapping).setup(keys, scale)

        for key, want in mapping.items():
            self.assert_equal(m(key), want)

        mapped = m(keys)
        assert len(mapped) == len(values)
        for have, want in zip(mapped, values):
            self.assert_equal(have, want)


class TestLineStyle(DiscreteBase):

    semantic = LineStyleSemantic

    def assert_equal(self, a, b):

        a = self.semantic()._get_dash_pattern(a)
        b = self.semantic()._get_dash_pattern(b)
        assert a == b

    def test_unique_dashes(self):

        n = 24
        dashes = self.semantic()._default_values(n)

        assert len(dashes) == n
        assert len(set(dashes)) == n

        assert dashes[0] == (0, None)
        for spec in dashes[1:]:
            assert isinstance(spec, tuple)
            assert spec[0] == 0
            assert not len(spec[1]) % 2

    def test_provided_list(self):

        values = ["-", (1, 4), "dashed", (.5, (5, 2))]
        self._test_provided_list(values)

    def test_provided_dict(self):

        values = ["-", (1, 4), "dashed", (.5, (5, 2))]
        self._test_provided_dict(values)

    def test_provided_dict_with_missing(self):

        m = self.semantic({})
        keys = pd.Series(["a", 1])
        scale = get_default_scale(keys)
        err = r"Missing linestyle for following value\(s\): 1, 'a'"
        with pytest.raises(ValueError, match=err):
            m.setup(keys, scale)


class TestMarker(DiscreteBase):

    semantic = MarkerSemantic

    def assert_equal(self, a, b):

        a = MarkerStyle(a)
        b = MarkerStyle(b)
        assert a.get_path() == b.get_path()
        assert a.get_joinstyle() == b.get_joinstyle()
        assert a.get_transform().to_values() == b.get_transform().to_values()
        assert a.get_fillstyle() == b.get_fillstyle()

    def test_unique_markers(self):

        n = 24
        markers = MarkerSemantic()._default_values(n)

        assert len(markers) == n
        assert len(set(
            (m.get_path(), m.get_joinstyle(), m.get_transform().to_values())
            for m in markers
        )) == n

        for m in markers:
            assert MarkerStyle(m).is_filled()

    def test_provided_list(self):

        markers = ["o", (5, 2, 0), MarkerStyle("o", fillstyle="none"), "x"]
        self._test_provided_list(markers)

    def test_provided_dict(self):

        values = ["o", (5, 2, 0), MarkerStyle("o", fillstyle="none"), "x"]
        self._test_provided_dict(values)

    def test_provided_dict_with_missing(self):

        m = MarkerSemantic({})
        keys = pd.Series(["a", 1])
        scale = get_default_scale(keys)
        err = r"Missing marker for following value\(s\): 1, 'a'"
        with pytest.raises(ValueError, match=err):
            m.setup(keys, scale)


class TestBoolean:

    def test_default(self):

        x = pd.Series(["a", "b"])
        scale = get_default_scale(x)
        m = BooleanSemantic().setup(x, scale)
        assert m("a") is True
        assert m("b") is False

    def test_default_warns(self):

        x = pd.Series(["a", "b", "c"])
        s = BooleanSemantic(variable="fill")
        msg = "There are only two possible fill values, so they will cycle"
        scale = get_default_scale(x)
        with pytest.warns(UserWarning, match=msg):
            m = s.setup(x, scale)
        assert m("a") is True
        assert m("b") is False
        assert m("c") is True

    def test_provided_list(self):

        x = pd.Series(["a", "b", "c"])
        values = [True, True, False]
        scale = get_default_scale(x)
        m = BooleanSemantic(values).setup(x, scale)
        for k, v in zip(x, values):
            assert m(k) is v


class ContinuousBase:

    @staticmethod
    def norm(x, vmin, vmax):
        normed = x - vmin
        normed /= vmax - vmin
        return normed

    @staticmethod
    def transform(x, lo, hi):
        return lo + x * (hi - lo)

    def test_default_numeric(self):

        x = pd.Series([-1, .4, 2, 1.2])
        scale = get_default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        normed = self.norm(x, x.min(), x.max())
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_default_categorical(self):

        x = pd.Series(["a", "c", "b", "c"])
        scale = get_default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        normed = np.array([1, .5, 0, .5])
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_range_numeric(self):

        values = (1, 5)
        x = pd.Series([-1, .4, 2, 1.2])
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        normed = self.norm(x, x.min(), x.max())
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)

    def test_range_categorical(self):

        values = (1, 5)
        x = pd.Series(["a", "c", "b", "c"])
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        normed = np.array([1, .5, 0, .5])
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)

    def test_list_numeric(self):

        values = [.3, .8, .5]
        x = pd.Series([2, 500, 10, 500])
        expected = [.3, .5, .8, .5]
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_list_categorical(self):

        values = [.2, .6, .4]
        x = pd.Series(["a", "c", "b", "c"])
        expected = [.2, .6, .4, .6]
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_list_implies_categorical(self):

        x = pd.Series([2, 500, 10, 500])
        values = [.2, .6, .4]
        expected = [.2, .4, .6, .4]
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, expected)

    def test_dict_numeric(self):

        x = pd.Series([2, 500, 10, 500])
        values = {2: .3, 500: .5, 10: .8}
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, x.map(values))

    def test_dict_categorical(self):

        x = pd.Series(["a", "c", "b", "c"])
        values = {"a": .3, "b": .5, "c": .8}
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        assert_array_equal(y, x.map(values))

    def test_norm_numeric(self):

        x = pd.Series([2, 500, 10])
        norm = mpl.colors.LogNorm(1, 100)
        scale = NumericScale(LinearScale("x"), norm=norm)
        y = self.semantic().setup(x, scale)(x)
        x = np.asarray(x)  # matplotlib<3.4.3 compatability
        expected = self.transform(norm(x), *self.semantic().default_range)
        assert_array_equal(y, expected)

    @pytest.mark.xfail(reason="Needs decision about behavior")
    def test_norm_categorical(self):

        # TODO is it right to raise here or should that happen upstream?
        # Or is there some reasonable way to actually use the norm?
        x = pd.Series(["a", "c", "b", "c"])
        norm = mpl.colors.LogNorm(1, 100)
        scale = NumericScale(LinearScale("x"), norm=norm)
        with pytest.raises(ValueError):
            self.semantic().setup(x, scale)

    def test_default_datetime(self):

        x = pd.Series(np.array([10000, 10100, 10101], dtype="datetime64[D]"))
        scale = get_default_scale(x)
        y = self.semantic().setup(x, scale)(x)
        tmp = x - x.min()
        normed = tmp / tmp.max()
        expected = self.transform(normed, *self.semantic().default_range)
        assert_array_equal(y, expected)

    def test_range_datetime(self):

        values = .2, .9
        x = pd.Series(np.array([10000, 10100, 10101], dtype="datetime64[D]"))
        scale = get_default_scale(x)
        y = self.semantic(values).setup(x, scale)(x)
        tmp = x - x.min()
        normed = tmp / tmp.max()
        expected = self.transform(normed, *values)
        assert_array_equal(y, expected)


class TestWidth(ContinuousBase):

    semantic = WidthSemantic


class TestLineWidth(ContinuousBase):

    semantic = LineWidthSemantic


class TestEdgeWidth(ContinuousBase):

    semantic = EdgeWidthSemantic
