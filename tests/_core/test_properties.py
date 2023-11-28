
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import same_color, to_rgb, to_rgba

import pytest
from numpy.testing import assert_array_equal

from seaborn.utils import _version_predates
from seaborn._core.rules import categorical_order
from seaborn._core.scales import Nominal, Continuous, Boolean
from seaborn._core.properties import (
    Alpha,
    Color,
    Coordinate,
    EdgeWidth,
    Fill,
    LineStyle,
    LineWidth,
    Marker,
    PointSize,
)
from seaborn._compat import MarkerStyle, get_colormap
from seaborn.palettes import color_palette


class DataFixtures:

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

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

    @pytest.fixture
    def bool_vector(self, long_df):
        return long_df["x"] > 10

    @pytest.fixture
    def vectors(self, num_vector, cat_vector, bool_vector):
        return {"num": num_vector, "cat": cat_vector, "bool": bool_vector}


class TestCoordinate(DataFixtures):

    def test_bad_scale_arg_str(self, num_vector):

        err = "Unknown magic arg for x scale: 'xxx'."
        with pytest.raises(ValueError, match=err):
            Coordinate("x").infer_scale("xxx", num_vector)

    def test_bad_scale_arg_type(self, cat_vector):

        err = "Magic arg for x scale must be str, not list."
        with pytest.raises(TypeError, match=err):
            Coordinate("x").infer_scale([1, 2, 3], cat_vector)


class TestColor(DataFixtures):

    def assert_same_rgb(self, a, b):
        assert_array_equal(a[:, :3], b[:, :3])

    def test_nominal_default_palette(self, cat_vector, cat_order):

        m = Color().get_mapping(Nominal(), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = color_palette(None, n)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_default_palette_large(self):

        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        m = Color().get_mapping(Nominal(), vector)
        actual = m(np.arange(26))
        expected = color_palette("husl", 26)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_named_palette(self, cat_vector, cat_order):

        palette = "Blues"
        m = Color().get_mapping(Nominal(palette), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = color_palette(palette, n)
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_list_palette(self, cat_vector, cat_order):

        palette = color_palette("Reds", len(cat_order))
        m = Color().get_mapping(Nominal(palette), cat_vector)
        actual = m(np.arange(len(palette)))
        expected = palette
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_dict_palette(self, cat_vector, cat_order):

        colors = color_palette("Greens")
        palette = dict(zip(cat_order, colors))
        m = Color().get_mapping(Nominal(palette), cat_vector)
        n = len(cat_order)
        actual = m(np.arange(n))
        expected = colors
        for have, want in zip(actual, expected):
            assert same_color(have, want)

    def test_nominal_dict_with_missing_keys(self, cat_vector, cat_order):

        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        with pytest.raises(ValueError, match="No entry in color dict"):
            Color("color").get_mapping(Nominal(palette), cat_vector)

    def test_nominal_list_too_short(self, cat_vector, cat_order):

        n = len(cat_order) - 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has fewer values \({n}\) than needed \({n + 1}\)"
        with pytest.warns(UserWarning, match=msg):
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    def test_nominal_list_too_long(self, cat_vector, cat_order):

        n = len(cat_order) + 1
        palette = color_palette("Oranges", n)
        msg = rf"The edgecolor list has more values \({n}\) than needed \({n - 1}\)"
        with pytest.warns(UserWarning, match=msg):
            Color("edgecolor").get_mapping(Nominal(palette), cat_vector)

    def test_continuous_default_palette(self, num_vector):

        cmap = color_palette("ch:", as_cmap=True)
        m = Color().get_mapping(Continuous(), num_vector)
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    def test_continuous_named_palette(self, num_vector):

        pal = "flare"
        cmap = color_palette(pal, as_cmap=True)
        m = Color().get_mapping(Continuous(pal), num_vector)
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    def test_continuous_tuple_palette(self, num_vector):

        vals = ("blue", "red")
        cmap = color_palette("blend:" + ",".join(vals), as_cmap=True)
        m = Color().get_mapping(Continuous(vals), num_vector)
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    def test_continuous_callable_palette(self, num_vector):

        cmap = get_colormap("viridis")
        m = Color().get_mapping(Continuous(cmap), num_vector)
        self.assert_same_rgb(m(num_vector), cmap(num_vector))

    def test_continuous_missing(self):

        x = pd.Series([1, 2, np.nan, 4])
        m = Color().get_mapping(Continuous(), x)
        assert np.isnan(m(x)[2]).all()

    def test_bad_scale_values_continuous(self, num_vector):

        with pytest.raises(TypeError, match="Scale values for color with a Continuous"):
            Color().get_mapping(Continuous(["r", "g", "b"]), num_vector)

    def test_bad_scale_values_nominal(self, cat_vector):

        with pytest.raises(TypeError, match="Scale values for color with a Nominal"):
            Color().get_mapping(Nominal(get_colormap("viridis")), cat_vector)

    def test_bad_inference_arg(self, cat_vector):

        with pytest.raises(TypeError, match="A single scale argument for color"):
            Color().infer_scale(123, cat_vector)

    @pytest.mark.parametrize(
        "data_type,scale_class",
        [("cat", Nominal), ("num", Continuous), ("bool", Boolean)]
    )
    def test_default(self, data_type, scale_class, vectors):

        scale = Color().default_scale(vectors[data_type])
        assert isinstance(scale, scale_class)

    def test_default_numeric_data_category_dtype(self, num_vector):

        scale = Color().default_scale(num_vector.astype("category"))
        assert isinstance(scale, Nominal)

    def test_default_binary_data(self):

        x = pd.Series([0, 0, 1, 0, 1], dtype=int)
        scale = Color().default_scale(x)
        assert isinstance(scale, Continuous)

    @pytest.mark.parametrize(
        "values,data_type,scale_class",
        [
            ("viridis", "cat", Nominal),  # Based on variable type
            ("viridis", "num", Continuous),  # Based on variable type
            ("viridis", "bool", Boolean),  # Based on variable type
            ("muted", "num", Nominal),  # Based on qualitative palette
            (["r", "g", "b"], "num", Nominal),  # Based on list palette
            ({2: "r", 4: "g", 8: "b"}, "num", Nominal),  # Based on dict palette
            (("r", "b"), "num", Continuous),  # Based on tuple / variable type
            (("g", "m"), "cat", Nominal),  # Based on tuple / variable type
            (("c", "y"), "bool", Boolean),  # Based on tuple / variable type
            (get_colormap("inferno"), "num", Continuous),  # Based on callable
        ]
    )
    def test_inference(self, values, data_type, scale_class, vectors):

        scale = Color().infer_scale(values, vectors[data_type])
        assert isinstance(scale, scale_class)
        assert scale.values == values

    def test_standardization(self):

        f = Color().standardize
        assert f("C3") == to_rgb("C3")
        assert f("dodgerblue") == to_rgb("dodgerblue")

        assert f((.1, .2, .3)) == (.1, .2, .3)
        assert f((.1, .2, .3, .4)) == (.1, .2, .3, .4)

        assert f("#123456") == to_rgb("#123456")
        assert f("#12345678") == to_rgba("#12345678")

        assert f("#123") == to_rgb("#123")
        assert f("#1234") == to_rgba("#1234")


class ObjectPropertyBase(DataFixtures):

    def assert_equal(self, a, b):

        assert self.unpack(a) == self.unpack(b)

    def unpack(self, x):
        return x

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_default(self, data_type, vectors):

        scale = self.prop().default_scale(vectors[data_type])
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_list(self, data_type, vectors):

        scale = self.prop().infer_scale(self.values, vectors[data_type])
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == self.values

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_dict(self, data_type, vectors):

        x = vectors[data_type]
        values = dict(zip(categorical_order(x), self.values))
        scale = self.prop().infer_scale(values, x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == values

    def test_dict_missing(self, cat_vector):

        levels = categorical_order(cat_vector)
        values = dict(zip(levels, self.values[:-1]))
        scale = Nominal(values)
        name = self.prop.__name__.lower()
        msg = f"No entry in {name} dictionary for {repr(levels[-1])}"
        with pytest.raises(ValueError, match=msg):
            self.prop().get_mapping(scale, cat_vector)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_default(self, data_type, vectors):

        x = vectors[data_type]
        mapping = self.prop().get_mapping(Nominal(), x)
        n = x.nunique()
        for i, expected in enumerate(self.prop()._default_values(n)):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_from_list(self, data_type, vectors):

        x = vectors[data_type]
        scale = Nominal(self.values)
        mapping = self.prop().get_mapping(scale, x)
        for i, expected in enumerate(self.standardized_values):
            actual, = mapping([i])
            self.assert_equal(actual, expected)

    @pytest.mark.parametrize("data_type", ["cat", "num"])
    def test_mapping_from_dict(self, data_type, vectors):

        x = vectors[data_type]
        levels = categorical_order(x)
        values = dict(zip(levels, self.values[::-1]))
        standardized_values = dict(zip(levels, self.standardized_values[::-1]))

        scale = Nominal(values)
        mapping = self.prop().get_mapping(scale, x)
        for i, level in enumerate(levels):
            actual, = mapping([i])
            expected = standardized_values[level]
            self.assert_equal(actual, expected)

    def test_mapping_with_null_value(self, cat_vector):

        mapping = self.prop().get_mapping(Nominal(self.values), cat_vector)
        actual = mapping(np.array([0, np.nan, 2]))
        v0, _, v2 = self.standardized_values
        expected = [v0, self.prop.null_value, v2]
        for a, b in zip(actual, expected):
            self.assert_equal(a, b)

    def test_unique_default_large_n(self):

        n = 24
        x = pd.Series(np.arange(n))
        mapping = self.prop().get_mapping(Nominal(), x)
        assert len({self.unpack(x_i) for x_i in mapping(x)}) == n

    def test_bad_scale_values(self, cat_vector):

        var_name = self.prop.__name__.lower()
        with pytest.raises(TypeError, match=f"Scale values for a {var_name} variable"):
            self.prop().get_mapping(Nominal(("o", "s")), cat_vector)


class TestMarker(ObjectPropertyBase):

    prop = Marker
    values = ["o", (5, 2, 0), MarkerStyle("^")]
    standardized_values = [MarkerStyle(x) for x in values]

    def unpack(self, x):
        return (
            x.get_path(),
            x.get_joinstyle(),
            x.get_transform().to_values(),
            x.get_fillstyle(),
        )


class TestLineStyle(ObjectPropertyBase):

    prop = LineStyle
    values = ["solid", "--", (1, .5)]
    standardized_values = [LineStyle._get_dash_pattern(x) for x in values]

    def test_bad_type(self):

        p = LineStyle()
        with pytest.raises(TypeError, match="^Linestyle must be .+, not list.$"):
            p.standardize([1, 2])

    def test_bad_style(self):

        p = LineStyle()
        with pytest.raises(ValueError, match="^Linestyle string must be .+, not 'o'.$"):
            p.standardize("o")

    def test_bad_dashes(self):

        p = LineStyle()
        with pytest.raises(TypeError, match="^Invalid dash pattern"):
            p.standardize((1, 2, "x"))


class TestFill(DataFixtures):

    @pytest.fixture
    def vectors(self):

        return {
            "cat": pd.Series(["a", "a", "b"]),
            "num": pd.Series([1, 1, 2]),
            "bool": pd.Series([True, True, False])
        }

    @pytest.fixture
    def cat_vector(self, vectors):
        return vectors["cat"]

    @pytest.fixture
    def num_vector(self, vectors):
        return vectors["num"]

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_default(self, data_type, vectors):

        x = vectors[data_type]
        scale = Fill().default_scale(x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_list(self, data_type, vectors):

        x = vectors[data_type]
        scale = Fill().infer_scale([True, False], x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == [True, False]

    @pytest.mark.parametrize("data_type", ["cat", "num", "bool"])
    def test_inference_dict(self, data_type, vectors):

        x = vectors[data_type]
        values = dict(zip(x.unique(), [True, False]))
        scale = Fill().infer_scale(values, x)
        assert isinstance(scale, Boolean if data_type == "bool" else Nominal)
        assert scale.values == values

    def test_mapping_categorical_data(self, cat_vector):

        mapping = Fill().get_mapping(Nominal(), cat_vector)
        assert_array_equal(mapping([0, 1, 0]), [True, False, True])

    def test_mapping_numeric_data(self, num_vector):

        mapping = Fill().get_mapping(Nominal(), num_vector)
        assert_array_equal(mapping([0, 1, 0]), [True, False, True])

    def test_mapping_list(self, cat_vector):

        mapping = Fill().get_mapping(Nominal([False, True]), cat_vector)
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])

    def test_mapping_truthy_list(self, cat_vector):

        mapping = Fill().get_mapping(Nominal([0, 1]), cat_vector)
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])

    def test_mapping_dict(self, cat_vector):

        values = dict(zip(cat_vector.unique(), [False, True]))
        mapping = Fill().get_mapping(Nominal(values), cat_vector)
        assert_array_equal(mapping([0, 1, 0]), [False, True, False])

    def test_cycle_warning(self):

        x = pd.Series(["a", "b", "c"])
        with pytest.warns(UserWarning, match="The variable assigned to fill"):
            Fill().get_mapping(Nominal(), x)

    def test_values_error(self):

        x = pd.Series(["a", "b"])
        with pytest.raises(TypeError, match="Scale values for fill must be"):
            Fill().get_mapping(Nominal("bad_values"), x)


class IntervalBase(DataFixtures):

    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min())

    @pytest.mark.parametrize("data_type,scale_class", [
        ("cat", Nominal),
        ("num", Continuous),
        ("bool", Boolean),
    ])
    def test_default(self, data_type, scale_class, vectors):

        x = vectors[data_type]
        scale = self.prop().default_scale(x)
        assert isinstance(scale, scale_class)

    @pytest.mark.parametrize("arg,data_type,scale_class", [
        ((1, 3), "cat", Nominal),
        ((1, 3), "num", Continuous),
        ((1, 3), "bool", Boolean),
        ([1, 2, 3], "cat", Nominal),
        ([1, 2, 3], "num", Nominal),
        ([1, 3], "bool", Boolean),
        ({"a": 1, "b": 3, "c": 2}, "cat", Nominal),
        ({2: 1, 4: 3, 8: 2}, "num", Nominal),
        ({True: 4, False: 2}, "bool", Boolean),
    ])
    def test_inference(self, arg, data_type, scale_class, vectors):

        x = vectors[data_type]
        scale = self.prop().infer_scale(arg, x)
        assert isinstance(scale, scale_class)
        assert scale.values == arg

    def test_mapped_interval_numeric(self, num_vector):

        mapping = self.prop().get_mapping(Continuous(), num_vector)
        assert_array_equal(mapping([0, 1]), self.prop().default_range)

    def test_mapped_interval_categorical(self, cat_vector):

        mapping = self.prop().get_mapping(Nominal(), cat_vector)
        n = cat_vector.nunique()
        assert_array_equal(mapping([n - 1, 0]), self.prop().default_range)

    def test_bad_scale_values_numeric_data(self, num_vector):

        prop_name = self.prop.__name__.lower()
        err_stem = (
            f"Values for {prop_name} variables with Continuous scale must be 2-tuple"
        )

        with pytest.raises(TypeError, match=f"{err_stem}; not <class 'str'>."):
            self.prop().get_mapping(Continuous("abc"), num_vector)

        with pytest.raises(TypeError, match=f"{err_stem}; not 3-tuple."):
            self.prop().get_mapping(Continuous((1, 2, 3)), num_vector)

    def test_bad_scale_values_categorical_data(self, cat_vector):

        prop_name = self.prop.__name__.lower()
        err_text = f"Values for {prop_name} variables with Nominal scale"
        with pytest.raises(TypeError, match=err_text):
            self.prop().get_mapping(Nominal("abc"), cat_vector)


class TestAlpha(IntervalBase):
    prop = Alpha


class TestLineWidth(IntervalBase):
    prop = LineWidth

    def test_rcparam_default(self):

        with mpl.rc_context({"lines.linewidth": 2}):
            assert self.prop().default_range == (1, 4)


class TestEdgeWidth(IntervalBase):
    prop = EdgeWidth

    def test_rcparam_default(self):

        with mpl.rc_context({"patch.linewidth": 2}):
            assert self.prop().default_range == (1, 4)


class TestPointSize(IntervalBase):
    prop = PointSize

    def test_areal_scaling_numeric(self, num_vector):

        limits = 5, 10
        scale = Continuous(limits)
        mapping = self.prop().get_mapping(scale, num_vector)
        x = np.linspace(0, 1, 6)
        expected = np.sqrt(np.linspace(*np.square(limits), num=len(x)))
        assert_array_equal(mapping(x), expected)

    def test_areal_scaling_categorical(self, cat_vector):

        limits = (2, 4)
        scale = Nominal(limits)
        mapping = self.prop().get_mapping(scale, cat_vector)
        assert_array_equal(mapping(np.arange(3)), [4, np.sqrt(10), 2])
