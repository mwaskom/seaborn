
import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._core.scales import (
    Nominal,
    Continuous,
)
from seaborn._core.properties import (
    SizedProperty,
    ObjectProperty,
    Coordinate,
    Alpha,
    Color,
    Fill,
)
from seaborn.palettes import color_palette


class TestContinuous:

    @pytest.fixture
    def x(self):
        return pd.Series([1, 3, 9], name="x", dtype=float)

    def test_coordinate_defaults(self, x):

        s = Continuous().setup(x, Coordinate())
        assert_series_equal(s(x), x)
        assert_series_equal(s.invert_transform(x), x)

    def test_coordinate_transform(self, x):

        s = Continuous(transform="log").setup(x, Coordinate())
        assert_series_equal(s(x), np.log10(x))
        assert_series_equal(s.invert_transform(s(x)), x)

    def test_coordinate_transform_with_parameter(self, x):

        s = Continuous(transform="pow3").setup(x, Coordinate())
        assert_series_equal(s(x), np.power(x, 3))
        assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_defaults(self, x):

        s = Continuous().setup(x, SizedProperty())
        assert_array_equal(s(x), [0, .25, 1])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_range(self, x):

        s = Continuous((1, 3)).setup(x, SizedProperty())
        assert_array_equal(s(x), [1, 1.5, 3])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_norm(self, x):

        s = Continuous(norm=(3, 7)).setup(x, SizedProperty())
        assert_array_equal(s(x), [-.5, 0, 1.5])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_sized_with_range_norm_and_transform(self, x):

        x = pd.Series([1, 10, 100])
        # TODO param order?
        s = Continuous((2, 3), (10, 100), "log").setup(x, SizedProperty())
        assert_array_equal(s(x), [1, 2, 3])
        # TODO assert_series_equal(s.invert_transform(s(x)), x)

    def test_color_defaults(self, x):

        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous().setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_named_range(self, x):

        cmap = color_palette("viridis", as_cmap=True)
        s = Continuous("viridis").setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_tuple_range(self, x):

        cmap = color_palette("blend:b,g", as_cmap=True)
        s = Continuous(("b", "g")).setup(x, Color())
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    def test_color_with_norm(self, x):

        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous(norm=(3, 7)).setup(x, Color())
        assert_array_equal(s(x), cmap([-.5, 0, 1.5])[:, :3])  # FIXME RGBA

    def test_color_with_transform(self, x):

        x = pd.Series([1, 10, 100], name="x", dtype=float)
        cmap = color_palette("ch:", as_cmap=True)
        s = Continuous(transform="log").setup(x, Color())
        assert_array_equal(s(x), cmap([0, .5, 1])[:, :3])  # FIXME RGBA


class TestNominal:

    @pytest.fixture
    def x(self):
        return pd.Series(["a", "c", "b", "c"], name="x")

    @pytest.fixture
    def y(self):
        return pd.Series([1, -1.5, 3, -1.5], name="y")

    def test_coordinate_defaults(self, x):

        s = Nominal().setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_with_order(self, x):

        s = Nominal(order=["a", "b", "c"]).setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_with_subset_order(self, x):

        s = Nominal(order=["c", "a"]).setup(x, Coordinate())
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        assert_array_equal(s.invert_transform(s(x)), s(x))

    def test_coordinate_axis(self, x):

        ax = mpl.figure.Figure().subplots()
        s = Nominal().setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["a", "c", "b"]

    def test_coordinate_axis_with_order(self, x):

        order = ["a", "b", "c"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order).setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == order

    def test_coordinate_axis_with_subset_order(self, x):

        order = ["c", "a"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order).setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == [*order, ""]

    def test_coordinate_numeric_data(self, y):

        ax = mpl.figure.Figure().subplots()
        s = Nominal().setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([1, 0, 2, 0], float))
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["-1.5", "1.0", "3.0"]

    def test_coordinate_numeric_data_with_order(self, y):

        order = [1, 4, -1.5]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order).setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([0, 2, np.nan, 2], float))
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["1.0", "4.0", "-1.5"]

    def test_color_defaults(self, x):

        s = Nominal().setup(x, Color())
        cs = color_palette()
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_named_palette(self, x):

        pal = "flare"
        s = Nominal(pal).setup(x, Color())
        cs = color_palette(pal, 3)
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_list_palette(self, x):

        cs = color_palette("crest", 3)
        s = Nominal(cs).setup(x, Color())
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    def test_color_dict_palette(self, x):

        cs = color_palette("crest", 3)
        pal = dict(zip("bac", cs))
        s = Nominal(pal).setup(x, Color())
        assert_array_equal(s(x), [cs[1], cs[2], cs[0], cs[2]])

    def test_color_numeric_data(self, y):

        s = Nominal().setup(y, Color())
        cs = color_palette()
        assert_array_equal(s(y), [cs[1], cs[0], cs[2], cs[0]])

    def test_color_numeric_with_order_subset(self, y):

        s = Nominal(order=[-1.5, 1]).setup(y, Color())
        c1, c2 = color_palette(n_colors=2)
        null = (np.nan, np.nan, np.nan)
        assert_array_equal(s(y), [c2, c1, null, c1])

    @pytest.mark.xfail(reason="Need to sort out float/int order")
    def test_color_numeric_int_float_mix(self):

        z = pd.Series([1, 2], name="z")
        s = Nominal(order=[1.0, 2]).setup(z, Color())
        c1, c2 = color_palette(n_colors=2)
        null = (np.nan, np.nan, np.nan)
        assert_array_equal(s(z), [c1, null, c2])

    def test_object_defaults(self, x):

        class MockProperty(ObjectProperty):
            def _default_values(self, n):
                return list("xyz"[:n])

        s = Nominal().setup(x, MockProperty())
        assert s(x) == ["x", "y", "z", "y"]

    def test_object_list(self, x):

        vs = ["x", "y", "z"]
        s = Nominal(vs).setup(x, ObjectProperty())
        assert s(x) == ["x", "y", "z", "y"]

    def test_object_dict(self, x):

        vs = {"a": "x", "b": "y", "c": "z"}
        s = Nominal(vs).setup(x, ObjectProperty())
        assert s(x) == ["x", "z", "y", "z"]

    def test_object_order(self, x):

        vs = ["x", "y", "z"]
        s = Nominal(vs, order=["c", "a", "b"]).setup(x, ObjectProperty())
        assert s(x) == ["y", "x", "z", "x"]

    def test_object_order_subset(self, x):

        vs = ["x", "y"]
        s = Nominal(vs, order=["a", "c"]).setup(x, ObjectProperty())
        assert s(x) == ["x", "y", None, "y"]

    def test_objects_that_are_weird(self, x):

        vs = [("x", 1), (None, None, 0), {}]
        s = Nominal(vs).setup(x, ObjectProperty())
        assert s(x) == [vs[0], vs[1], vs[2], vs[1]]

    def test_alpha_default(self, x):

        s = Nominal().setup(x, Alpha())
        assert_array_equal(s(x), [.95, .55, .15, .55])

    def test_fill(self):

        x = pd.Series(["a", "a", "b", "a"], name="x")
        s = Nominal().setup(x, Fill())
        assert_array_equal(s(x), [True, True, False, True])

    def test_fill_dict(self):

        x = pd.Series(["a", "a", "b", "a"], name="x")
        vs = {"a": False, "b": True}
        s = Nominal(vs).setup(x, Fill())
        assert_array_equal(s(x), [False, False, True, False])

    def test_fill_nunique_warning(self):

        x = pd.Series(["a", "b", "c", "a", "b"], name="x")
        with pytest.warns(UserWarning, match="There are only two possible"):
            s = Nominal().setup(x, Fill())
        assert_array_equal(s(x), [True, False, True, True, False])
