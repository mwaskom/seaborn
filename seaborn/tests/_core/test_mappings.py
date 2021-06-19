
import numpy as np
import pandas as pd
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize, to_rgb

import pytest
from numpy.testing import assert_array_equal

from seaborn.palettes import color_palette
from seaborn._core.rules import categorical_order
from seaborn._core.scales import ScaleWrapper, CategoricalScale
from seaborn._core.mappings import GroupMapping, HueMapping


class TestGroupMapping:

    def test_levels(self):

        x = pd.Series(["a", "c", "b", "b", "d"])
        m = GroupMapping().setup(x)
        assert m.levels == categorical_order(x)


class TestHueMapping:

    @pytest.fixture
    def num_vector(self, long_df):
        return long_df["s"]

    @pytest.fixture
    def num_order(self, num_vector):
        return categorical_order(num_vector)

    @pytest.fixture
    def num_norm(self, num_vector):
        norm = Normalize()
        norm.autoscale(num_vector)
        return norm

    @pytest.fixture
    def cat_vector(self, long_df):
        return long_df["a"]

    @pytest.fixture
    def cat_order(self, cat_vector):
        return categorical_order(cat_vector)

    def test_categorical_default_palette(self, cat_vector, cat_order):

        expected_lookup_table = dict(zip(cat_order, color_palette()))
        m = HueMapping().setup(cat_vector)

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_default_palette_large(self):

        vector = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
        n_colors = len(vector)
        expected_lookup_table = dict(zip(vector, color_palette("husl", n_colors)))
        m = HueMapping().setup(vector)

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_named_palette(self, cat_vector, cat_order):

        palette = "Blues"
        m = HueMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette
        assert m.levels == cat_order

        expected_lookup_table = dict(
            zip(cat_order, color_palette(palette, len(cat_order)))
        )

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_list_palette(self, cat_vector, cat_order):

        palette = color_palette("Reds", len(cat_order))
        m = HueMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette

        expected_lookup_table = dict(zip(cat_order, palette))

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_implied_by_list_palette(self, num_vector, num_order):

        palette = color_palette("Reds", len(num_order))
        m = HueMapping(palette=palette).setup(num_vector)
        assert m.palette == palette

        expected_lookup_table = dict(zip(num_order, palette))

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_dict_palette(self, cat_vector, cat_order):

        palette = dict(zip(cat_order, color_palette("Greens")))
        m = HueMapping(palette=palette).setup(cat_vector)
        assert m.palette == palette

        for level, color in palette.items():
            assert m(level) == color

    def test_categorical_implied_by_dict_palette(self, num_vector, num_order):

        palette = dict(zip(num_order, color_palette("Greens")))
        m = HueMapping(palette=palette).setup(num_vector)
        assert m.palette == palette

        for level, color in palette.items():
            assert m(level) == color

    def test_categorical_dict_with_missing_keys(self, cat_vector, cat_order):

        palette = dict(zip(cat_order[1:], color_palette("Purples")))
        with pytest.raises(ValueError):
            HueMapping(palette=palette).setup(cat_vector)

    def test_categorical_list_with_wrong_length(self, cat_vector, cat_order):

        palette = color_palette("Oranges", len(cat_order) - 1)
        with pytest.raises(ValueError):
            HueMapping(palette=palette).setup(cat_vector)

    def test_categorical_with_ordered_scale(self, cat_vector):

        cat_order = list(cat_vector.unique()[::-1])
        scale = ScaleWrapper(CategoricalScale(order=cat_order), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(cat_order))

        m = HueMapping(palette=palette).setup(cat_vector, scale)
        assert m.levels == cat_order

        expected_lookup_table = dict(zip(cat_order, colors))

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_implied_by_scale(self, num_vector, num_order):

        scale = ScaleWrapper(CategoricalScale(), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(num_order))

        m = HueMapping(palette=palette).setup(num_vector, scale)
        assert m.levels == num_order

        expected_lookup_table = dict(zip(num_order, colors))

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_implied_by_ordered_scale(self, num_vector):

        order = num_vector.unique()
        if order[0] < order[1]:
            order[[0, 1]] = order[[1, 0]]
        order = list(order)

        scale = ScaleWrapper(CategoricalScale(order=order), "categorical")

        palette = "deep"
        colors = color_palette(palette, len(order))

        m = HueMapping(palette=palette).setup(num_vector, scale)
        assert m.levels == order

        expected_lookup_table = dict(zip(order, colors))

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_with_ordered_categories(self, cat_vector, cat_order):

        new_order = list(reversed(cat_order))
        new_vector = cat_vector.astype("category").cat.set_categories(new_order)

        expected_lookup_table = dict(zip(new_order, color_palette()))

        m = HueMapping().setup(new_vector)

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_implied_by_categories(self, num_vector):

        new_vector = num_vector.astype("category")
        new_order = categorical_order(new_vector)

        expected_lookup_table = dict(zip(new_order, color_palette()))

        m = HueMapping().setup(new_vector)

        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_implied_by_palette(self, num_vector, num_order):

        palette = "bright"
        expected_lookup_table = dict(zip(num_order, color_palette(palette)))
        m = HueMapping(palette=palette).setup(num_vector)
        for level, color in expected_lookup_table.items():
            assert m(level) == color

    def test_categorical_from_binary_data(self):

        vector = pd.Series([1, 0, 0, 0, 1, 1, 1])
        expected_palette = dict(zip([0, 1], color_palette()))
        m = HueMapping().setup(vector)

        for level, color in expected_palette.items():
            assert m(level) == color

        first_color, *_ = color_palette()

        for val in [0, 1]:
            m = HueMapping().setup(pd.Series([val] * 4))
            assert m(val) == first_color

    def test_categorical_multi_lookup(self):

        x = pd.Series(["a", "b", "c"])
        colors = color_palette(n_colors=len(x))
        m = HueMapping().setup(x)
        assert_array_equal(m(x), np.stack(colors))

    def test_categorical_multi_lookup_categorical(self):

        x = pd.Series(["a", "b", "c"]).astype("category")
        colors = color_palette(n_colors=len(x))
        m = HueMapping().setup(x)
        assert_array_equal(m(x), np.stack(colors))

    def test_numeric_default_palette(self, num_vector, num_order, num_norm):

        m = HueMapping().setup(num_vector)
        expected_cmap = color_palette("ch:", as_cmap=True)
        for level in num_order:
            assert m(level) == to_rgb(expected_cmap(num_norm(level)))

    def test_numeric_named_palette(self, num_vector, num_order, num_norm):

        palette = "viridis"
        m = HueMapping(palette=palette).setup(num_vector)
        expected_cmap = color_palette(palette, as_cmap=True)
        for level in num_order:
            assert m(level) == to_rgb(expected_cmap(num_norm(level)))

    def test_numeric_colormap_palette(self, num_vector, num_order, num_norm):

        cmap = color_palette("rocket", as_cmap=True)
        m = HueMapping(palette=cmap).setup(num_vector)
        for level in num_order:
            assert m(level) == to_rgb(cmap(num_norm(level)))

    def test_numeric_norm_limits(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        cmap = color_palette("rocket", as_cmap=True)
        scale = ScaleWrapper(LinearScale("hue"), "numeric", norm=lims)
        norm = Normalize(*lims)
        m = HueMapping(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert m(level) == to_rgb(cmap(norm(level)))

    def test_numeric_norm_object(self, num_vector, num_order):

        lims = (num_vector.min() - 1, num_vector.quantile(.5))
        norm = Normalize(*lims)
        cmap = color_palette("rocket", as_cmap=True)
        scale = ScaleWrapper(LinearScale("hue"), "numeric", norm=norm)
        m = HueMapping(palette=cmap).setup(num_vector, scale)
        for level in num_order:
            assert m(level) == to_rgb(cmap(norm(level)))

    def test_numeric_dict_palette_with_norm(self, num_vector, num_order, num_norm):

        palette = dict(zip(num_order, color_palette()))
        scale = ScaleWrapper(LinearScale("hue"), "numeric", norm=num_norm)
        m = HueMapping(palette=palette).setup(num_vector, scale)
        for level, color in palette.items():
            assert m(level) == to_rgb(color)

    def test_numeric_multi_lookup(self, num_vector, num_norm):

        cmap = color_palette("mako", as_cmap=True)
        m = HueMapping(palette=cmap).setup(num_vector)
        assert_array_equal(m(num_vector), cmap(num_norm(num_vector))[:, :3])

    def test_bad_palette(self, num_vector):

        with pytest.raises(ValueError):
            HueMapping(palette="not_a_palette").setup(num_vector)

    def test_bad_norm(self, num_vector):

        norm = "not_a_norm"
        scale = ScaleWrapper(LinearScale("hue"), "numeric", norm=norm)
        with pytest.raises(ValueError):
            HueMapping().setup(num_vector, scale)
