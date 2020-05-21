from itertools import product
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pytest
from numpy.testing import assert_array_equal

from ..palettes import color_palette
from ..utils import categorical_order

from ..core import (
    unique_dashes,
    unique_markers,
)

from ..relational import (
    _RelationalPlotter,
    _LinePlotter,
    _ScatterPlotter,
    relplot,
    lineplot,
    scatterplot
)


@pytest.fixture(params=[
    dict(x="x", y="y"),
    dict(x="t", y="y"),
    dict(x="a", y="y"),
    dict(x="x", y="y", hue="y"),
    dict(x="x", y="y", hue="a"),
    dict(x="x", y="y", size="a"),
    dict(x="x", y="y", style="a"),
    dict(x="x", y="y", hue="s"),
    dict(x="x", y="y", size="s"),
    dict(x="x", y="y", style="s"),
    dict(x="x", y="y", hue="a", style="a"),
    dict(x="x", y="y", hue="a", size="b", style="b"),
])
def long_semantics(request):
    return request.param


class Helpers:

    # TODO Better place for these?

    def scatter_rgbs(self, collections):
        rgbs = []
        for col in collections:
            rgb = tuple(col.get_facecolor().squeeze()[:3])
            rgbs.append(rgb)
        return rgbs

    def colors_equal(self, *args):

        equal = True
        for c1, c2 in zip(*args):
            c1 = mpl.colors.colorConverter.to_rgb(np.squeeze(c1))
            c2 = mpl.colors.colorConverter.to_rgb(np.squeeze(c1))
            equal &= c1 == c2
        return equal

    def paths_equal(self, *args):

        equal = True
        for p1, p2 in zip(*args):
            equal &= np.array_equal(p1.vertices, p2.vertices)
            equal &= np.array_equal(p1.codes, p2.codes)
        return equal


class TestRelationalPlotter(Helpers):

    def test_wide_df_variables(self, wide_df):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == np.product(wide_df.shape)

        x = p.plot_data["x"]
        expected_x = np.tile(wide_df.index, wide_df.shape[1])
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = wide_df.values.ravel(order="f")
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(wide_df.columns.values, wide_df.shape[0])
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    def test_wide_df_with_nonnumeric_variables(self, long_df):

        p = _RelationalPlotter()
        p.establish_variables(data=long_df)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        numeric_df = long_df.select_dtypes("number")

        assert len(p.plot_data) == np.product(numeric_df.shape)

        x = p.plot_data["x"]
        expected_x = np.tile(numeric_df.index, numeric_df.shape[1])
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = numeric_df.values.ravel(order="f")
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(
            numeric_df.columns.values, numeric_df.shape[0]
        )
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] == numeric_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == numeric_df.columns.name
        assert p.variables["style"] == numeric_df.columns.name

    def test_wide_array_variables(self, wide_array):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_array)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert len(p.plot_data) == np.product(wide_array.shape)

        nrow, ncol = wide_array.shape

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(nrow), ncol)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = wide_array.ravel(order="f")
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(np.arange(ncol), nrow)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_flat_array_variables(self, flat_array):

        p = _RelationalPlotter()
        p.establish_variables(data=flat_array)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y"]
        assert len(p.plot_data) == np.product(flat_array.shape)

        x = p.plot_data["x"]
        expected_x = np.arange(flat_array.shape[0])
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = flat_array
        assert_array_equal(y, expected_y)

        assert p.plot_data["hue"].isnull().all()
        assert p.plot_data["style"].isnull().all()
        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None

    def test_flat_list_variables(self, flat_list):

        p = _RelationalPlotter()
        p.establish_variables(data=flat_list)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y"]
        assert len(p.plot_data) == len(flat_list)

        x = p.plot_data["x"]
        expected_x = np.arange(len(flat_list))
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = flat_list
        assert_array_equal(y, expected_y)

        assert p.plot_data["hue"].isnull().all()
        assert p.plot_data["style"].isnull().all()
        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None

    def test_flat_series_variables(self, flat_series):

        p = _RelationalPlotter()
        p.establish_variables(data=flat_series)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y"]
        assert len(p.plot_data) == len(flat_series)

        x = p.plot_data["x"]
        expected_x = flat_series.index
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = flat_series
        assert_array_equal(y, expected_y)

        assert p.variables["x"] is flat_series.index.name
        assert p.variables["y"] is flat_series.name

    def test_wide_list_of_series_variables(self, wide_list_of_series):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_list_of_series)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_list_of_series)
        chunk_size = max(len(l) for l in wide_list_of_series)

        assert len(p.plot_data) == chunks * chunk_size

        index_union = np.unique(
            np.concatenate([s.index for s in wide_list_of_series])
        )

        x = p.plot_data["x"]
        expected_x = np.tile(index_union, chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"]
        expected_y = np.concatenate([
            s.reindex(index_union) for s in wide_list_of_series
        ])
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        series_names = [s.name for s in wide_list_of_series]
        expected_hue = np.repeat(series_names, chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_wide_list_of_arrays_variables(self, wide_list_of_arrays):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_list_of_arrays)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_list_of_arrays)
        chunk_size = max(len(l) for l in wide_list_of_arrays)

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(wide_list_of_arrays)
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(np.arange(chunks), chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_wide_list_of_list_variables(self, wide_list_of_lists):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_list_of_lists)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_list_of_lists)
        chunk_size = max(len(l) for l in wide_list_of_lists)

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(wide_list_of_lists)
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(np.arange(chunks), chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_wide_dict_of_series_variables(self, wide_dict_of_series):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_dict_of_series)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_dict_of_series)
        chunk_size = max(len(l) for l in wide_dict_of_series.values())

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(list(wide_dict_of_series.values()))
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(list(wide_dict_of_series), chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_wide_dict_of_arrays_variables(self, wide_dict_of_arrays):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_dict_of_arrays)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_dict_of_arrays)
        chunk_size = max(len(l) for l in wide_dict_of_arrays.values())

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(list(wide_dict_of_arrays.values()))
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(list(wide_dict_of_arrays), chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_wide_dict_of_lists_variables(self, wide_dict_of_lists):

        p = _RelationalPlotter()
        p.establish_variables(data=wide_dict_of_lists)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_dict_of_lists)
        chunk_size = max(len(l) for l in wide_dict_of_lists.values())

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(list(wide_dict_of_lists.values()))
        assert_array_equal(y, expected_y)

        hue = p.plot_data["hue"]
        expected_hue = np.repeat(list(wide_dict_of_lists), chunk_size)
        assert_array_equal(hue, expected_hue)

        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        assert p.plot_data["size"].isnull().all()

        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    def test_long_df(self, long_df, long_semantics):

        p = _RelationalPlotter(data=long_df, variables=long_semantics)
        assert p.input_format == "long"
        assert p.variables == long_semantics

        for key, val in long_semantics.items():
            assert_array_equal(p.plot_data[key], long_df[val])

        for col in set(p.semantics) - set(long_semantics):
            assert p.plot_data[col].isnull().all()

    def test_long_df_with_index(self, long_df, long_semantics):

        p = _RelationalPlotter(
            data=long_df.set_index("a"),
            variables=long_semantics,
        )
        assert p.input_format == "long"
        assert p.variables == long_semantics

        for key, val in long_semantics.items():
            assert_array_equal(p.plot_data[key], long_df[val])

        for col in set(p.semantics) - set(long_semantics):
            assert p.plot_data[col].isnull().all()

    def test_long_df_with_multiindex(self, long_df, long_semantics):

        p = _RelationalPlotter(
            data=long_df.set_index(["a", "x"]),
            variables=long_semantics,
        )
        assert p.input_format == "long"
        assert p.variables == long_semantics

        for key, val in long_semantics.items():
            assert_array_equal(p.plot_data[key], long_df[val])

        for col in set(p.semantics) - set(long_semantics):
            assert p.plot_data[col].isnull().all()

    def test_long_dict(self, long_dict, long_semantics):

        p = _RelationalPlotter(
            data=long_dict,
            variables=long_semantics,
        )
        assert p.input_format == "long"
        assert p.variables == long_semantics

        for key, val in long_semantics.items():
            assert_array_equal(p.plot_data[key], pd.Series(long_dict[val]))

        for col in set(p.semantics) - set(long_semantics):
            assert p.plot_data[col].isnull().all()

    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    def test_long_vectors(self, long_df, long_semantics, vector_type):

        variables = {key: long_df[val] for key, val in long_semantics.items()}
        if vector_type == "numpy":
            # Requires pandas >= 0.24
            # {key: val.to_numpy() for key, val in variables.items()}
            variables = {
                key: np.asarray(val) for key, val in variables.items()
            }
        elif vector_type == "list":
            # Requires pandas >= 0.24
            # {key: val.to_list() for key, val in variables.items()}
            variables = {
                key: val.tolist() for key, val in variables.items()
            }

        p = _RelationalPlotter(variables=variables)
        assert p.input_format == "long"

        assert list(p.variables) == list(long_semantics)
        if vector_type == "series":
            assert p.variables == long_semantics

        for key, val in long_semantics.items():
            assert_array_equal(p.plot_data[key], long_df[val])

        for col in set(p.semantics) - set(long_semantics):
            assert p.plot_data[col].isnull().all()

    def test_long_undefined_variables(self, long_df):

        p = _RelationalPlotter()

        with pytest.raises(ValueError):
            p.establish_variables(
                data=long_df, variables=dict(x="not_in_df"),
            )

        with pytest.raises(ValueError):
            p.establish_variables(
                data=long_df, variables=dict(x="x", y="not_in_df"),
            )

        with pytest.raises(ValueError):
            p.establish_variables(
                data=long_df, variables=dict(x="x", y="y", hue="not_in_df"),
            )

    @pytest.mark.parametrize(
        "arg", [[], np.array([]), pd.DataFrame()],
    )
    def test_empty_data_input(self, arg):

        p = _RelationalPlotter(data=arg)
        assert not p.variables

        if not isinstance(arg, pd.DataFrame):
            p = _RelationalPlotter(variables=dict(x=arg, y=arg))
            assert not p.variables

    def test_units(self, repeated_df):

        p = _RelationalPlotter(
            data=repeated_df,
            variables=dict(x="x", y="y", units="u"),
        )
        assert_array_equal(p.plot_data["units"], repeated_df["u"])

    def test_parse_hue_null(self, long_df, null_series):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=null_series)
        )
        m = p._hue_map
        assert m.levels == [None]
        assert m.palette is None
        assert m.map_type is None
        assert m.cmap is None
    def test_parse_size(self, long_df):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="s"),
        )

        # Test default size limits and range
        default_limits = p.plot_data["size"].min(), p.plot_data["size"].max()
        default_range = p._default_size_range
        p.parse_size(p.plot_data["size"])
        assert p.size_limits == default_limits
        size_range = min(p.sizes.values()), max(p.sizes.values())
        assert size_range == default_range

        # Test specified size limits
        size_limits = (1, 5)
        p.parse_size(p.plot_data["size"], norm=size_limits)
        assert p.size_limits == size_limits

        # Test specified size range
        sizes = (.1, .5)
        p.parse_size(p.plot_data["size"], sizes=sizes)
        assert p.size_limits == default_limits

        # Test size values with normalization range
        sizes = (1, 5)
        size_norm = (1, 10)
        p.parse_size(p.plot_data["size"], sizes=sizes, norm=size_norm)
        normalize = mpl.colors.Normalize(*size_norm, clip=True)
        for level, width in p.sizes.items():
            assert width == sizes[0] + (sizes[1] - sizes[0]) * normalize(level)

        # Test size values with normalization object
        sizes = (1, 5)
        size_norm = mpl.colors.LogNorm(1, 10, clip=False)
        p.parse_size(p.plot_data["size"], sizes=sizes, norm=size_norm)
        assert p.size_norm.clip
        for level, width in p.sizes.items():
            assert width == sizes[0] + (sizes[1] - sizes[0]) * size_norm(level)

        # Use a categorical variable
        var = "a"
        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size=var),
        )

        # Test specified size order
        levels = long_df[var].unique()
        sizes = [1, 4, 6]
        size_order = [levels[1], levels[2], levels[0]]
        p.parse_size(p.plot_data["size"], sizes=sizes, order=size_order)
        assert p.sizes == dict(zip(size_order, sizes))

        # Test list of sizes
        levels = categorical_order(long_df[var])
        sizes = list(np.random.rand(len(levels)))
        p.parse_size(p.plot_data["size"], sizes=sizes)
        assert p.sizes == dict(zip(levels, sizes))

        # Test dict of sizes
        sizes = dict(zip(levels, np.random.rand(len(levels))))
        p.parse_size(p.plot_data["size"], sizes=sizes)
        assert p.sizes == sizes

        # Test sizes list with wrong length
        sizes = list(np.random.rand(len(levels) + 1))
        with pytest.raises(ValueError):
            p.parse_size(p.plot_data["size"], sizes=sizes)

        # Test sizes dict with missing levels
        sizes = dict(zip(levels, np.random.rand(len(levels) - 1)))
        with pytest.raises(ValueError):
            p.parse_size(p.plot_data["size"], sizes=sizes)

        # Test bad sizes argument
        sizes = "bad_size"
        with pytest.raises(ValueError):
            p.parse_size(p.plot_data["size"], sizes=sizes)

        # Test bad norm argument
        size_norm = "not a norm"
        with pytest.raises(ValueError):
            p.parse_size(p.plot_data["size"], norm=size_norm)

    def test_parse_style(self, long_df):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="a"),
        )

        # Test defaults
        markers, dashes = True, True
        p.parse_style(p.plot_data["style"], markers, dashes)

        n = len(p.style_levels)
        assert p.dashes == dict(zip(p.style_levels, unique_dashes(n)))

        actual_marker_paths = {
            k: mpl.markers.MarkerStyle(m).get_path()
            for k, m in p.markers.items()
        }
        expected_marker_paths = {
            k: mpl.markers.MarkerStyle(m).get_path()
            for k, m in zip(p.style_levels, unique_markers(n))
        }
        assert actual_marker_paths == expected_marker_paths

        # Test lists
        markers, dashes = ["o", "s", "d"], [(1, 0), (1, 1), (2, 1, 3, 1)]
        p.parse_style(p.plot_data["style"], markers, dashes)
        assert p.markers == dict(zip(p.style_levels, markers))
        assert p.dashes == dict(zip(p.style_levels, dashes))

        # Test dicts
        markers = dict(zip(p.style_levels, markers))
        dashes = dict(zip(p.style_levels, dashes))
        p.parse_style(p.plot_data["style"], markers, dashes)
        assert p.markers == markers
        assert p.dashes == dashes

        # Test style order with defaults
        style_order = np.take(p.style_levels, [1, 2, 0])
        markers = dashes = True
        p.parse_style(p.plot_data["style"], markers, dashes, style_order)

        n = len(style_order)
        assert p.dashes == dict(zip(style_order, unique_dashes(n)))

        actual_marker_paths = {
            k: mpl.markers.MarkerStyle(m).get_path()
            for k, m in p.markers.items()
        }
        expected_marker_paths = {
            k: mpl.markers.MarkerStyle(m).get_path()
            for k, m in zip(style_order, unique_markers(n))
        }
        assert actual_marker_paths == expected_marker_paths

        # Test too many levels with style lists
        markers, dashes = ["o", "s"], False
        with pytest.raises(ValueError):
            p.parse_style(p.plot_data["style"], markers, dashes)

        markers, dashes = False, [(2, 1)]
        with pytest.raises(ValueError):
            p.parse_style(p.plot_data["style"], markers, dashes)

        # Test too many levels with style dicts
        markers, dashes = {"a": "o", "b": "s"}, False
        with pytest.raises(ValueError):
            p.parse_style(p.plot_data["style"], markers, dashes)

        markers, dashes = False, {"a": (1, 0), "b": (2, 1)}
        with pytest.raises(ValueError):
            p.parse_style(p.plot_data["style"], markers, dashes)

        # Test mixture of filled and unfilled markers
        markers, dashes = ["o", "x", "s"], None
        with pytest.raises(ValueError):
            p.parse_style(p.plot_data["style"], markers, dashes)

    def test_subset_data_quantities(self, long_df):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        assert len(list(p.subset_data())) == 1

        # --

        var = "a"
        n_subsets = len(long_df[var].unique())

        for semantic in ["hue", "size", "style"]:

            p = _RelationalPlotter(
                data=long_df,
                variables={"x": "x", "y": "y", semantic: var},
            )
            p.parse_size(p.plot_data["size"])
            p.parse_style(p.plot_data["style"])

            assert len(list(p.subset_data())) == n_subsets

        # --

        var = "a"
        n_subsets = len(long_df[var].unique())

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var, style=var),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        assert len(list(p.subset_data())) == n_subsets

        # --

        var1, var2 = "a", "s"
        n_subsets = len(set(list(map(tuple, long_df[[var1, var2]].values))))

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, style=var2),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        assert len(list(p.subset_data())) == n_subsets

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2, style=var1),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        assert len(list(p.subset_data())) == n_subsets

        # --

        var1, var2, var3 = "a", "s", "b"
        cols = [var1, var2, var3]
        n_subsets = len(set(list(map(tuple, long_df[cols].values))))

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2, style=var3),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        assert len(list(p.subset_data())) == n_subsets

    def test_subset_data_keys(self, long_df):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue is None
            assert size is None
            assert style is None

        # --

        var = "a"

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue in long_df[var].values
            assert size is None
            assert style is None

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size=var),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue is None
            assert size in long_df[var].values
            assert style is None

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style=var),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue is None
            assert size is None
            assert style in long_df[var].values

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var, style=var),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue in long_df[var].values
            assert size is None
            assert style in long_df[var].values

        # --

        var1, var2 = "a", "s"

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        for (hue, size, style), _ in p.subset_data():
            assert hue in long_df[var1].values
            assert size in long_df[var2].values
            assert style is None

    def test_subset_data_values(self, long_df):

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = True
        _, data = next(p.subset_data())
        expected = p.plot_data.loc[:, ["x", "y"]].sort_values(["x", "y"])
        assert_array_equal(data.values, expected)

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = False
        _, data = next(p.subset_data())
        expected = p.plot_data.loc[:, ["x", "y"]]
        assert_array_equal(data.values, expected)

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = True
        for (hue, _, _), data in p.subset_data():
            rows = p.plot_data["hue"] == hue
            cols = ["x", "y"]
            expected = p.plot_data.loc[rows, cols].sort_values(cols)
            assert_array_equal(data.values, expected.values)

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = False
        for (hue, _, _), data in p.subset_data():
            rows = p.plot_data["hue"] == hue
            cols = ["x", "y"]
            expected = p.plot_data.loc[rows, cols]
            assert_array_equal(data.values, expected.values)

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="a"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = True
        for (_, _, style), data in p.subset_data():
            rows = p.plot_data["style"] == style
            cols = ["x", "y"]
            expected = p.plot_data.loc[rows, cols].sort_values(cols)
            assert_array_equal(data.values, expected.values)

        p = _RelationalPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", size="s"),
        )
        p.parse_size(p.plot_data["size"])
        p.parse_style(p.plot_data["style"])
        p.sort = True
        for (hue, size, _), data in p.subset_data():
            rows = (p.plot_data["hue"] == hue) & (p.plot_data["size"] == size)
            cols = ["x", "y"]
            expected = p.plot_data.loc[rows, cols].sort_values(cols)
            assert_array_equal(data.values, expected.values)

    def test_relplot_simple(self, long_df):

        g = relplot(data=long_df, x="x", y="y", kind="scatter")
        x, y = g.ax.collections[0].get_offsets().T
        assert_array_equal(x, long_df["x"])
        assert_array_equal(y, long_df["y"])

        g = relplot(data=long_df, x="x", y="y", kind="line")
        x, y = g.ax.lines[0].get_xydata().T
        expected = long_df.groupby("x").y.mean()
        assert_array_equal(x, expected.index)
        assert y == pytest.approx(expected.values)

        with pytest.raises(ValueError):
            g = relplot(data=long_df, x="x", y="y", kind="not_a_kind")

    def test_relplot_complex(self, long_df):

        for sem in ["hue", "size", "style"]:
            g = relplot(data=long_df, x="x", y="y", **{sem: "a"})
            x, y = g.ax.collections[0].get_offsets().T
            assert_array_equal(x, long_df["x"])
            assert_array_equal(y, long_df["y"])

        for sem in ["hue", "size", "style"]:
            g = relplot(
                data=long_df, x="x", y="y", col="c", **{sem: "a"}
            )
            grouped = long_df.groupby("c")
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                x, y = ax.collections[0].get_offsets().T
                assert_array_equal(x, grp_df["x"])
                assert_array_equal(y, grp_df["y"])

        for sem in ["size", "style"]:
            g = relplot(
                data=long_df, x="x", y="y", hue="b", col="c", **{sem: "a"}
            )
            grouped = long_df.groupby("c")
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                x, y = ax.collections[0].get_offsets().T
                assert_array_equal(x, grp_df["x"])
                assert_array_equal(y, grp_df["y"])

        for sem in ["hue", "size", "style"]:
            g = relplot(
                data=long_df.sort_values(["c", "b"]),
                x="x", y="y", col="b", row="c", **{sem: "a"}
            )
            grouped = long_df.groupby(["c", "b"])
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                x, y = ax.collections[0].get_offsets().T
                assert_array_equal(x, grp_df["x"])
                assert_array_equal(y, grp_df["y"])

    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    def test_relplot_vectors(self, long_df, vector_type):

        semantics = dict(x="x", y="y", hue="f", col="c")
        kws = {key: long_df[val] for key, val in semantics.items()}
        g = relplot(data=long_df, **kws)
        grouped = long_df.groupby("c")
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            x, y = ax.collections[0].get_offsets().T
            assert_array_equal(x, grp_df["x"])
            assert_array_equal(y, grp_df["y"])

    def test_relplot_wide(self, wide_df):

        g = relplot(data=wide_df)
        x, y = g.ax.collections[0].get_offsets().T
        assert_array_equal(y, wide_df.values.T.ravel())

    def test_relplot_hues(self, long_df):

        palette = ["r", "b", "g"]
        g = relplot(
            x="x", y="y", hue="a", style="b", col="c",
            palette=palette, data=long_df
        )

        palette = dict(zip(long_df["a"].unique(), palette))
        grouped = long_df.groupby("c")
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            points = ax.collections[0]
            expected_hues = [palette[val] for val in grp_df["a"]]
            assert self.colors_equal(points.get_facecolors(), expected_hues)

    def test_relplot_sizes(self, long_df):

        sizes = [5, 12, 7]
        g = relplot(
            data=long_df,
            x="x", y="y", size="a", hue="b", col="c",
            sizes=sizes,
        )

        sizes = dict(zip(long_df["a"].unique(), sizes))
        grouped = long_df.groupby("c")
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            points = ax.collections[0]
            expected_sizes = [sizes[val] for val in grp_df["a"]]
            assert_array_equal(points.get_sizes(), expected_sizes)

    def test_relplot_styles(self, long_df):

        markers = ["o", "d", "s"]
        g = relplot(
            data=long_df,
            x="x", y="y", style="a", hue="b", col="c",
            markers=markers,
        )

        paths = []
        for m in markers:
            m = mpl.markers.MarkerStyle(m)
            paths.append(m.get_path().transformed(m.get_transform()))
        paths = dict(zip(long_df["a"].unique(), paths))

        grouped = long_df.groupby("c")
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            points = ax.collections[0]
            expected_paths = [paths[val] for val in grp_df["a"]]
            assert self.paths_equal(points.get_paths(), expected_paths)

    def test_relplot_stringy_numerics(self, long_df):

        long_df["x_str"] = long_df["x"].astype(str)

        g = relplot(data=long_df, x="x", y="y", hue="x_str")
        points = g.ax.collections[0]
        xys = points.get_offsets()
        mask = np.ma.getmask(xys)
        assert not mask.any()
        assert_array_equal(xys, long_df[["x", "y"]])

        g = relplot(data=long_df, x="x", y="y", size="x_str")
        points = g.ax.collections[0]
        xys = points.get_offsets()
        mask = np.ma.getmask(xys)
        assert not mask.any()
        assert_array_equal(xys, long_df[["x", "y"]])

    def test_relplot_legend(self, long_df):

        g = relplot(data=long_df, x="x", y="y")
        assert g._legend is None

        g = relplot(data=long_df, x="x", y="y", hue="a")
        texts = [t.get_text() for t in g._legend.texts]
        expected_texts = np.append(["a"], long_df["a"].unique())
        assert_array_equal(texts, expected_texts)

        g = relplot(data=long_df, x="x", y="y", hue="s", size="s")
        texts = [t.get_text() for t in g._legend.texts]
        assert_array_equal(texts[1:], np.sort(texts[1:]))

        g = relplot(data=long_df, x="x", y="y", hue="a", legend=False)
        assert g._legend is None

        palette = color_palette("deep", len(long_df["b"].unique()))
        a_like_b = dict(zip(long_df["a"].unique(), long_df["b"].unique()))
        long_df["a_like_b"] = long_df["a"].map(a_like_b)
        g = relplot(
            data=long_df,
            x="x", y="y", hue="b", style="a_like_b",
            palette=palette, kind="line", estimator=None,
        )
        lines = g._legend.get_lines()[1:]  # Chop off title dummy
        for line, color in zip(lines, palette):
            assert line.get_color() == color

    def test_ax_kwarg_removal(self, long_df):

        f, ax = plt.subplots()
        with pytest.warns(UserWarning):
            g = relplot(data=long_df, x="x", y="y", ax=ax)
        assert len(ax.collections) == 0
        assert len(g.ax.collections) > 0


class TestLinePlotter(Helpers):

    def test_aggregate(self, long_df):

        p = _LinePlotter(data=long_df, variables=dict(x="x", y="y"))
        p.n_boot = 10000
        p.sort = False

        x = pd.Series(np.tile([1, 2], 100))
        y = pd.Series(np.random.randn(200))
        y_mean = y.groupby(x).mean()

        def sem(x):
            return np.std(x) / np.sqrt(len(x))

        y_sem = y.groupby(x).apply(sem)
        y_cis = pd.DataFrame(dict(low=y_mean - y_sem,
                                  high=y_mean + y_sem),
                             columns=["low", "high"])

        p.ci = 68
        p.estimator = "mean"
        index, est, cis = p.aggregate(y, x)
        assert_array_equal(index.values, x.unique())
        assert est.index.equals(index)
        assert est.values == pytest.approx(y_mean.values)
        assert cis.values == pytest.approx(y_cis.values, 4)
        assert list(cis.columns) == ["low", "high"]

        p.estimator = np.mean
        index, est, cis = p.aggregate(y, x)
        assert_array_equal(index.values, x.unique())
        assert est.index.equals(index)
        assert est.values == pytest.approx(y_mean.values)
        assert cis.values == pytest.approx(y_cis.values, 4)
        assert list(cis.columns) == ["low", "high"]

        p.seed = 0
        _, _, ci1 = p.aggregate(y, x)
        _, _, ci2 = p.aggregate(y, x)
        assert_array_equal(ci1, ci2)

        y_std = y.groupby(x).std()
        y_cis = pd.DataFrame(dict(low=y_mean - y_std,
                                  high=y_mean + y_std),
                             columns=["low", "high"])

        p.ci = "sd"
        index, est, cis = p.aggregate(y, x)
        assert_array_equal(index.values, x.unique())
        assert est.index.equals(index)
        assert est.values == pytest.approx(y_mean.values)
        assert cis.values == pytest.approx(y_cis.values)
        assert list(cis.columns) == ["low", "high"]

        p.ci = None
        index, est, cis = p.aggregate(y, x)
        assert cis is None

        p.ci = 68
        x, y = pd.Series([1, 2, 3]), pd.Series([4, 3, 2])
        index, est, cis = p.aggregate(y, x)
        assert_array_equal(index.values, x)
        assert_array_equal(est.values, y)
        assert cis is None

        x, y = pd.Series([1, 1, 2]), pd.Series([2, 3, 4])
        index, est, cis = p.aggregate(y, x)
        assert cis.loc[2].isnull().all()

        p = _LinePlotter(data=long_df, variables=dict(x="x", y="y"))
        p.estimator = "mean"
        p.n_boot = 100
        p.ci = 95
        x = pd.Categorical(["a", "b", "a", "b"], ["a", "b", "c"])
        y = pd.Series([1, 1, 2, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            index, est, cis = p.aggregate(y, x)
            assert cis.loc[["c"]].isnull().all().all()

    def test_legend_data(self, long_df):

        f, ax = plt.subplots()

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
            legend="full"
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert handles == []

        # --

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_color() for h in handles]
        assert labels == ["a"] + p._hue_map.levels
        assert colors == ["w"] + p._hue_map(p._hue_map.levels)

        # --

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="a"),
            markers=True,
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_color() for h in handles]
        markers = [h.get_marker() for h in handles]
        assert labels == ["a"] + p._hue_map.levels
        assert labels == ["a"] + p.style_levels
        assert colors == ["w"] + p._hue_map(p._hue_map.levels)
        assert markers == [""] + [p.markers[l] for l in p.style_levels]

        # --

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="b"),
            markers=True,
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_color() for h in handles]
        markers = [h.get_marker() for h in handles]
        expected_colors = (["w"] + p._hue_map(p._hue_map.levels)
                           + ["w"] + [".2" for _ in p.style_levels])
        expected_markers = ([""] + ["None" for _ in p._hue_map.levels]
                            + [""] + [p.markers[l] for l in p.style_levels])
        assert labels == ["a"] + p._hue_map.levels + ["b"] + p.style_levels
        assert colors == expected_colors
        assert markers == expected_markers

        # --

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", size="a"),
            legend="full"
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_color() for h in handles]
        widths = [h.get_linewidth() for h in handles]
        assert labels == ["a"] + p._hue_map.levels == ["a"] + p.size_levels
        assert colors == ["w"] + p._hue_map(p._hue_map.levels)
        assert widths == [0] + [p.sizes[l] for l in p.size_levels]

        # --

        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)

        p = _LinePlotter(variables=dict(x=x, y=y, hue=z))

        ax.clear()
        p.legend = "full"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert labels == [str(l) for l in p._hue_map.levels]

        ax.clear()
        p.legend = "brief"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 4

        p = _LinePlotter(variables=dict(x=x, y=y, size=z))

        ax.clear()
        p.legend = "full"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert labels == [str(l) for l in p.size_levels]

        ax.clear()
        p.legend = "brief"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 4

        ax.clear()
        p.legend = "bad_value"
        with pytest.raises(ValueError):
            p.add_legend_data(ax)

        ax.clear()
        p = _LinePlotter(
            variables=dict(x=x, y=y, hue=z),
            legend="brief"
        )
        p.map_hue(norm=mpl.colors.LogNorm()),
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert float(labels[2]) / float(labels[1]) == 10

        ax.clear()
        p = _LinePlotter(
            variables=dict(x=x, y=y, size=z),
            size_norm=mpl.colors.LogNorm(),
            legend="brief"
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert float(labels[2]) / float(labels[1]) == 10

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="f"),
            legend="brief",
        )
        p.add_legend_data(ax)
        expected_levels = ['0.20', '0.24', '0.28', '0.32']
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ["f"] + expected_levels

        ax.clear()
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="f"),
            legend="brief",
        )
        p.add_legend_data(ax)
        expected_levels = ['0.20', '0.24', '0.28', '0.32']
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ["f"] + expected_levels

    def test_plot(self, long_df, repeated_df):

        f, ax = plt.subplots()

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
            sort=False,
            estimator=None
        )
        p.plot(ax, {})
        line, = ax.lines
        assert_array_equal(line.get_xdata(), long_df.x.values)
        assert_array_equal(line.get_ydata(), long_df.y.values)

        ax.clear()
        p.plot(ax, {"color": "k", "label": "test"})
        line, = ax.lines
        assert line.get_color() == "k"
        assert line.get_label() == "test"

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
            sort=True, estimator=None
        )

        ax.clear()
        p.plot(ax, {})
        line, = ax.lines
        sorted_data = long_df.sort_values(["x", "y"])
        assert_array_equal(line.get_xdata(), sorted_data.x.values)
        assert_array_equal(line.get_ydata(), sorted_data.y.values)

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )

        ax.clear()
        p.plot(ax, {})
        assert len(ax.lines) == len(p._hue_map.levels)
        for line, level in zip(ax.lines, p._hue_map.levels):
            assert line.get_color() == p._hue_map(level)

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="a"),
        )

        ax.clear()
        p.plot(ax, {})
        assert len(ax.lines) == len(p.size_levels)
        for line, level in zip(ax.lines, p.size_levels):
            assert line.get_linewidth() == p.sizes[level]

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="a"),
            markers=True,
        )

        ax.clear()
        p.plot(ax, {})
        assert len(ax.lines) == len(p._hue_map.levels) == len(p.style_levels)
        for line, level in zip(ax.lines, p._hue_map.levels):
            assert line.get_color() == p._hue_map(level)
            assert line.get_marker() == p.markers[level]

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="b"),
            markers=True,
        )

        ax.clear()
        p.plot(ax, {})
        levels = product(p._hue_map.levels, p.style_levels)
        assert len(ax.lines) == (len(p._hue_map.levels) * len(p.style_levels))
        for line, (hue, style) in zip(ax.lines, levels):
            assert line.get_color() == p._hue_map(hue)
            assert line.get_marker() == p.markers[style]

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
            estimator="mean", err_style="band", ci="sd", sort=True
        )

        ax.clear()
        p.plot(ax, {})
        line, = ax.lines
        expected_data = long_df.groupby("x").y.mean()
        assert_array_equal(line.get_xdata(), expected_data.index.values)
        assert np.allclose(line.get_ydata(), expected_data.values)
        assert len(ax.collections) == 1

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            estimator="mean", err_style="band", ci="sd"
        )

        ax.clear()
        p.plot(ax, {})
        assert len(ax.lines) == len(ax.collections) == len(p._hue_map.levels)
        for c in ax.collections:
            assert isinstance(c, mpl.collections.PolyCollection)

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            estimator="mean", err_style="bars", ci="sd"
        )

        ax.clear()
        p.plot(ax, {})
        n_lines = len(ax.lines)
        assert n_lines / 2 == len(ax.collections) == len(p._hue_map.levels)
        assert len(ax.collections) == len(p._hue_map.levels)
        for c in ax.collections:
            assert isinstance(c, mpl.collections.LineCollection)

        p = _LinePlotter(
            data=repeated_df,
            variables=dict(x="x", y="y", units="u"),
            estimator=None
        )

        ax.clear()
        p.plot(ax, {})
        n_units = len(repeated_df["u"].unique())
        assert len(ax.lines) == n_units

        p = _LinePlotter(
            data=repeated_df,
            variables=dict(x="x", y="y", hue="a", units="u"),
            estimator=None
        )

        ax.clear()
        p.plot(ax, {})
        n_units *= len(repeated_df["a"].unique())
        assert len(ax.lines) == n_units

        p.estimator = "mean"
        with pytest.raises(ValueError):
            p.plot(ax, {})

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            err_style="band", err_kws={"alpha": .5},
        )

        ax.clear()
        p.plot(ax, {})
        for band in ax.collections:
            assert band.get_alpha() == .5

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            err_style="bars", err_kws={"elinewidth": 2},
        )

        ax.clear()
        p.plot(ax, {})
        for lines in ax.collections:
            assert lines.get_linestyles() == 2

        p.err_style = "invalid"
        with pytest.raises(ValueError):
            p.plot(ax, {})

        x_str = long_df["x"].astype(str)
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=x_str),
        )
        ax.clear()
        p.plot(ax, {})

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y", size=x_str),
        )
        ax.clear()
        p.plot(ax, {})

    def test_axis_labels(self, long_df):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )

        p.plot(ax1, {})
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "y"

        p.plot(ax2, {})
        assert ax2.get_xlabel() == "x"
        assert ax2.get_ylabel() == "y"
        assert not ax2.yaxis.label.get_visible()

    def test_lineplot_axes(self, wide_df):

        f1, ax1 = plt.subplots()
        f2, ax2 = plt.subplots()

        ax = lineplot(data=wide_df)
        assert ax is ax2

        ax = lineplot(data=wide_df, ax=ax1)
        assert ax is ax1

    def test_lineplot_smoke(
        self,
        wide_df, wide_array,
        wide_list_of_series, wide_list_of_arrays, wide_list_of_lists,
        flat_array, flat_series, flat_list,
        long_df, missing_df
    ):

        f, ax = plt.subplots()

        lineplot(x=[], y=[])
        ax.clear()

        lineplot(data=wide_df)
        ax.clear()

        lineplot(data=wide_array)
        ax.clear()

        lineplot(data=wide_list_of_series)
        ax.clear()

        lineplot(data=wide_list_of_arrays)
        ax.clear()

        lineplot(data=wide_list_of_lists)
        ax.clear()

        lineplot(data=flat_series)
        ax.clear()

        lineplot(data=flat_array)
        ax.clear()

        lineplot(data=flat_list)
        ax.clear()

        lineplot(x="x", y="y", data=long_df)
        ax.clear()

        lineplot(x=long_df.x, y=long_df.y)
        ax.clear()

        lineplot(x=long_df.x, y="y", data=long_df)
        ax.clear()

        lineplot(x="x", y=long_df.y.values, data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", style="a", data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", style="b", data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", style="a", data=missing_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", style="b", data=missing_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", size="a", data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", size="s", data=long_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", size="a", data=missing_df)
        ax.clear()

        lineplot(x="x", y="y", hue="a", size="s", data=missing_df)
        ax.clear()


class TestScatterPlotter(Helpers):

    def test_legend_data(self, long_df):

        m = mpl.markers.MarkerStyle("o")
        default_mark = m.get_path().transformed(m.get_transform())

        m = mpl.markers.MarkerStyle("")
        null_mark = m.get_path().transformed(m.get_transform())

        f, ax = plt.subplots()

        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert handles == []

        # --

        ax.clear()
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_facecolors()[0] for h in handles]
        expected_colors = ["w"] + p._hue_map(p._hue_map.levels)
        assert labels == ["a"] + p._hue_map.levels
        assert self.colors_equal(colors, expected_colors)

        # --

        ax.clear()
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="a"),
            markers=True,
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_facecolors()[0] for h in handles]
        expected_colors = ["w"] + p._hue_map(p._hue_map.levels)
        paths = [h.get_paths()[0] for h in handles]
        expected_paths = [null_mark] + [p.paths[l] for l in p.style_levels]
        assert labels == ["a"] + p._hue_map.levels == ["a"] + p.style_levels
        assert self.colors_equal(colors, expected_colors)
        assert self.paths_equal(paths, expected_paths)

        # --

        ax.clear()
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="b"),
            markers=True,
            legend="full",
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_facecolors()[0] for h in handles]
        paths = [h.get_paths()[0] for h in handles]
        expected_colors = (
            ["w"] + p._hue_map(p._hue_map.levels)
            + ["w"] + [".2" for _ in p.style_levels]
        )
        expected_paths = (
            [null_mark] + [default_mark for _ in p._hue_map.levels]
            + [null_mark] + [p.paths[l] for l in p.style_levels]
        )
        assert labels == ["a"] + p._hue_map.levels + ["b"] + p.style_levels
        assert self.colors_equal(colors, expected_colors)
        assert self.paths_equal(paths, expected_paths)

        # --

        ax.clear()
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", size="a"),
            legend="full"
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        colors = [h.get_facecolors()[0] for h in handles]
        expected_colors = ["w"] + p._hue_map(p._hue_map.levels)
        sizes = [h.get_sizes()[0] for h in handles]
        expected_sizes = [0] + [p.sizes[l] for l in p.size_levels]
        assert labels == ["a"] + p._hue_map.levels == ["a"] + p.size_levels
        assert self.colors_equal(colors, expected_colors)
        assert sizes == expected_sizes

        # --

        ax.clear()
        sizes_list = [10, 100, 200]
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="s"),
            legend="full",
            sizes=sizes_list,
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        sizes = [h.get_sizes()[0] for h in handles]
        expected_sizes = [0] + [p.sizes[l] for l in p.size_levels]
        assert labels == ["s"] + [str(l) for l in p.size_levels]
        assert sizes == expected_sizes

        # --

        ax.clear()
        sizes_dict = {2: 10, 4: 100, 8: 200}
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="s"),
            sizes=sizes_dict,
            legend="full"
        )
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        sizes = [h.get_sizes()[0] for h in handles]
        expected_sizes = [0] + [p.sizes[l] for l in p.size_levels]
        assert labels == ["s"] + [str(l) for l in p.size_levels]
        assert sizes == expected_sizes

        # --

        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)

        p = _ScatterPlotter(
            variables=dict(x=x, y=y, hue=z),
        )

        ax.clear()
        p.legend = "full"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert labels == [str(l) for l in p._hue_map.levels]

        ax.clear()
        p.legend = "brief"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 4

        p = _ScatterPlotter(
            variables=dict(x=x, y=y, size=z),
        )

        ax.clear()
        p.legend = "full"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert labels == [str(l) for l in p.size_levels]

        ax.clear()
        p.legend = "brief"
        p.add_legend_data(ax)
        handles, labels = ax.get_legend_handles_labels()
        assert len(labels) == 4

        ax.clear()
        p.legend = "bad_value"
        with pytest.raises(ValueError):
            p.add_legend_data(ax)

    def test_plot(self, long_df, repeated_df):

        f, ax = plt.subplots()

        p = _ScatterPlotter(data=long_df, variables=dict(x="x", y="y"))

        p.plot(ax, {})
        points = ax.collections[0]
        assert_array_equal(points.get_offsets(), long_df[["x", "y"]].values)

        ax.clear()
        p.plot(ax, {"color": "k", "label": "test"})
        points = ax.collections[0]
        assert self.colors_equal(points.get_facecolor(), "k")
        assert points.get_label() == "test"

        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", hue="a")
        )

        ax.clear()
        p.plot(ax, {})
        points = ax.collections[0]
        expected_colors = p._hue_map(p.plot_data["hue"])
        assert self.colors_equal(points.get_facecolors(), expected_colors)

        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="c"),
            markers=["+", "x"]
        )

        ax.clear()
        color = (1, .3, .8)
        p.plot(ax, {"color": color})
        points = ax.collections[0]
        assert self.colors_equal(points.get_edgecolors(), [color])

        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", size="a"),
        )

        ax.clear()
        p.plot(ax, {})
        points = ax.collections[0]
        expected_sizes = [p.size_lookup(k) for k in p.plot_data["size"]]
        assert_array_equal(points.get_sizes(), expected_sizes)

        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="a"),
            markers=True,
        )

        ax.clear()
        p.plot(ax, {})
        expected_colors = p._hue_map(p.plot_data["hue"])
        expected_paths = [p.paths[k] for k in p.plot_data["style"]]
        assert self.colors_equal(points.get_facecolors(), expected_colors)
        assert self.paths_equal(points.get_paths(), expected_paths)

        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="b"),
            markers=True,
        )

        ax.clear()
        p.plot(ax, {})
        expected_colors = p._hue_map(p.plot_data["hue"])
        expected_paths = [p.paths[k] for k in p.plot_data["style"]]
        assert self.colors_equal(points.get_facecolors(), expected_colors)
        assert self.paths_equal(points.get_paths(), expected_paths)

        x_str = long_df["x"].astype(str)
        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", hue=x_str),
        )
        ax.clear()
        p.plot(ax, {})

        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", size=x_str),
        )
        ax.clear()
        p.plot(ax, {})

    def test_axis_labels(self, long_df):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        p = _ScatterPlotter(data=long_df, variables=dict(x="x", y="y"))

        p.plot(ax1, {})
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "y"

        p.plot(ax2, {})
        assert ax2.get_xlabel() == "x"
        assert ax2.get_ylabel() == "y"
        assert not ax2.yaxis.label.get_visible()

    def test_scatterplot_axes(self, wide_df):

        f1, ax1 = plt.subplots()
        f2, ax2 = plt.subplots()

        ax = scatterplot(data=wide_df)
        assert ax is ax2

        ax = scatterplot(data=wide_df, ax=ax1)
        assert ax is ax1

    def test_literal_attribute_vectors(self):

        f, ax = plt.subplots()

        x = y = [1, 2, 3]
        s = [5, 10, 15]
        c = [(1, 1, 0, 1), (1, 0, 1, .5), (.5, 1, 0, 1)]

        scatterplot(x=x, y=y, c=c, s=s, ax=ax)

        points, = ax.collections

        assert_array_equal(points.get_sizes().squeeze(), s)
        assert_array_equal(points.get_facecolors(), c)

    def test_linewidths(self, long_df):

        f, ax = plt.subplots()

        scatterplot(data=long_df, x="x", y="y", s=10)
        scatterplot(data=long_df, x="x", y="y", s=20)
        points1, points2 = ax.collections
        assert (
            points1.get_linewidths().item() < points2.get_linewidths().item()
        )

        ax.clear()
        scatterplot(data=long_df, x="x", y="y", s=long_df["x"])
        scatterplot(data=long_df, x="x", y="y", s=long_df["x"] * 2)
        points1, points2 = ax.collections
        assert (
            points1.get_linewidths().item() < points2.get_linewidths().item()
        )

        ax.clear()
        scatterplot(data=long_df, x="x", y="y", size=long_df["x"])
        scatterplot(data=long_df, x="x", y="y", size=long_df["x"] * 2)
        points1, points2, *_ = ax.collections
        assert (
            points1.get_linewidths().item() < points2.get_linewidths().item()
        )

        ax.clear()
        lw = 2
        scatterplot(data=long_df, x="x", y="y", linewidth=lw)
        assert ax.collections[0].get_linewidths().item() == lw

    def test_scatterplot_smoke(
        self,
        wide_df, wide_array,
        flat_series, flat_array, flat_list,
        wide_list_of_series, wide_list_of_arrays, wide_list_of_lists,
        long_df, missing_df
    ):

        f, ax = plt.subplots()

        scatterplot(x=[], y=[])
        ax.clear()

        scatterplot(data=wide_df)
        ax.clear()

        scatterplot(data=wide_array)
        ax.clear()

        scatterplot(data=wide_list_of_series)
        ax.clear()

        scatterplot(data=wide_list_of_arrays)
        ax.clear()

        scatterplot(data=wide_list_of_lists)
        ax.clear()

        scatterplot(data=flat_series)
        ax.clear()

        scatterplot(data=flat_array)
        ax.clear()

        scatterplot(data=flat_list)
        ax.clear()

        scatterplot(x="x", y="y", data=long_df)
        ax.clear()

        scatterplot(x=long_df.x, y=long_df.y)
        ax.clear()

        scatterplot(x=long_df.x, y="y", data=long_df)
        ax.clear()

        scatterplot(x="x", y=long_df.y.values, data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="b", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="a", data=missing_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", style="b", data=missing_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="a", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="s", data=long_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="a", data=missing_df)
        ax.clear()

        scatterplot(x="x", y="y", hue="a", size="s", data=missing_df)
        ax.clear()
