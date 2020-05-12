import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal


from ..core import _VectorPlotter


class TestVectorPlotter:

    def test_wide_df_variables(self, wide_df):

        p = _VectorPlotter()
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

    def test_wide_array_variables(self, wide_array):

        p = _VectorPlotter()
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

        p = _VectorPlotter()
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

    def test_flat_series_variables(self, flat_series):

        p = _VectorPlotter()
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

    def test_wide_list_variables(self, wide_list):

        p = _VectorPlotter()
        p.establish_variables(data=wide_list)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_list)
        chunk_size = max(len(l) for l in wide_list)

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(wide_list)
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

    def test_wide_list_of_series_variables(self, wide_list_of_series):

        p = _VectorPlotter()
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

    def test_wide_list_of_list_variables(self, wide_list_of_series):

        data = [s.tolist() for s in wide_list_of_series]

        p = _VectorPlotter()
        p.establish_variables(data=data)
        assert p.input_format == "wide"
        assert list(p.variables) == ["x", "y", "hue", "style"]

        chunks = len(wide_list_of_series)
        chunk_size = max(len(l) for l in wide_list_of_series)

        assert len(p.plot_data) == chunks * chunk_size

        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(wide_list_of_series)
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

    def test_long_df(self, long_df):

        p = _VectorPlotter()
        p.establish_variables(x="x", y="y", data=long_df)
        assert p.input_format == "long"
        assert list(p.variables) == ["x", "y"]

        assert_array_equal(p.plot_data["x"], long_df["x"])
        assert_array_equal(p.plot_data["y"], long_df["y"])
        for col in ["hue", "style", "size"]:
            assert p.plot_data[col].isnull().all()
        assert (p.variables["x"], p.variables["y"]) == ("x", "y")

        p.establish_variables(x=long_df.x, y="y", data=long_df)
        assert list(p.variables) == ["x", "y"]
        assert_array_equal(p.plot_data["x"], long_df["x"])
        assert_array_equal(p.plot_data["y"], long_df["y"])
        assert (p.variables["x"], p.variables["y"]) == ("x", "y")

        p.establish_variables(x="x", y=long_df.y, data=long_df)
        assert list(p.variables) == ["x", "y"]
        assert_array_equal(p.plot_data["x"], long_df["x"])
        assert_array_equal(p.plot_data["y"], long_df["y"])
        assert (p.variables["x"], p.variables["y"]) == ("x", "y")

        p.establish_variables(x="x", y="y", hue="a", data=long_df)
        assert list(p.variables) == ["x", "y", "hue"]
        assert_array_equal(p.plot_data["hue"], long_df["a"])
        for col in ["style", "size"]:
            assert p.plot_data[col].isnull().all()
        assert p.variables["hue"] == "a"

        p.establish_variables(x="x", y="y", hue="a", style="a", data=long_df)
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert_array_equal(p.plot_data["hue"], long_df["a"])
        assert_array_equal(p.plot_data["style"], long_df["a"])
        assert p.plot_data["size"].isnull().all()
        assert p.variables["hue"] == p.variables["style"] == "a"

        p.establish_variables(x="x", y="y", hue="a", style="b", data=long_df)
        assert list(p.variables) == ["x", "y", "hue", "style"]
        assert_array_equal(p.plot_data["hue"], long_df["a"])
        assert_array_equal(p.plot_data["style"], long_df["b"])
        assert p.plot_data["size"].isnull().all()

        p.establish_variables(x="x", y="y", size="y", data=long_df)
        assert list(p.variables) == ["x", "y", "size"]
        assert_array_equal(p.plot_data["size"], long_df["y"])
        assert p.variables["size"] == "y"

    def test_bad_input(self, long_df):

        p = _VectorPlotter()

        with pytest.raises(ValueError):
            p.establish_variables(x="not_in_df", data=long_df)

        with pytest.raises(ValueError):
            p.establish_variables(x="x", y="not_in_df", data=long_df)

        with pytest.raises(ValueError):
            p.establish_variables(x="x", y="not_in_df", data=long_df)

    def test_empty_input(self):

        p = _VectorPlotter()

        p.establish_variables(data=[])
        p.establish_variables(data=np.array([]))
        p.establish_variables(data=pd.DataFrame())
        p.establish_variables(x=[], y=[])

    def test_units(self, repeated_df):

        p = _VectorPlotter()
        p.establish_variables(x="x", y="y", units="u", data=repeated_df)
        assert_array_equal(p.plot_data["units"], repeated_df["u"])
