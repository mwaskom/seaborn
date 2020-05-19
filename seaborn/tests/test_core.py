import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal

from ..core import (
    _VectorPlotter,
    variable_type,
    infer_orient,
    unique_dashes,
    unique_markers,
)


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

        infer_orient(cats, None, require_numeric_dv=False) == "h"
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, None)

        infer_orient(cats, None, require_numeric_dv=False) == "v"
        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(None, cats)

        assert infer_orient(nums, nums, "vert") == "v"
        assert infer_orient(nums, nums, "hori") == "h"

        assert infer_orient(cats, cats, "h", require_numeric_dv=False) == "h"
        assert infer_orient(cats, cats, "v", require_numeric_dv=False) == "v"
        assert infer_orient(cats, cats, require_numeric_dv=False) == "v"

        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(cats, cats, "v")
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, cats, "h")
        with pytest.raises(TypeError, match="Neither"):
            infer_orient(cats, cats)
