import functools
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._core.data import PlotData


assert_vector_equal = functools.partial(assert_series_equal, check_names=False)


class TestPlotData:

    @pytest.fixture
    def long_variables(self):
        variables = dict(x="x", y="y", color="a", size="z", style="s_cat")
        return variables

    def test_named_vectors(self, long_df, long_variables):

        p = PlotData(long_df, long_variables)
        assert p.source_data is long_df
        assert p.source_vars is long_variables
        for key, val in long_variables.items():
            assert p.names[key] == val
            assert_vector_equal(p.frame[key], long_df[val])

    def test_named_and_given_vectors(self, long_df, long_variables):

        long_variables["y"] = long_df["b"]
        long_variables["size"] = long_df["z"].to_numpy()

        p = PlotData(long_df, long_variables)

        assert_vector_equal(p.frame["color"], long_df[long_variables["color"]])
        assert_vector_equal(p.frame["y"], long_df["b"])
        assert_vector_equal(p.frame["size"], long_df["z"])

        assert p.names["color"] == long_variables["color"]
        assert p.names["y"] == "b"
        assert p.names["size"] is None

        assert p.ids["color"] == long_variables["color"]
        assert p.ids["y"] == "b"
        assert p.ids["size"] == id(long_variables["size"])

    def test_index_as_variable(self, long_df, long_variables):

        index = pd.Index(np.arange(len(long_df)) * 2 + 10, name="i", dtype=int)
        long_variables["x"] = "i"
        p = PlotData(long_df.set_index(index), long_variables)

        assert p.names["x"] == p.ids["x"] == "i"
        assert_vector_equal(p.frame["x"], pd.Series(index, index))

    def test_multiindex_as_variables(self, long_df, long_variables):

        index_i = pd.Index(np.arange(len(long_df)) * 2 + 10, name="i", dtype=int)
        index_j = pd.Index(np.arange(len(long_df)) * 3 + 5, name="j", dtype=int)
        index = pd.MultiIndex.from_arrays([index_i, index_j])
        long_variables.update({"x": "i", "y": "j"})

        p = PlotData(long_df.set_index(index), long_variables)
        assert_vector_equal(p.frame["x"], pd.Series(index_i, index))
        assert_vector_equal(p.frame["y"], pd.Series(index_j, index))

    def test_int_as_variable_key(self, rng):

        df = pd.DataFrame(rng.uniform(size=(10, 3)))

        var = "x"
        key = 2

        p = PlotData(df, {var: key})
        assert_vector_equal(p.frame[var], df[key])
        assert p.names[var] == p.ids[var] == str(key)

    def test_int_as_variable_value(self, long_df):

        p = PlotData(long_df, {"x": 0, "y": "y"})
        assert (p.frame["x"] == 0).all()
        assert p.names["x"] is None
        assert p.ids["x"] == id(0)

    def test_tuple_as_variable_key(self, rng):

        cols = pd.MultiIndex.from_product([("a", "b", "c"), ("x", "y")])
        df = pd.DataFrame(rng.uniform(size=(10, 6)), columns=cols)

        var = "color"
        key = ("b", "y")
        p = PlotData(df, {var: key})
        assert_vector_equal(p.frame[var], df[key])
        assert p.names[var] == p.ids[var] == str(key)

    def test_dict_as_data(self, long_dict, long_variables):

        p = PlotData(long_dict, long_variables)
        assert p.source_data is long_dict
        for key, val in long_variables.items():
            assert_vector_equal(p.frame[key], pd.Series(long_dict[val]))

    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    def test_vectors_various_types(self, long_df, long_variables, vector_type):

        variables = {key: long_df[val] for key, val in long_variables.items()}
        if vector_type == "numpy":
            variables = {key: val.to_numpy() for key, val in variables.items()}
        elif vector_type == "list":
            variables = {key: val.to_list() for key, val in variables.items()}

        p = PlotData(None, variables)

        assert list(p.names) == list(long_variables)
        if vector_type == "series":
            assert p.source_vars is variables
            assert p.names == p.ids == {key: val.name for key, val in variables.items()}
        else:
            assert p.names == {key: None for key in variables}
            assert p.ids == {key: id(val) for key, val in variables.items()}

        for key, val in long_variables.items():
            if vector_type == "series":
                assert_vector_equal(p.frame[key], long_df[val])
            else:
                assert_array_equal(p.frame[key], long_df[val])

    def test_none_as_variable_value(self, long_df):

        p = PlotData(long_df, {"x": "z", "y": None})
        assert list(p.frame.columns) == ["x"]
        assert p.names == p.ids == {"x": "z"}

    def test_frame_and_vector_mismatched_lengths(self, long_df):

        vector = np.arange(len(long_df) * 2)
        with pytest.raises(ValueError):
            PlotData(long_df, {"x": "x", "y": vector})

    @pytest.mark.parametrize(
        "arg", [[], np.array([]), pd.DataFrame()],
    )
    def test_empty_data_input(self, arg):

        p = PlotData(arg, {})
        assert p.frame.empty
        assert not p.names

        if not isinstance(arg, pd.DataFrame):
            p = PlotData(None, dict(x=arg, y=arg))
            assert p.frame.empty
            assert not p.names

    def test_index_alignment_series_to_dataframe(self):

        x = [1, 2, 3]
        x_index = pd.Index(x, dtype=int)

        y_values = [3, 4, 5]
        y_index = pd.Index(y_values, dtype=int)
        y = pd.Series(y_values, y_index, name="y")

        data = pd.DataFrame(dict(x=x), index=x_index)

        p = PlotData(data, {"x": "x", "y": y})

        x_col_expected = pd.Series([1, 2, 3, np.nan, np.nan], np.arange(1, 6))
        y_col_expected = pd.Series([np.nan, np.nan, 3, 4, 5], np.arange(1, 6))
        assert_vector_equal(p.frame["x"], x_col_expected)
        assert_vector_equal(p.frame["y"], y_col_expected)

    def test_index_alignment_between_series(self):

        x_index = [1, 2, 3]
        x_values = [10, 20, 30]
        x = pd.Series(x_values, x_index, name="x")

        y_index = [3, 4, 5]
        y_values = [300, 400, 500]
        y = pd.Series(y_values, y_index, name="y")

        p = PlotData(None, {"x": x, "y": y})

        idx_expected = [1, 2, 3, 4, 5]
        x_col_expected = pd.Series([10, 20, 30, np.nan, np.nan], idx_expected)
        y_col_expected = pd.Series([np.nan, np.nan, 300, 400, 500], idx_expected)
        assert_vector_equal(p.frame["x"], x_col_expected)
        assert_vector_equal(p.frame["y"], y_col_expected)

    def test_key_not_in_data_raises(self, long_df):

        var = "x"
        key = "what"
        msg = f"Could not interpret value `{key}` for `{var}`. An entry with this name"
        with pytest.raises(ValueError, match=msg):
            PlotData(long_df, {var: key})

    def test_key_with_no_data_raises(self):

        var = "x"
        key = "what"
        msg = f"Could not interpret value `{key}` for `{var}`. Value is a string,"
        with pytest.raises(ValueError, match=msg):
            PlotData(None, {var: key})

    def test_data_vector_different_lengths_raises(self, long_df):

        vector = np.arange(len(long_df) - 5)
        msg = "Length of ndarray vectors must match length of `data`"
        with pytest.raises(ValueError, match=msg):
            PlotData(long_df, {"y": vector})

    def test_undefined_variables_raise(self, long_df):

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="y", color="not_in_df"))

    def test_contains_operation(self, long_df):

        p = PlotData(long_df, {"x": "y", "color": long_df["a"]})
        assert "x" in p
        assert "y" not in p
        assert "color" in p

    def test_join_add_variable(self, long_df):

        v1 = {"x": "x", "y": "f"}
        v2 = {"color": "a"}

        p1 = PlotData(long_df, v1)
        p2 = p1.join(None, v2)

        for var, key in dict(**v1, **v2).items():
            assert var in p2
            assert p2.names[var] == key
            assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_replace_variable(self, long_df):

        v1 = {"x": "x", "y": "y"}
        v2 = {"y": "s"}

        p1 = PlotData(long_df, v1)
        p2 = p1.join(None, v2)

        variables = v1.copy()
        variables.update(v2)

        for var, key in variables.items():
            assert var in p2
            assert p2.names[var] == key
            assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_remove_variable(self, long_df):

        variables = {"x": "x", "y": "f"}
        drop_var = "y"

        p1 = PlotData(long_df, variables)
        p2 = p1.join(None, {drop_var: None})

        assert drop_var in p1
        assert drop_var not in p2
        assert drop_var not in p2.frame
        assert drop_var not in p2.names

    def test_join_all_operations(self, long_df):

        v1 = {"x": "x", "y": "y", "color": "a"}
        v2 = {"y": "s", "size": "s", "color": None}

        p1 = PlotData(long_df, v1)
        p2 = p1.join(None, v2)

        for var, key in v2.items():
            if key is None:
                assert var not in p2
            else:
                assert p2.names[var] == key
                assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_all_operations_same_data(self, long_df):

        v1 = {"x": "x", "y": "y", "color": "a"}
        v2 = {"y": "s", "size": "s", "color": None}

        p1 = PlotData(long_df, v1)
        p2 = p1.join(long_df, v2)

        for var, key in v2.items():
            if key is None:
                assert var not in p2
            else:
                assert p2.names[var] == key
                assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_add_variable_new_data(self, long_df):

        d1 = long_df[["x", "y"]]
        d2 = long_df[["a", "s"]]

        v1 = {"x": "x", "y": "y"}
        v2 = {"color": "a"}

        p1 = PlotData(d1, v1)
        p2 = p1.join(d2, v2)

        for var, key in dict(**v1, **v2).items():
            assert p2.names[var] == key
            assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_replace_variable_new_data(self, long_df):

        d1 = long_df[["x", "y"]]
        d2 = long_df[["a", "s"]]

        v1 = {"x": "x", "y": "y"}
        v2 = {"x": "a"}

        p1 = PlotData(d1, v1)
        p2 = p1.join(d2, v2)

        variables = v1.copy()
        variables.update(v2)

        for var, key in variables.items():
            assert p2.names[var] == key
            assert_vector_equal(p2.frame[var], long_df[key])

    def test_join_add_variable_different_index(self, long_df):

        d1 = long_df.iloc[:70]
        d2 = long_df.iloc[30:]

        v1 = {"x": "a"}
        v2 = {"y": "z"}

        p1 = PlotData(d1, v1)
        p2 = p1.join(d2, v2)

        (var1, key1), = v1.items()
        (var2, key2), = v2.items()

        assert_vector_equal(p2.frame.loc[d1.index, var1], d1[key1])
        assert_vector_equal(p2.frame.loc[d2.index, var2], d2[key2])

        assert p2.frame.loc[d2.index.difference(d1.index), var1].isna().all()
        assert p2.frame.loc[d1.index.difference(d2.index), var2].isna().all()

    def test_join_replace_variable_different_index(self, long_df):

        d1 = long_df.iloc[:70]
        d2 = long_df.iloc[30:]

        var = "x"
        k1, k2 = "a", "z"
        v1 = {var: k1}
        v2 = {var: k2}

        p1 = PlotData(d1, v1)
        p2 = p1.join(d2, v2)

        (var1, key1), = v1.items()
        (var2, key2), = v2.items()

        assert_vector_equal(p2.frame.loc[d2.index, var], d2[k2])
        assert p2.frame.loc[d1.index.difference(d2.index), var].isna().all()

    def test_join_subset_data_inherit_variables(self, long_df):

        sub_df = long_df[long_df["a"] == "b"]

        var = "y"
        p1 = PlotData(long_df, {var: var})
        p2 = p1.join(sub_df, None)

        assert_vector_equal(p2.frame.loc[sub_df.index, var], sub_df[var])
        assert p2.frame.loc[long_df.index.difference(sub_df.index), var].isna().all()

    def test_join_multiple_inherits_from_orig(self, rng):

        d1 = pd.DataFrame(dict(a=rng.normal(0, 1, 100), b=rng.normal(0, 1, 100)))
        d2 = pd.DataFrame(dict(a=rng.normal(0, 1, 100)))

        p = PlotData(d1, {"x": "a"}).join(d2, {"y": "a"}).join(None, {"y": "a"})
        assert_vector_equal(p.frame["x"], d1["a"])
        assert_vector_equal(p.frame["y"], d1["a"])
