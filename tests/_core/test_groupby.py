
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.groupby import GroupBy


@pytest.fixture
def df():

    return pd.DataFrame(
        columns=["a", "b", "x", "y"],
        data=[
            ["a", "g", 1, .2],
            ["b", "h", 3, .5],
            ["a", "f", 2, .8],
            ["a", "h", 1, .3],
            ["b", "f", 2, .4],
        ]
    )


def test_init_from_list():
    g = GroupBy(["a", "c", "b"])
    assert g.order == {"a": None, "c": None, "b": None}


def test_init_from_dict():
    order = {"a": [3, 2, 1], "c": None, "b": ["x", "y", "z"]}
    g = GroupBy(order)
    assert g.order == order


def test_init_requires_order():

    with pytest.raises(ValueError, match="GroupBy requires at least one"):
        GroupBy([])


def test_agg_one_grouper(df):

    res = GroupBy(["a"]).agg(df, {"y": "max"})
    assert_array_equal(res.index, [0, 1])
    assert_array_equal(res.columns, ["a", "y"])
    assert_array_equal(res["a"], ["a", "b"])
    assert_array_equal(res["y"], [.8, .5])


def test_agg_two_groupers(df):

    res = GroupBy(["a", "x"]).agg(df, {"y": "min"})
    assert_array_equal(res.index, [0, 1, 2, 3, 4, 5])
    assert_array_equal(res.columns, ["a", "x", "y"])
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b", "b"])
    assert_array_equal(res["x"], [1, 2, 3, 1, 2, 3])
    assert_array_equal(res["y"], [.2, .8, np.nan, np.nan, .4, .5])


def test_agg_two_groupers_ordered(df):

    order = {"b": ["h", "g", "f"], "x": [3, 2, 1]}
    res = GroupBy(order).agg(df, {"a": "min", "y": lambda x: x.iloc[0]})
    assert_array_equal(res.index, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert_array_equal(res.columns, ["a", "b", "x", "y"])
    assert_array_equal(res["b"], ["h", "h", "h", "g", "g", "g", "f", "f", "f"])
    assert_array_equal(res["x"], [3, 2, 1, 3, 2, 1, 3, 2, 1])

    T, F = True, False
    assert_array_equal(res["a"].isna(), [F, T, F, T, T, F, T, F, T])
    assert_array_equal(res["a"].dropna(), ["b", "a", "a", "a"])
    assert_array_equal(res["y"].dropna(), [.5, .3, .2, .8])


def test_apply_no_grouper(df):

    df = df[["x", "y"]]
    res = GroupBy(["a"]).apply(df, lambda x: x.sort_values("x"))
    assert_array_equal(res.columns, ["x", "y"])
    assert_array_equal(res["x"], df["x"].sort_values())
    assert_array_equal(res["y"], df.loc[np.argsort(df["x"]), "y"])


def test_apply_one_grouper(df):

    res = GroupBy(["a"]).apply(df, lambda x: x.sort_values("x"))
    assert_array_equal(res.index, [0, 1, 2, 3, 4])
    assert_array_equal(res.columns, ["a", "b", "x", "y"])
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b"])
    assert_array_equal(res["b"], ["g", "h", "f", "f", "h"])
    assert_array_equal(res["x"], [1, 1, 2, 2, 3])


def test_apply_mutate_columns(df):

    xx = np.arange(0, 5)
    hats = []

    def polyfit(df):
        fit = np.polyfit(df["x"], df["y"], 1)
        hat = np.polyval(fit, xx)
        hats.append(hat)
        return pd.DataFrame(dict(x=xx, y=hat))

    res = GroupBy(["a"]).apply(df, polyfit)
    assert_array_equal(res.index, np.arange(xx.size * 2))
    assert_array_equal(res.columns, ["a", "x", "y"])
    assert_array_equal(res["a"], ["a"] * xx.size + ["b"] * xx.size)
    assert_array_equal(res["x"], xx.tolist() + xx.tolist())
    assert_array_equal(res["y"], np.concatenate(hats))


def test_apply_replace_columns(df):

    def add_sorted_cumsum(df):

        x = df["x"].sort_values()
        z = df.loc[x.index, "y"].cumsum()
        return pd.DataFrame(dict(x=x.values, z=z.values))

    res = GroupBy(["a"]).apply(df, add_sorted_cumsum)
    assert_array_equal(res.index, df.index)
    assert_array_equal(res.columns, ["a", "x", "z"])
    assert_array_equal(res["a"], ["a", "a", "a", "b", "b"])
    assert_array_equal(res["x"], [1, 1, 2, 2, 3])
    assert_array_equal(res["z"], [.2, .5, 1.3, .4, .9])
