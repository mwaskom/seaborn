
from itertools import product

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.moves import Dodge, Jitter, Shift, Stack, Norm
from seaborn._core.rules import categorical_order
from seaborn._core.groupby import GroupBy

import pytest


class MoveFixtures:

    @pytest.fixture
    def df(self, rng):

        n = 50
        data = {
            "x": rng.choice([0., 1., 2., 3.], n),
            "y": rng.normal(0, 1, n),
            "grp2": rng.choice(["a", "b"], n),
            "grp3": rng.choice(["x", "y", "z"], n),
            "width": 0.8,
            "baseline": 0,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def toy_df(self):

        data = {
            "x": [0, 0, 1],
            "y": [1, 2, 3],
            "grp": ["a", "b", "b"],
            "width": .8,
            "baseline": 0,
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def toy_df_widths(self, toy_df):

        toy_df["width"] = [.8, .2, .4]
        return toy_df

    @pytest.fixture
    def toy_df_facets(self):

        data = {
            "x": [0, 0, 1, 0, 1, 2],
            "y": [1, 2, 3, 1, 2, 3],
            "grp": ["a", "b", "a", "b", "a", "b"],
            "col": ["x", "x", "x", "y", "y", "y"],
            "width": .8,
            "baseline": 0,
        }
        return pd.DataFrame(data)


class TestJitter(MoveFixtures):

    def get_groupby(self, data, orient):
        other = {"x": "y", "y": "x"}[orient]
        variables = [v for v in data if v not in [other, "width"]]
        return GroupBy(variables)

    def check_same(self, res, df, *cols):
        for col in cols:
            assert_series_equal(res[col], df[col])

    def check_pos(self, res, df, var, limit):

        assert (res[var] != df[var]).all()
        assert (res[var] < df[var] + limit / 2).all()
        assert (res[var] > df[var] - limit / 2).all()

    def test_default(self, df):

        orient = "x"
        groupby = self.get_groupby(df, orient)
        res = Jitter()(df, groupby, orient, {})
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", 0.2 * df["width"])
        assert (res["x"] - df["x"]).abs().min() > 0

    def test_width(self, df):

        width = .4
        orient = "x"
        groupby = self.get_groupby(df, orient)
        res = Jitter(width=width)(df, groupby, orient, {})
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", width * df["width"])

    def test_x(self, df):

        val = .2
        orient = "x"
        groupby = self.get_groupby(df, orient)
        res = Jitter(x=val)(df, groupby, orient, {})
        self.check_same(res, df, "y", "grp2", "width")
        self.check_pos(res, df, "x", val)

    def test_y(self, df):

        val = .2
        orient = "x"
        groupby = self.get_groupby(df, orient)
        res = Jitter(y=val)(df, groupby, orient, {})
        self.check_same(res, df, "x", "grp2", "width")
        self.check_pos(res, df, "y", val)

    def test_seed(self, df):

        kws = dict(width=.2, y=.1, seed=0)
        orient = "x"
        groupby = self.get_groupby(df, orient)
        res1 = Jitter(**kws)(df, groupby, orient, {})
        res2 = Jitter(**kws)(df, groupby, orient, {})
        for var in "xy":
            assert_series_equal(res1[var], res2[var])


class TestDodge(MoveFixtures):

    # First some very simple toy examples

    def test_default(self, toy_df):

        groupby = GroupBy(["x", "grp"])
        res = Dodge()(toy_df, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3]),
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    def test_fill(self, toy_df):

        groupby = GroupBy(["x", "grp"])
        res = Dodge(empty="fill")(toy_df, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3]),
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .8])

    def test_drop(self, toy_df):

        groupby = GroupBy(["x", "grp"])
        res = Dodge("drop")(toy_df, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    def test_gap(self, toy_df):

        groupby = GroupBy(["x", "grp"])
        res = Dodge(gap=.25)(toy_df, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        assert_array_almost_equal(res["width"], [.3, .3, .3])

    def test_widths_default(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"])
        res = Dodge()(toy_df_widths, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1.1])
        assert_array_almost_equal(res["width"], [.64, .16, .2])

    def test_widths_fill(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"])
        res = Dodge(empty="fill")(toy_df_widths, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        assert_array_almost_equal(res["width"], [.64, .16, .4])

    def test_widths_drop(self, toy_df_widths):

        groupby = GroupBy(["x", "grp"])
        res = Dodge(empty="drop")(toy_df_widths, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3])
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        assert_array_almost_equal(res["width"], [.64, .16, .2])

    def test_faceted_default(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"])
        res = Dodge()(toy_df_facets, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, .8, .2, .8, 2.2])
        assert_array_almost_equal(res["width"], [.4] * 6)

    def test_faceted_fill(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"])
        res = Dodge(empty="fill")(toy_df_facets, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1, 0, 1, 2])
        assert_array_almost_equal(res["width"], [.4, .4, .8, .8, .8, .8])

    def test_faceted_drop(self, toy_df_facets):

        groupby = GroupBy(["x", "grp", "col"])
        res = Dodge(empty="drop")(toy_df_facets, groupby, "x", {})

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        assert_array_almost_equal(res["x"], [-.2, .2, 1, 0, 1, 2])
        assert_array_almost_equal(res["width"], [.4] * 6)

    def test_orient(self, toy_df):

        df = toy_df.assign(x=toy_df["y"], y=toy_df["x"])

        groupby = GroupBy(["y", "grp"])
        res = Dodge("drop")(df, groupby, "y", {})

        assert_array_equal(res["x"], [1, 2, 3])
        assert_array_almost_equal(res["y"], [-.2, .2, 1])
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    # Now tests with slightly more complicated data

    @pytest.mark.parametrize("grp", ["grp2", "grp3"])
    def test_single_semantic(self, df, grp):

        groupby = GroupBy(["x", grp])
        res = Dodge()(df, groupby, "x", {})

        levels = categorical_order(df[grp])
        w, n = 0.8, len(levels)

        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        assert_series_equal(res["y"], df["y"])
        assert_series_equal(res["width"], df["width"] / n)

        for val, shift in zip(levels, shifts):
            rows = df[grp] == val
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)

    def test_two_semantics(self, df):

        groupby = GroupBy(["x", "grp2", "grp3"])
        res = Dodge()(df, groupby, "x", {})

        levels = categorical_order(df["grp2"]), categorical_order(df["grp3"])
        w, n = 0.8, len(levels[0]) * len(levels[1])

        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        assert_series_equal(res["y"], df["y"])
        assert_series_equal(res["width"], df["width"] / n)

        for (v2, v3), shift in zip(product(*levels), shifts):
            rows = (df["grp2"] == v2) & (df["grp3"] == v3)
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)


class TestStack(MoveFixtures):

    def test_basic(self, toy_df):

        groupby = GroupBy(["color", "group"])
        res = Stack()(toy_df, groupby, "x", {})

        assert_array_equal(res["x"], [0, 0, 1])
        assert_array_equal(res["y"], [1, 3, 3])
        assert_array_equal(res["baseline"], [0, 1, 0])

    def test_faceted(self, toy_df_facets):

        groupby = GroupBy(["color", "group"])
        res = Stack()(toy_df_facets, groupby, "x", {})

        assert_array_equal(res["x"], [0, 0, 1, 0, 1, 2])
        assert_array_equal(res["y"], [1, 3, 3, 1, 2, 3])
        assert_array_equal(res["baseline"], [0, 1, 0, 0, 0, 0])

    def test_misssing_data(self, toy_df):

        df = pd.DataFrame({
            "x": [0, 0, 0],
            "y": [2, np.nan, 1],
            "baseline": [0, 0, 0],
        })
        res = Stack()(df, None, "x", {})
        assert_array_equal(res["y"], [2, np.nan, 3])
        assert_array_equal(res["baseline"], [0, np.nan, 2])

    def test_baseline_homogeneity_check(self, toy_df):

        toy_df["baseline"] = [0, 1, 2]
        groupby = GroupBy(["color", "group"])
        move = Stack()
        err = "Stack move cannot be used when baselines"
        with pytest.raises(RuntimeError, match=err):
            move(toy_df, groupby, "x", {})


class TestShift(MoveFixtures):

    def test_default(self, toy_df):

        gb = GroupBy(["color", "group"])
        res = Shift()(toy_df, gb, "x", {})
        for col in toy_df:
            assert_series_equal(toy_df[col], res[col])

    @pytest.mark.parametrize("x,y", [(.3, 0), (0, .2), (.1, .3)])
    def test_moves(self, toy_df, x, y):

        gb = GroupBy(["color", "group"])
        res = Shift(x=x, y=y)(toy_df, gb, "x", {})
        assert_array_equal(res["x"], toy_df["x"] + x)
        assert_array_equal(res["y"], toy_df["y"] + y)


class TestNorm(MoveFixtures):

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_default_no_groups(self, df, orient):

        other = {"x": "y", "y": "x"}[orient]
        gb = GroupBy(["null"])
        res = Norm()(df, gb, orient, {})
        assert res[other].max() == pytest.approx(1)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_default_groups(self, df, orient):

        other = {"x": "y", "y": "x"}[orient]
        gb = GroupBy(["grp2"])
        res = Norm()(df, gb, orient, {})
        for _, grp in res.groupby("grp2"):
            assert grp[other].max() == pytest.approx(1)

    def test_sum(self, df):

        gb = GroupBy(["null"])
        res = Norm("sum")(df, gb, "x", {})
        assert res["y"].sum() == pytest.approx(1)

    def test_where(self, df):

        gb = GroupBy(["null"])
        res = Norm(where="x == 2")(df, gb, "x", {})
        assert res.loc[res["x"] == 2, "y"].max() == pytest.approx(1)

    def test_percent(self, df):

        gb = GroupBy(["null"])
        res = Norm(percent=True)(df, gb, "x", {})
        assert res["y"].max() == pytest.approx(100)
