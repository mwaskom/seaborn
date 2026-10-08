
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.regression import PolyFit


class TestPolyFit:

    @pytest.fixture
    def df(self, rng):

        n = 100
        return pd.DataFrame(dict(
            x=rng.normal(0, 1, n),
            y=rng.normal(0, 1, n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def test_no_grouper(self, df):

        groupby = GroupBy(["group"])
        res = PolyFit(order=1, gridsize=100)(df[["x", "y"]], groupby, "x", {})

        assert_array_equal(res.columns, ["x", "y"])

        grid = np.linspace(df["x"].min(), df["x"].max(), 100)
        assert_array_equal(res["x"], grid)
        assert_array_almost_equal(
            res["y"].diff().diff().dropna(), np.zeros(grid.size - 2)
        )

    def test_one_grouper(self, df):

        groupby = GroupBy(["group"])
        gridsize = 50
        res = PolyFit(gridsize=gridsize)(df, groupby, "x", {})

        assert res.columns.to_list() == ["x", "y", "group"]

        ngroups = df["group"].nunique()
        assert_array_equal(res.index, np.arange(ngroups * gridsize))

        for _, part in res.groupby("group"):
            grid = np.linspace(part["x"].min(), part["x"].max(), gridsize)
            assert_array_equal(part["x"], grid)
            assert part["y"].diff().diff().dropna().abs().gt(0).all()

    def test_missing_data(self, df):

        groupby = GroupBy(["group"])
        df.iloc[5:10] = np.nan
        res1 = PolyFit()(df[["x", "y"]], groupby, "x", {})
        res2 = PolyFit()(df[["x", "y"]].dropna(), groupby, "x", {})
        assert_frame_equal(res1, res2)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_unique_x_order_boundary(self, order):

        groupby = GroupBy(["group"])
        gridsize = 10
        x = np.arange(order + 1, dtype=float)
        df = pd.DataFrame(dict(x=x, y=x ** order))

        # order + 1 unique x values are enough to fit
        res = PolyFit(order=order, gridsize=gridsize)(df, groupby, "x", {})
        grid = np.linspace(x.min(), x.max(), gridsize)
        assert_array_almost_equal(res["x"], grid)
        assert_array_almost_equal(res["y"], grid ** order)

        # one fewer is too few, so there is no fit
        res = PolyFit(order=order, gridsize=gridsize)(df.iloc[:order], groupby, "x", {})
        assert res.empty
