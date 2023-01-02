
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.order import Perc
from seaborn.utils import _version_predates


class Fixtures:

    @pytest.fixture
    def df(self, rng):
        return pd.DataFrame(dict(x="", y=rng.normal(size=30)))

    def get_groupby(self, df, orient):
        # TODO note, copied from aggregation
        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)


class TestPerc(Fixtures):

    def test_int_k(self, df):

        ori = "x"
        gb = self.get_groupby(df, ori)
        res = Perc(3)(df, gb, ori, {})
        percentiles = [0, 50, 100]
        assert_array_equal(res["percentile"], percentiles)
        assert_array_equal(res["y"], np.percentile(df["y"], percentiles))

    def test_list_k(self, df):

        ori = "x"
        gb = self.get_groupby(df, ori)
        percentiles = [0, 20, 100]
        res = Perc(k=percentiles)(df, gb, ori, {})
        assert_array_equal(res["percentile"], percentiles)
        assert_array_equal(res["y"], np.percentile(df["y"], percentiles))

    def test_orientation(self, df):

        df = df.rename(columns={"x": "y", "y": "x"})
        ori = "y"
        gb = self.get_groupby(df, ori)
        res = Perc(k=3)(df, gb, ori, {})
        assert_array_equal(res["x"], np.percentile(df["x"], [0, 50, 100]))

    def test_method(self, df):

        ori = "x"
        gb = self.get_groupby(df, ori)
        method = "nearest"
        res = Perc(k=5, method=method)(df, gb, ori, {})
        percentiles = [0, 25, 50, 75, 100]
        if _version_predates(np, "1.22.0"):
            expected = np.percentile(df["y"], percentiles, interpolation=method)
        else:
            expected = np.percentile(df["y"], percentiles, method=method)
        assert_array_equal(res["y"], expected)

    def test_grouped(self, df, rng):

        ori = "x"
        df = df.assign(x=rng.choice(["a", "b", "c"], len(df)))
        gb = self.get_groupby(df, ori)
        k = [10, 90]
        res = Perc(k)(df, gb, ori, {})
        for x, res_x in res.groupby("x"):
            assert_array_equal(res_x["percentile"], k)
            expected = np.percentile(df.loc[df["x"] == x, "y"], k)
            assert_array_equal(res_x["y"], expected)

    def test_with_na(self, df):

        ori = "x"
        df.loc[:5, "y"] = np.nan
        gb = self.get_groupby(df, ori)
        k = [10, 90]
        res = Perc(k)(df, gb, ori, {})
        expected = np.percentile(df["y"].dropna(), k)
        assert_array_equal(res["y"], expected)
