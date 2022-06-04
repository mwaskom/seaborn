
import pandas as pd

import pytest
from pandas.testing import assert_frame_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.aggregation import Agg


class TestAgg:

    @pytest.fixture
    def df(self, rng):

        n = 30
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),
            y=rng.normal(size=n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    def get_groupby(self, df, orient):

        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)

    def test_default(self, df):

        ori = "x"
        df = df[["x", "y"]]
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})

        expected = df.groupby("x", as_index=False)["y"].mean()
        assert_frame_equal(res, expected)

    def test_default_multi(self, df):

        ori = "x"
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})

        grp = ["x", "color", "group"]
        index = pd.MultiIndex.from_product(
            [sorted(df["x"].unique()), df["color"].unique(), df["group"].unique()],
            names=["x", "color", "group"]
        )
        expected = (
            df
            .groupby(grp)
            .agg("mean")
            .reindex(index=index)
            .dropna()
            .reset_index()
            .reindex(columns=df.columns)
        )
        assert_frame_equal(res, expected)

    @pytest.mark.parametrize("func", ["max", lambda x: float(len(x) % 2)])
    def test_func(self, df, func):

        ori = "x"
        df = df[["x", "y"]]
        gb = self.get_groupby(df, ori)
        res = Agg(func)(df, gb, ori, {})

        expected = df.groupby("x", as_index=False)["y"].agg(func)
        assert_frame_equal(res, expected)
