import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.density import KDE, _no_scipy


class TestKDE:

    @pytest.fixture
    def df(self, rng):

        n = 100
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),
            y=rng.normal(size=n),
            color=rng.choice(["a", "b", "c"], n),
            alpha=rng.choice(["x", "y"], n),
        ))

    def get_groupby(self, df, orient):

        cols = [c for c in df if c != orient]
        return GroupBy([*cols, "group"])

    def integrate(self, y, x):
        y = np.asarray(y)
        x = np.asarray(x)
        dx = np.diff(x)
        return (dx * y[:-1] + dx * y[1:]).sum() / 2

    @pytest.mark.parametrize("ori", ["x", "y"])
    def test_columns(self, df, ori):

        df = df[[ori, "alpha"]]
        gb = self.get_groupby(df, ori)
        res = KDE()(df, gb, ori, {})
        other = {"x": "y", "y": "x"}[ori]
        expected = [ori, "alpha", "weight", "density", "group_weight", other]
        assert list(res.columns) == expected

    @pytest.mark.parametrize("gridsize", [20, 30, None])
    def test_gridsize(self, df, gridsize):

        ori = "y"
        df = df[[ori]]
        gb = self.get_groupby(df, ori)
        res = KDE(gridsize=gridsize)(df, gb, ori, {})
        if gridsize is None:
            assert_array_equal(res[ori], df[ori])
        else:
            assert len(res) == gridsize

    @pytest.mark.parametrize("cut", [1, 2])
    def test_cut(self, df, cut):

        ori = "y"
        df = df[[ori]]
        gb = self.get_groupby(df, ori)
        res = KDE(cut=cut, bw_method=1)(df, gb, ori, {})

        vals = df[ori]
        bw = vals.std()
        assert res[ori].min() == pytest.approx(vals.min() - bw * cut, abs=1e-2)
        assert res[ori].max() == pytest.approx(vals.max() + bw * cut, abs=1e-2)

    @pytest.mark.parametrize("common_grid", [True, False])
    def test_common_grid(self, df, common_grid):

        ori = "y"
        df = df[[ori, "alpha"]]
        gb = self.get_groupby(df, ori)
        res = KDE(common_grid=common_grid)(df, gb, ori, {})

        vals = df["alpha"].unique()
        a = res.loc[res["alpha"] == vals[0], ori].to_numpy()
        b = res.loc[res["alpha"] == vals[1], ori].to_numpy()
        if common_grid:
            assert_array_equal(a, b)
        else:
            assert np.not_equal(a, b).all()

    @pytest.mark.parametrize("common_norm", [True, False])
    def test_common_norm(self, df, common_norm):

        ori = "y"
        df = df[[ori, "alpha"]]
        gb = self.get_groupby(df, ori)
        res = KDE(common_norm=common_norm)(df, gb, ori, {})

        areas = (
            res.groupby("alpha")
            .apply(lambda x: self.integrate(x["density"], x[ori]))
        )

        if common_norm:
            assert areas.sum() == pytest.approx(1, abs=1e-3)
        else:
            assert_array_almost_equal(areas, [1, 1], decimal=3)

    def test_bw_adjust(self, df):

        ori = "y"
        df = df[[ori]]
        gb = self.get_groupby(df, ori)
        res1 = KDE(bw_adjust=0.5)(df, gb, ori, {})
        res2 = KDE(bw_adjust=2.0)(df, gb, ori, {})

        mad1 = res1["density"].diff().abs().mean()
        mad2 = res2["density"].diff().abs().mean()
        assert mad1 > mad2

    def test_bw_method_scalar(self, df):

        ori = "y"
        df = df[[ori]]
        gb = self.get_groupby(df, ori)
        res1 = KDE(bw_method=0.5)(df, gb, ori, {})
        res2 = KDE(bw_method=2.0)(df, gb, ori, {})

        mad1 = res1["density"].diff().abs().mean()
        mad2 = res2["density"].diff().abs().mean()
        assert mad1 > mad2

    @pytest.mark.skipif(_no_scipy, reason="KDE.cumulative requires scipy")
    @pytest.mark.parametrize("common_norm", [True, False])
    def test_cumulative(self, df, common_norm):

        ori = "y"
        df = df[[ori, "alpha"]]
        gb = self.get_groupby(df, ori)

        res = KDE(cumulative=True, common_norm=common_norm)(df, gb, ori, {})

        for _, group_res in res.groupby("alpha"):
            assert (group_res["density"].diff().dropna() >= 0).all()
            if not common_norm:
                assert group_res["density"].max() == pytest.approx(1, abs=1e-3)

    def test_cumulative_requires_scipy(self):

        if _no_scipy:
            err = "Cumulative KDE evaluation requires scipy"
            with pytest.raises(RuntimeError, match=err):
                KDE(cumulative=True)
