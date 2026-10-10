import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal

from seaborn._core.groupby import GroupBy
from seaborn._stats.density import KDE, _no_scipy
from seaborn._compat import groupby_apply_include_groups
from seaborn.external.kde import gaussian_kde


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
        expected = [ori, "alpha", "density", other]
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
            .apply(
                lambda x: self.integrate(x["density"], x[ori]),
                **groupby_apply_include_groups(False),
            )
        )

        if common_norm:
            assert areas.sum() == pytest.approx(1, abs=1e-3)
        else:
            assert_array_almost_equal(areas, [1, 1], decimal=3)

    def test_common_norm_variables(self, df):

        ori = "y"
        df = df[[ori, "alpha", "color"]]
        gb = self.get_groupby(df, ori)
        res = KDE(common_norm=["alpha"])(df, gb, ori, {})

        def integrate_by_color_and_sum(x):
            return (
                x.groupby("color")
                .apply(
                    lambda y: self.integrate(y["density"], y[ori]),
                    **groupby_apply_include_groups(False)
                )
                .sum()
            )

        areas = (
            res
            .groupby("alpha")
            .apply(integrate_by_color_and_sum, **groupby_apply_include_groups(False))
        )
        assert_array_almost_equal(areas, [1, 1], decimal=3)

    @pytest.mark.parametrize("param", ["norm", "grid"])
    def test_common_input_checks(self, df, param):

        ori = "y"
        df = df[[ori, "alpha"]]
        gb = self.get_groupby(df, ori)
        msg = rf"Undefined variable\(s\) passed for KDE.common_{param}"
        with pytest.warns(UserWarning, match=msg):
            KDE(**{f"common_{param}": ["color", "alpha"]})(df, gb, ori, {})

        msg = f"KDE.common_{param} must be a boolean or list of strings"
        with pytest.raises(TypeError, match=msg):
            KDE(**{f"common_{param}": "alpha"})(df, gb, ori, {})

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

    @pytest.mark.parametrize("vals", [[], [1], [1] * 5, [1929245168.06679] * 18])
    def test_singular(self, df, vals):

        df1 = pd.DataFrame({"y": vals, "alpha": ["z"] * len(vals)})
        gb = self.get_groupby(df1, "y")
        res = KDE()(df1, gb, "y", {})
        assert res.empty

        df2 = pd.concat([df[["y", "alpha"]], df1], ignore_index=True)
        gb = self.get_groupby(df2, "y")
        res = KDE()(df2, gb, "y", {})
        assert set(res["alpha"]) == set(df["alpha"])

    @pytest.mark.parametrize("col", ["y", "weight"])
    def test_missing(self, df, col):

        val, ori = "xy"
        df["weight"] = 1
        df = df[[ori, "weight"]].astype(float)
        df.loc[:4, col] = np.nan
        gb = self.get_groupby(df, ori)
        res = KDE()(df, gb, ori, {})
        assert self.integrate(res[val], res[ori]) == pytest.approx(1, abs=1e-3)


class TestVendoredKDE:
    """Regression tests for the vendored gaussian_kde (gh-3883)."""

    def test_2d_matches_scipy_reference(self):
        # The vendored copy substitutes numpy.linalg for scipy.linalg, but
        # numpy.linalg.cholesky returns the lower-triangular factor while
        # scipy.linalg.cholesky returns the upper-triangular factor by
        # default. Reference values below were computed with
        # scipy.stats.gaussian_kde on the same fixed-seed data.
        rng = np.random.default_rng(0)
        x = rng.multivariate_normal([0, 0], [[1, .5], [.5, 1]], 200).T
        grid = np.linspace(-3, 3, 7)
        gx, gy = np.meshgrid(grid, grid)
        points = np.vstack([gx.ravel(), gy.ravel()])
        expected = np.array([
            4.392005640740602e-05, 2.301882196848023e-04, 8.181101797375473e-04,
            5.253047324776023e-05, 1.518593585537811e-03, 7.034420397670235e-05,
            1.956470306545655e-09, 5.407036076935740e-03, 8.152753979941451e-03,
            2.021448546036320e-02, 9.515995179086230e-03, 1.965627053184416e-03,
            1.886517054301043e-03, 2.181027246260232e-06, 2.113859844126985e-03,
            3.894833487485186e-02, 8.829352770993749e-02, 7.732720268059057e-02,
            2.401198849506442e-02, 7.923109731600118e-04, 4.017848850769105e-06,
            2.258676072484105e-04, 1.484372043947954e-02, 8.796656669729505e-02,
            1.454017714262485e-01, 1.159222619476802e-01, 1.611121838343852e-02,
            8.672563320022962e-04, 1.552291566204162e-07, 1.265434963482988e-03,
            2.632606856601287e-02, 8.779978237521208e-02, 9.522174163867828e-02,
            3.311565104433512e-02, 1.530996563714401e-03, 5.120445602466335e-10,
            6.946221481163648e-05, 8.636271846297207e-03, 2.183586651506960e-02,
            2.473931538347198e-02, 1.764548365835012e-02, 5.547317387096062e-03,
            2.147639541945436e-14, 1.063374635213697e-07, 3.182238394277357e-04,
            1.168857510575398e-03, 2.432017867259575e-03, 3.250501715620722e-03,
            2.496168912414599e-03,
        ])
        result = gaussian_kde(x)(points)
        assert_allclose(result, expected, rtol=1e-8, atol=1e-12)
