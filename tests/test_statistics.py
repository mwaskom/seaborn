import numpy as np
import pandas as pd

try:
    import statsmodels.distributions as smdist
except ImportError:
    smdist = None

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._statistics import (
    KDE,
    Histogram,
    ECDF,
    EstimateAggregator,
    LetterValues,
    WeightedEstimateAggregator,
    _validate_errorbar_arg,
    _no_scipy,
)


class DistributionFixtures:

    @pytest.fixture
    def x(self, rng):
        return rng.normal(0, 1, 100)

    @pytest.fixture
    def x2(self, rng):
        return rng.normal(0, 1, 742)  # random value to avoid edge cases

    @pytest.fixture
    def y(self, rng):
        return rng.normal(0, 5, 100)

    @pytest.fixture
    def weights(self, rng):
        return rng.uniform(0, 5, 100)


class TestKDE:

    def integrate(self, y, x):
        y = np.asarray(y)
        x = np.asarray(x)
        dx = np.diff(x)
        return (dx * y[:-1] + dx * y[1:]).sum() / 2

    def test_gridsize(self, rng):

        x = rng.normal(0, 3, 1000)

        n = 200
        kde = KDE(gridsize=n)
        density, support = kde(x)
        assert density.size == n
        assert support.size == n

    def test_cut(self, rng):

        x = rng.normal(0, 3, 1000)

        kde = KDE(cut=0)
        _, support = kde(x)
        assert support.min() == x.min()
        assert support.max() == x.max()

        cut = 2
        bw_scale = .5
        bw = x.std() * bw_scale
        kde = KDE(cut=cut, bw_method=bw_scale, gridsize=1000)
        _, support = kde(x)
        assert support.min() == pytest.approx(x.min() - bw * cut, abs=1e-2)
        assert support.max() == pytest.approx(x.max() + bw * cut, abs=1e-2)

    def test_clip(self, rng):

        x = rng.normal(0, 3, 100)
        clip = -1, 1
        kde = KDE(clip=clip)
        _, support = kde(x)

        assert support.min() >= clip[0]
        assert support.max() <= clip[1]

    def test_density_normalization(self, rng):

        x = rng.normal(0, 3, 1000)
        kde = KDE()
        density, support = kde(x)
        assert self.integrate(density, support) == pytest.approx(1, abs=1e-5)

    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_cumulative(self, rng):

        x = rng.normal(0, 3, 1000)
        kde = KDE(cumulative=True)
        density, _ = kde(x)
        assert density[0] == pytest.approx(0, abs=1e-5)
        assert density[-1] == pytest.approx(1, abs=1e-5)

    def test_cached_support(self, rng):

        x = rng.normal(0, 3, 100)
        kde = KDE()
        kde.define_support(x)
        _, support = kde(x[(x > -1) & (x < 1)])
        assert_array_equal(support, kde.support)

    def test_bw_method(self, rng):

        x = rng.normal(0, 3, 100)
        kde1 = KDE(bw_method=.2)
        kde2 = KDE(bw_method=2)

        d1, _ = kde1(x)
        d2, _ = kde2(x)

        assert np.abs(np.diff(d1)).mean() > np.abs(np.diff(d2)).mean()

    def test_bw_adjust(self, rng):

        x = rng.normal(0, 3, 100)
        kde1 = KDE(bw_adjust=.2)
        kde2 = KDE(bw_adjust=2)

        d1, _ = kde1(x)
        d2, _ = kde2(x)

        assert np.abs(np.diff(d1)).mean() > np.abs(np.diff(d2)).mean()

    def test_bivariate_grid(self, rng):

        n = 100
        x, y = rng.normal(0, 3, (2, 50))
        kde = KDE(gridsize=n)
        density, (xx, yy) = kde(x, y)

        assert density.shape == (n, n)
        assert xx.size == n
        assert yy.size == n

    def test_bivariate_normalization(self, rng):

        x, y = rng.normal(0, 3, (2, 50))
        kde = KDE(gridsize=100)
        density, (xx, yy) = kde(x, y)

        dx = xx[1] - xx[0]
        dy = yy[1] - yy[0]

        total = density.sum() * (dx * dy)
        assert total == pytest.approx(1, abs=1e-2)

    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_bivariate_cumulative(self, rng):

        x, y = rng.normal(0, 3, (2, 50))
        kde = KDE(gridsize=100, cumulative=True)
        density, _ = kde(x, y)

        assert density[0, 0] == pytest.approx(0, abs=1e-2)
        assert density[-1, -1] == pytest.approx(1, abs=1e-2)


class TestHistogram(DistributionFixtures):

    def test_string_bins(self, x):

        h = Histogram(bins="sqrt")
        bin_kws = h.define_bin_params(x)
        assert bin_kws["range"] == (x.min(), x.max())
        assert bin_kws["bins"] == int(np.sqrt(len(x)))

    def test_int_bins(self, x):

        n = 24
        h = Histogram(bins=n)
        bin_kws = h.define_bin_params(x)
        assert bin_kws["range"] == (x.min(), x.max())
        assert bin_kws["bins"] == n

    def test_array_bins(self, x):

        bins = [-3, -2, 1, 2, 3]
        h = Histogram(bins=bins)
        bin_kws = h.define_bin_params(x)
        assert_array_equal(bin_kws["bins"], bins)

    def test_bivariate_string_bins(self, x, y):

        s1, s2 = "sqrt", "fd"

        h = Histogram(bins=s1)
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert_array_equal(e1, np.histogram_bin_edges(x, s1))
        assert_array_equal(e2, np.histogram_bin_edges(y, s1))

        h = Histogram(bins=(s1, s2))
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert_array_equal(e1, np.histogram_bin_edges(x, s1))
        assert_array_equal(e2, np.histogram_bin_edges(y, s2))

    def test_bivariate_int_bins(self, x, y):

        b1, b2 = 5, 10

        h = Histogram(bins=b1)
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert len(e1) == b1 + 1
        assert len(e2) == b1 + 1

        h = Histogram(bins=(b1, b2))
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert len(e1) == b1 + 1
        assert len(e2) == b2 + 1

    def test_bivariate_array_bins(self, x, y):

        b1 = [-3, -2, 1, 2, 3]
        b2 = [-5, -2, 3, 6]

        h = Histogram(bins=b1)
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert_array_equal(e1, b1)
        assert_array_equal(e2, b1)

        h = Histogram(bins=(b1, b2))
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert_array_equal(e1, b1)
        assert_array_equal(e2, b2)

    def test_binwidth(self, x):

        binwidth = .5
        h = Histogram(binwidth=binwidth)
        bin_kws = h.define_bin_params(x)
        n_bins = bin_kws["bins"]
        left, right = bin_kws["range"]
        assert (right - left) / n_bins == pytest.approx(binwidth)

    def test_bivariate_binwidth(self, x, y):

        w1, w2 = .5, 1

        h = Histogram(binwidth=w1)
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert np.all(np.diff(e1) == w1)
        assert np.all(np.diff(e2) == w1)

        h = Histogram(binwidth=(w1, w2))
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert np.all(np.diff(e1) == w1)
        assert np.all(np.diff(e2) == w2)

    def test_binrange(self, x):

        binrange = (-4, 4)
        h = Histogram(binrange=binrange)
        bin_kws = h.define_bin_params(x)
        assert bin_kws["range"] == binrange

    def test_bivariate_binrange(self, x, y):

        r1, r2 = (-4, 4), (-10, 10)

        h = Histogram(binrange=r1)
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert e1.min() == r1[0]
        assert e1.max() == r1[1]
        assert e2.min() == r1[0]
        assert e2.max() == r1[1]

        h = Histogram(binrange=(r1, r2))
        e1, e2 = h.define_bin_params(x, y)["bins"]
        assert e1.min() == r1[0]
        assert e1.max() == r1[1]
        assert e2.min() == r2[0]
        assert e2.max() == r2[1]

    def test_discrete_bins(self, rng):

        x = rng.binomial(20, .5, 100)
        h = Histogram(discrete=True)
        bin_kws = h.define_bin_params(x)
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    def test_odd_single_observation(self):
        # GH2721
        x = np.array([0.49928])
        h, e = Histogram(binwidth=0.03)(x)
        assert len(h) == 1
        assert (e[1] - e[0]) == pytest.approx(.03)

    def test_binwidth_roundoff(self):
        # GH2785
        x = np.array([2.4, 2.5, 2.6])
        h, e = Histogram(binwidth=0.01)(x)
        assert h.sum() == 3

    def test_histogram(self, x):

        h = Histogram()
        heights, edges = h(x)
        heights_mpl, edges_mpl = np.histogram(x, bins="auto")

        assert_array_equal(heights, heights_mpl)
        assert_array_equal(edges, edges_mpl)

    def test_count_stat(self, x):

        h = Histogram(stat="count")
        heights, _ = h(x)
        assert heights.sum() == len(x)

    def test_density_stat(self, x):

        h = Histogram(stat="density")
        heights, edges = h(x)
        assert (heights * np.diff(edges)).sum() == 1

    def test_probability_stat(self, x):

        h = Histogram(stat="probability")
        heights, _ = h(x)
        assert heights.sum() == 1

    def test_frequency_stat(self, x):

        h = Histogram(stat="frequency")
        heights, edges = h(x)
        assert (heights * np.diff(edges)).sum() == len(x)

    def test_cumulative_count(self, x):

        h = Histogram(stat="count", cumulative=True)
        heights, _ = h(x)
        assert heights[-1] == len(x)

    def test_cumulative_density(self, x):

        h = Histogram(stat="density", cumulative=True)
        heights, _ = h(x)
        assert heights[-1] == 1

    def test_cumulative_probability(self, x):

        h = Histogram(stat="probability", cumulative=True)
        heights, _ = h(x)
        assert heights[-1] == 1

    def test_cumulative_frequency(self, x):

        h = Histogram(stat="frequency", cumulative=True)
        heights, _ = h(x)
        assert heights[-1] == len(x)

    def test_bivariate_histogram(self, x, y):

        h = Histogram()
        heights, edges = h(x, y)
        bins_mpl = (
            np.histogram_bin_edges(x, "auto"),
            np.histogram_bin_edges(y, "auto"),
        )
        heights_mpl, *edges_mpl = np.histogram2d(x, y, bins_mpl)
        assert_array_equal(heights, heights_mpl)
        assert_array_equal(edges[0], edges_mpl[0])
        assert_array_equal(edges[1], edges_mpl[1])

    def test_bivariate_count_stat(self, x, y):

        h = Histogram(stat="count")
        heights, _ = h(x, y)
        assert heights.sum() == len(x)

    def test_bivariate_density_stat(self, x, y):

        h = Histogram(stat="density")
        heights, (edges_x, edges_y) = h(x, y)
        areas = np.outer(np.diff(edges_x), np.diff(edges_y))
        assert (heights * areas).sum() == pytest.approx(1)

    def test_bivariate_probability_stat(self, x, y):

        h = Histogram(stat="probability")
        heights, _ = h(x, y)
        assert heights.sum() == 1

    def test_bivariate_frequency_stat(self, x, y):

        h = Histogram(stat="frequency")
        heights, (x_edges, y_edges) = h(x, y)
        area = np.outer(np.diff(x_edges), np.diff(y_edges))
        assert (heights * area).sum() == len(x)

    def test_bivariate_cumulative_count(self, x, y):

        h = Histogram(stat="count", cumulative=True)
        heights, _ = h(x, y)
        assert heights[-1, -1] == len(x)

    def test_bivariate_cumulative_density(self, x, y):

        h = Histogram(stat="density", cumulative=True)
        heights, _ = h(x, y)
        assert heights[-1, -1] == pytest.approx(1)

    def test_bivariate_cumulative_frequency(self, x, y):

        h = Histogram(stat="frequency", cumulative=True)
        heights, _ = h(x, y)
        assert heights[-1, -1] == len(x)

    def test_bivariate_cumulative_probability(self, x, y):

        h = Histogram(stat="probability", cumulative=True)
        heights, _ = h(x, y)
        assert heights[-1, -1] == pytest.approx(1)

    def test_bad_stat(self):

        with pytest.raises(ValueError):
            Histogram(stat="invalid")


class TestECDF(DistributionFixtures):

    def test_univariate_proportion(self, x):

        ecdf = ECDF()
        stat, vals = ecdf(x)
        assert_array_equal(vals[1:], np.sort(x))
        assert_array_almost_equal(stat[1:], np.linspace(0, 1, len(x) + 1)[1:])
        assert stat[0] == 0

    def test_univariate_count(self, x):

        ecdf = ECDF(stat="count")
        stat, vals = ecdf(x)

        assert_array_equal(vals[1:], np.sort(x))
        assert_array_almost_equal(stat[1:], np.arange(len(x)) + 1)
        assert stat[0] == 0

    def test_univariate_percent(self, x2):

        ecdf = ECDF(stat="percent")
        stat, vals = ecdf(x2)

        assert_array_equal(vals[1:], np.sort(x2))
        assert_array_almost_equal(stat[1:], (np.arange(len(x2)) + 1) / len(x2) * 100)
        assert stat[0] == 0

    def test_univariate_proportion_weights(self, x, weights):

        ecdf = ECDF()
        stat, vals = ecdf(x, weights=weights)
        assert_array_equal(vals[1:], np.sort(x))
        expected_stats = weights[x.argsort()].cumsum() / weights.sum()
        assert_array_almost_equal(stat[1:], expected_stats)
        assert stat[0] == 0

    def test_univariate_count_weights(self, x, weights):

        ecdf = ECDF(stat="count")
        stat, vals = ecdf(x, weights=weights)
        assert_array_equal(vals[1:], np.sort(x))
        assert_array_almost_equal(stat[1:], weights[x.argsort()].cumsum())
        assert stat[0] == 0

    @pytest.mark.skipif(smdist is None, reason="Requires statsmodels")
    def test_against_statsmodels(self, x):

        sm_ecdf = smdist.empirical_distribution.ECDF(x)

        ecdf = ECDF()
        stat, vals = ecdf(x)
        assert_array_equal(vals, sm_ecdf.x)
        assert_array_almost_equal(stat, sm_ecdf.y)

        ecdf = ECDF(complementary=True)
        stat, vals = ecdf(x)
        assert_array_equal(vals, sm_ecdf.x)
        assert_array_almost_equal(stat, sm_ecdf.y[::-1])

    def test_invalid_stat(self, x):

        with pytest.raises(ValueError, match="`stat` must be one of"):
            ECDF(stat="density")

    def test_bivariate_error(self, x, y):

        with pytest.raises(NotImplementedError, match="Bivariate ECDF"):
            ecdf = ECDF()
            ecdf(x, y)


class TestEstimateAggregator:

    def test_func_estimator(self, long_df):

        func = np.mean
        agg = EstimateAggregator(func)
        out = agg(long_df, "x")
        assert out["x"] == func(long_df["x"])

    def test_name_estimator(self, long_df):

        agg = EstimateAggregator("mean")
        out = agg(long_df, "x")
        assert out["x"] == long_df["x"].mean()

    def test_custom_func_estimator(self, long_df):

        def func(x):
            return np.asarray(x).min()

        agg = EstimateAggregator(func)
        out = agg(long_df, "x")
        assert out["x"] == func(long_df["x"])

    def test_se_errorbars(self, long_df):

        agg = EstimateAggregator("mean", "se")
        out = agg(long_df, "x")
        assert out["x"] == long_df["x"].mean()
        assert out["xmin"] == (long_df["x"].mean() - long_df["x"].sem())
        assert out["xmax"] == (long_df["x"].mean() + long_df["x"].sem())

        agg = EstimateAggregator("mean", ("se", 2))
        out = agg(long_df, "x")
        assert out["x"] == long_df["x"].mean()
        assert out["xmin"] == (long_df["x"].mean() - 2 * long_df["x"].sem())
        assert out["xmax"] == (long_df["x"].mean() + 2 * long_df["x"].sem())

    def test_sd_errorbars(self, long_df):

        agg = EstimateAggregator("mean", "sd")
        out = agg(long_df, "x")
        assert out["x"] == long_df["x"].mean()
        assert out["xmin"] == (long_df["x"].mean() - long_df["x"].std())
        assert out["xmax"] == (long_df["x"].mean() + long_df["x"].std())

        agg = EstimateAggregator("mean", ("sd", 2))
        out = agg(long_df, "x")
        assert out["x"] == long_df["x"].mean()
        assert out["xmin"] == (long_df["x"].mean() - 2 * long_df["x"].std())
        assert out["xmax"] == (long_df["x"].mean() + 2 * long_df["x"].std())

    def test_pi_errorbars(self, long_df):

        agg = EstimateAggregator("mean", "pi")
        out = agg(long_df, "y")
        assert out["ymin"] == np.percentile(long_df["y"], 2.5)
        assert out["ymax"] == np.percentile(long_df["y"], 97.5)

        agg = EstimateAggregator("mean", ("pi", 50))
        out = agg(long_df, "y")
        assert out["ymin"] == np.percentile(long_df["y"], 25)
        assert out["ymax"] == np.percentile(long_df["y"], 75)

    def test_ci_errorbars(self, long_df):

        agg = EstimateAggregator("mean", "ci", n_boot=100000, seed=0)
        out = agg(long_df, "y")

        agg_ref = EstimateAggregator("mean", ("se", 1.96))
        out_ref = agg_ref(long_df, "y")

        assert out["ymin"] == pytest.approx(out_ref["ymin"], abs=1e-2)
        assert out["ymax"] == pytest.approx(out_ref["ymax"], abs=1e-2)

        agg = EstimateAggregator("mean", ("ci", 68), n_boot=100000, seed=0)
        out = agg(long_df, "y")

        agg_ref = EstimateAggregator("mean", ("se", 1))
        out_ref = agg_ref(long_df, "y")

        assert out["ymin"] == pytest.approx(out_ref["ymin"], abs=1e-2)
        assert out["ymax"] == pytest.approx(out_ref["ymax"], abs=1e-2)

        agg = EstimateAggregator("mean", "ci", seed=0)
        out_orig = agg_ref(long_df, "y")
        out_test = agg_ref(long_df, "y")
        assert_array_equal(out_orig, out_test)

    def test_custom_errorbars(self, long_df):

        f = lambda x: (x.min(), x.max())  # noqa: E731
        agg = EstimateAggregator("mean", f)
        out = agg(long_df, "y")
        assert out["ymin"] == long_df["y"].min()
        assert out["ymax"] == long_df["y"].max()

    def test_singleton_errorbars(self):

        agg = EstimateAggregator("mean", "ci")
        val = 7
        out = agg(pd.DataFrame(dict(y=[val])), "y")
        assert out["y"] == val
        assert pd.isna(out["ymin"])
        assert pd.isna(out["ymax"])

    def test_errorbar_validation(self):

        method, level = _validate_errorbar_arg(("ci", 99))
        assert method == "ci"
        assert level == 99

        method, level = _validate_errorbar_arg("sd")
        assert method == "sd"
        assert level == 1

        f = lambda x: (x.min(), x.max())  # noqa: E731
        method, level = _validate_errorbar_arg(f)
        assert method is f
        assert level is None

        bad_args = [
            ("sem", ValueError),
            (("std", 2), ValueError),
            (("pi", 5, 95), ValueError),
            (95, TypeError),
            (("ci", "large"), TypeError),
        ]

        for arg, exception in bad_args:
            with pytest.raises(exception, match="`errorbar` must be"):
                _validate_errorbar_arg(arg)


class TestWeightedEstimateAggregator:

    def test_weighted_mean(self, long_df):

        long_df["weight"] = long_df["x"]
        est = WeightedEstimateAggregator("mean")
        out = est(long_df, "y")
        expected = np.average(long_df["y"], weights=long_df["weight"])
        assert_array_equal(out["y"], expected)
        assert_array_equal(out["ymin"], np.nan)
        assert_array_equal(out["ymax"], np.nan)

    def test_weighted_ci(self, long_df):

        long_df["weight"] = long_df["x"]
        est = WeightedEstimateAggregator("mean", "ci")
        out = est(long_df, "y")
        expected = np.average(long_df["y"], weights=long_df["weight"])
        assert_array_equal(out["y"], expected)
        assert (out["ymin"] <= out["y"]).all()
        assert (out["ymax"] >= out["y"]).all()

    def test_limited_estimator(self):

        with pytest.raises(ValueError, match="Weighted estimator must be 'mean'"):
            WeightedEstimateAggregator("median")

    def test_limited_ci(self):

        with pytest.raises(ValueError, match="Error bar method must be 'ci'"):
            WeightedEstimateAggregator("mean", "sd")


class TestLetterValues:

    @pytest.fixture
    def x(self, rng):
        return pd.Series(rng.standard_t(10, 10_000))

    def test_levels(self, x):

        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        k = res["k"]
        expected = np.concatenate([np.arange(k), np.arange(k - 1)[::-1]])
        assert_array_equal(res["levels"], expected)

    def test_values(self, x):

        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        assert_array_equal(np.percentile(x, res["percs"]), res["values"])

    def test_fliers(self, x):

        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        fliers = res["fliers"]
        values = res["values"]
        assert ((fliers < values.min()) | (fliers > values.max())).all()

    def test_median(self, x):

        res = LetterValues(k_depth="tukey", outlier_prop=0, trust_alpha=0)(x)
        assert res["median"] == np.median(x)

    def test_k_depth_int(self, x):

        res = LetterValues(k_depth=(k := 12), outlier_prop=0, trust_alpha=0)(x)
        assert res["k"] == k
        assert len(res["levels"]) == (2 * k - 1)

    def test_trust_alpha(self, x):

        res1 = LetterValues(k_depth="trustworthy", outlier_prop=0, trust_alpha=.1)(x)
        res2 = LetterValues(k_depth="trustworthy", outlier_prop=0, trust_alpha=.001)(x)
        assert res1["k"] > res2["k"]

    def test_outlier_prop(self, x):

        res1 = LetterValues(k_depth="proportion", outlier_prop=.001, trust_alpha=0)(x)
        res2 = LetterValues(k_depth="proportion", outlier_prop=.1, trust_alpha=0)(x)
        assert res1["k"] > res2["k"]
