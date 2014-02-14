import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import moss

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt

from .. import linearmodels as lm
from ..utils import color_palette

rs = np.random.RandomState(0)


class TestLinearPlotter(object):

    rs = np.random.RandomState(77)
    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           d=rs.randint(-2, 3, 60),
                           y=rs.gamma(4, size=60),
                           s=np.tile(list("abcdefghij"), 6)))
    df["z"] = df.y + rs.randn(60)
    df["y_na"] = df.y.copy()
    df.y_na.ix[[10, 20, 30]] = np.nan

    def test_establish_variables_from_frame(self):

        p = lm._LinearPlotter()
        p.establish_variables(self.df, x="x", y="y")
        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_frame_equal(p.data, self.df)

    def test_establish_variables_from_series(self):

        p = lm._LinearPlotter()
        p.establish_variables(None, x=self.df.x, y=self.df.y)
        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        nt.assert_is(p.data, None)

    def test_establish_variables_from_array(self):

        p = lm._LinearPlotter()
        p.establish_variables(None,
                              x=self.df.x.values,
                              y=self.df.y.values)
        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        nt.assert_is(p.data, None)

    def test_establish_variables_from_mix(self):

        p = lm._LinearPlotter()
        p.establish_variables(self.df, x="x", y=self.df.y)
        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_frame_equal(p.data, self.df)

    def test_establish_variables_from_bad(self):

        p = lm._LinearPlotter()
        with nt.assert_raises(ValueError):
            p.establish_variables(None, x="x", y=self.df.y)

    def test_dropna(self):

        p = lm._LinearPlotter()
        p.establish_variables(self.df, x="x", y_na="y_na")
        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y_na, self.df.y_na)

        p.dropna("x", "y_na")
        mask = self.df.y_na.notnull()
        pdt.assert_series_equal(p.x, self.df.x[mask])
        pdt.assert_series_equal(p.y_na, self.df.y_na[mask])


class TestRegressionPlotter(object):

    rs = np.random.RandomState(49)

    grid = np.linspace(-3, 3, 30)
    n_boot = 100
    bins_numeric = 3
    bins_given = [-1, 0, 1]

    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           d=rs.randint(-2, 3, 60),
                           y=rs.gamma(4, size=60),
                           s=np.tile(list("abcdefghij"), 6)))
    df["z"] = df.y + rs.randn(60)
    df["y_na"] = df.y.copy()

    p = 1 / (1 + np.exp(-(df.x * 2 + rs.randn(60))))
    df["c"] = [rs.binomial(1, p_i) for p_i in p]
    df.y_na.ix[[10, 20, 30]] = np.nan

    def test_variables_from_frame(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, units="s")

        pdt.assert_series_equal(p.x, self.df.x)
        pdt.assert_series_equal(p.y, self.df.y)
        pdt.assert_series_equal(p.units, self.df.s)
        pdt.assert_frame_equal(p.data, self.df)

    def test_variables_from_series(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y, units=self.df.s)

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        npt.assert_array_equal(p.units, self.df.s)
        nt.assert_is(p.data, None)

    def test_variables_from_mix(self):

        p = lm._RegressionPlotter("x", self.df.y + 1, data=self.df)

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y + 1)
        pdt.assert_frame_equal(p.data, self.df)

    def test_dropna(self):

        p = lm._RegressionPlotter("x", "y_na", data=self.df)
        nt.assert_equal(len(p.x), pd.notnull(self.df.y_na).sum())

        p = lm._RegressionPlotter("x", "y_na", data=self.df, dropna=False)
        nt.assert_equal(len(p.x), len(self.df.y_na))

    def test_ci(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, ci=95)
        nt.assert_equal(p.ci, 95)
        nt.assert_equal(p.x_ci, 95)

        p = lm._RegressionPlotter("x", "y", data=self.df, ci=95, x_ci=68)
        nt.assert_equal(p.ci, 95)
        nt.assert_equal(p.x_ci, 68)

    def test_fast_regression(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # Fit with the "fast" function, which just does linear algebra
        yhat_fast, _ = p.fit_fast(self.grid)

        # Fit using the statsmodels function with an OLS model
        yhat_smod, _ = p.fit_statsmodels(self.grid, sm.OLS)

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_fast, yhat_smod)

    def test_regress_poly(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # Fit an first-order polynomial
        yhat_poly, _ = p.fit_poly(self.grid, 1)

        # Fit using the statsmodels function with an OLS model
        yhat_smod, _ = p.fit_statsmodels(self.grid, sm.OLS)

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_poly, yhat_smod)

    def test_regress_n_boot(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, n_boot=self.n_boot)

        # Fast (linear algebra) version
        _, boots_fast = p.fit_fast(self.grid)
        npt.assert_equal(boots_fast.shape, (self.n_boot, self.grid.size))

        # Slower (np.polyfit) version
        _, boots_poly = p.fit_poly(self.grid, 1)
        npt.assert_equal(boots_poly.shape, (self.n_boot, self.grid.size))

        # Slowest (statsmodels) version
        _, boots_smod = p.fit_statsmodels(self.grid, sm.OLS)
        npt.assert_equal(boots_smod.shape, (self.n_boot, self.grid.size))

    def test_regress_without_bootstrap(self):

        p = lm._RegressionPlotter("x", "y", data=self.df,
                                  n_boot=self.n_boot, ci=None)

        # Fast (linear algebra) version
        _, boots_fast = p.fit_fast(self.grid)
        nt.assert_is(boots_fast, None)

        # Slower (np.polyfit) version
        _, boots_poly = p.fit_poly(self.grid, 1)
        nt.assert_is(boots_poly, None)

        # Slowest (statsmodels) version
        _, boots_smod = p.fit_statsmodels(self.grid, sm.OLS)
        nt.assert_is(boots_smod, None)

    def test_numeric_bins(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x_binned, bins = p.bin_predictor(self.bins_numeric)
        npt.assert_equal(len(bins), self.bins_numeric)
        npt.assert_array_equal(np.unique(x_binned), bins)

    def test_provided_bins(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x_binned, bins = p.bin_predictor(self.bins_given)
        npt.assert_array_equal(np.unique(x_binned), self.bins_given)

    def test_bin_results(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x_binned, bins = p.bin_predictor(self.bins_given)
        nt.assert_greater(self.df.x[x_binned == 0].min(),
                          self.df.x[x_binned == -1].max())
        nt.assert_greater(self.df.x[x_binned == 1].min(),
                          self.df.x[x_binned == 0].max())

    def test_scatter_data(self):

        p = lm._RegressionPlotter(self.df.x, self.df.y)
        x, y = p.scatter_data
        npt.assert_array_equal(x, self.df.x)
        npt.assert_array_equal(y, self.df.y)

        p = lm._RegressionPlotter(self.df.d, self.df.y)
        x, y = p.scatter_data
        npt.assert_array_equal(x, self.df.d)
        npt.assert_array_equal(y, self.df.y)

        p = lm._RegressionPlotter(self.df.d, self.df.y, x_jitter=.1)
        x, y = p.scatter_data
        nt.assert_true((x != self.df.d).any())
        npt.assert_array_less(np.abs(self.df.d - x), np.repeat(.1, len(x)))
        npt.assert_array_equal(y, self.df.y)

        p = lm._RegressionPlotter(self.df.d, self.df.y, y_jitter=.05)
        x, y = p.scatter_data
        npt.assert_array_equal(x, self.df.d)
        npt.assert_array_less(np.abs(self.df.y - y), np.repeat(.1, len(y)))

    def test_estimate_data(self):

        p = lm._RegressionPlotter(self.df.d, self.df.y, x_estimator=np.mean)

        x, y, ci = p.estimate_data

        npt.assert_array_equal(x, np.sort(np.unique(self.df.d)))
        npt.assert_array_equal(y, self.df.groupby("d").y.mean())
        npt.assert_array_less(np.array(ci)[:, 0], y)
        npt.assert_array_less(y, np.array(ci)[:, 1])

    def test_estimate_cis(self):

        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=95)
        _, _, ci_big = p.estimate_data

        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=50)
        _, _, ci_wee = p.estimate_data
        npt.assert_array_less(np.diff(ci_wee), np.diff(ci_big))

        p = lm._RegressionPlotter(self.df.d, self.df.y,
                                  x_estimator=np.mean, ci=None)
        _, _, ci_nil = p.estimate_data
        npt.assert_array_equal(ci_nil, [None] * len(ci_nil))

    def test_logistic_regression(self):

        p = lm._RegressionPlotter("x", "c", data=self.df,
                                  logistic=True, n_boot=self.n_boot)
        _, yhat, _ = p.fit_regression(x_range=(-3, 3))
        npt.assert_array_less(yhat, 1)
        npt.assert_array_less(0, yhat)

    def test_robust_regression(self):

        p_ols = lm._RegressionPlotter("x", "y", data=self.df,
                                      n_boot=self.n_boot)
        _, ols_yhat, _ = p_ols.fit_regression(x_range=(-3, 3))

        p_robust = lm._RegressionPlotter("x", "y", data=self.df,
                                         robust=True, n_boot=self.n_boot)
        _, robust_yhat, _ = p_robust.fit_regression(x_range=(-3, 3))

        nt.assert_equal(len(ols_yhat), len(robust_yhat))

    def test_lowess_regression(self):

        p = lm._RegressionPlotter("x", "y", data=self.df, lowess=True)
        grid, yhat, err_bands = p.fit_regression(x_range=(-3, 3))

        nt.assert_equal(len(grid), len(yhat))
        nt.assert_is(err_bands, None)


class TestDiscretePlotter(object):

    rs = np.random.RandomState(341)
    df = pd.DataFrame(dict(x=np.repeat(list("abc"), 30),
                           y=rs.randn(90),
                           g=np.tile(list("xy"), 45),
                           u=np.tile(np.arange(6), 15)))
    bw_err = rs.randn(6)[df.u.values]
    df.y += bw_err
    df["y_na"] = df.y.copy()
    df.y_na.ix[[10, 20, 30]] = np.nan

    def test_variables_from_frame(self):

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, units="u")

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        npt.assert_array_equal(p.hue, self.df.g)
        npt.assert_array_equal(p.units, self.df.u)
        pdt.assert_frame_equal(p.data, self.df)

    def test_variables_from_series(self):

        p = lm._DiscretePlotter(self.df.x, self.df.y, self.df.g,
                                units=self.df.u)

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        npt.assert_array_equal(p.hue, self.df.g)
        npt.assert_array_equal(p.units, self.df.u)
        nt.assert_is(p.data, None)

    def test_variables_from_mix(self):

        p = lm._DiscretePlotter("x", self.df.y + 1, data=self.df)

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y + 1)
        pdt.assert_frame_equal(p.data, self.df)

    def test_variables_var_order(self):

        p = lm._DiscretePlotter("x", "y", "g", data=self.df)

        npt.assert_array_equal(p.x_order, list("abc"))
        npt.assert_array_equal(p.hue_order, list("xy"))

        x_order = list("bca")
        hue_order = list("yx")
        p = lm._DiscretePlotter("x", "y", "g", data=self.df,
                                x_order=x_order, hue_order=hue_order)

        npt.assert_array_equal(p.x_order, x_order)
        npt.assert_array_equal(p.hue_order, hue_order)

    def test_count_x(self):

        p = lm._DiscretePlotter("x", hue="g", data=self.df)
        nt.assert_true(p.y_count)
        npt.assert_array_equal(p.x, p.y)
        nt.assert_is(p.estimator, len)

    def test_dropna(self):

        p = lm._DiscretePlotter("x", "y_na", data=self.df)
        nt.assert_equal(len(p.x), pd.notnull(self.df.y_na).sum())

        p = lm._DiscretePlotter("x", "y_na", data=self.df, dropna=False)
        nt.assert_equal(len(p.x), len(self.df.y_na))

    def test_palette(self):

        p = lm._DiscretePlotter("x", "y", data=self.df)
        nt.assert_equal(p.palette, [color_palette()[0]] * 3)

        p = lm._DiscretePlotter("x", "y", data=self.df, color="green")
        nt.assert_equal(p.palette, ["green"] * 3)

        p = lm._DiscretePlotter("x", "y", data=self.df, palette="husl")
        nt.assert_equal(p.palette, color_palette("husl", 3))
        nt.assert_true(p.x_palette)

        p = lm._DiscretePlotter("x", "y", "g", data=self.df)
        nt.assert_equal(p.palette, color_palette(n_colors=2))

        pal = {"x": "pink", "y": "green"}
        p = lm._DiscretePlotter("x", "y", "g", data=self.df, palette=pal)
        nt.assert_equal(p.palette, color_palette(["pink", "green"], 2))

        p = lm._DiscretePlotter("x", "y", "g", data=self.df,
                                palette=pal, hue_order=list("yx"))
        nt.assert_equal(p.palette, color_palette(["green", "pink"], 2))

    def test_plot_kind(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, kind="bar")
        nt.assert_equal(p.kind, "bar")

        p = lm._DiscretePlotter("x", "y", data=self.df, kind="point")
        nt.assert_equal(p.kind, "point")

        p = lm._DiscretePlotter("x", "y", data=self.df, kind="box")
        nt.assert_equal(p.kind, "box")

        p = lm._DiscretePlotter("x", data=self.df, kind="auto")
        nt.assert_equal(p.kind, "bar")

        p = lm._DiscretePlotter("x", np.ones(len(self.df)),
                                data=self.df, kind="auto")
        nt.assert_equal(p.kind, "point")

        with nt.assert_raises(ValueError):
            p = lm._DiscretePlotter("x", "y", data=self.df, kind="dino")

    def test_positions(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, kind="bar")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [0])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, kind="bar")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [-.2, .2])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, kind="box")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [-.2, .2])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, kind="point")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [0, 0])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df,
                                kind="point", dodge=.4)
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [-.2, .2])

    def test_estimate_data(self):

        p = lm._DiscretePlotter("x", "y", data=self.df)
        nt.assert_equal(len(list(p.estimate_data)), 1)
        pos, height, ci = next(p.estimate_data)

        npt.assert_array_equal(pos, [0, 1, 2])

        height_want = self.df.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        get_cis = lambda x: moss.ci(moss.bootstrap(x, random_seed=0), 95)
        ci_want = np.array(self.df.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

        p = lm._DiscretePlotter("x", "y", "g", data=self.df)
        nt.assert_equal(len(list(p.estimate_data)), 2)
        data_gen = p.estimate_data

        first_hue = self.df[self.df.g == "x"]
        pos, height, ci = next(data_gen)

        npt.assert_array_equal(pos, [-.2, .8, 1.8])

        height_want = first_hue.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        ci_want = np.array(first_hue.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

        second_hue = self.df[self.df.g == "y"]
        pos, height, ci = next(data_gen)

        npt.assert_array_equal(pos, [.2, 1.2, 2.2])

        height_want = second_hue.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        ci_want = np.array(second_hue.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

    def test_plot_cis(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, ci=95)
        _, _, ci_big = next(p.estimate_data)
        ci_big = np.diff(ci_big, axis=1)

        p = lm._DiscretePlotter("x", "y", data=self.df, ci=68)
        _, _, ci_wee = next(p.estimate_data)
        ci_wee = np.diff(ci_wee, axis=1)

        npt.assert_array_less(ci_wee, ci_big)

    def test_plot_units(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, units="u")
        _, _, ci_big = next(p.estimate_data)
        ci_big = np.diff(ci_big, axis=1)

        p = lm._DiscretePlotter("x", "y", data=self.df)
        _, _, ci_wee = next(p.estimate_data)
        ci_wee = np.diff(ci_wee, axis=1)

        npt.assert_array_less(ci_wee, ci_big)

    def test_annotations(self):

        f, ax = plt.subplots()
        p = lm._DiscretePlotter("x", "y", "g", data=self.df)
        p.plot(ax)
        nt.assert_equal(ax.get_xlabel(), "x")
        nt.assert_equal(ax.get_ylabel(), "y")
        nt.assert_equal(ax.legend_.get_title().get_text(), "g")


class TestDiscretePlots(object):

    rs = np.random.RandomState(341)
    df = pd.DataFrame(dict(x=np.repeat(list("abc"), 30),
                           y=rs.randn(90),
                           g=np.tile(list("xy"), 45),
                           u=np.tile(np.arange(6), 15)))
    bw_err = rs.randn(6)[df.u.values]
    df.y += bw_err

    def test_barplot(self):

        f, ax = plt.subplots()
        lm.barplot("x", "y", data=self.df, hline=None, ax=ax)

        nt.assert_equal(len(ax.patches), 3)
        nt.assert_equal(len(ax.lines), 3)

        f, ax = plt.subplots()
        lm.barplot("x", "y", data=self.df, palette="husl",
                   hline=None, ax=ax)

        nt.assert_equal(len(ax.patches), 3)
        nt.assert_equal(len(ax.lines), 3)
        bar_colors = np.array([el.get_facecolor() for el in ax.patches])
        npt.assert_array_equal(color_palette("husl", 3), bar_colors[:, :3])

        plt.close("all")

    def test_bar_data(self):

        f, ax = plt.subplots()
        lm.barplot("x", "y", data=self.df, hline=None, ax=ax)

        nt.assert_equal(len(ax.patches), 3)
        nt.assert_equal(len(ax.lines), 3)

        f, ax = plt.subplots()
        lm.barplot("x", "y", data=self.df, palette="husl",
                   hline=None, ax=ax)

        nt.assert_equal(len(ax.patches), 3)
        nt.assert_equal(len(ax.lines), 3)
        bar_colors = np.array([el.get_facecolor() for el in ax.patches])
        npt.assert_array_equal(color_palette("husl", 3), bar_colors[:, :3])

        plt.close("all")
