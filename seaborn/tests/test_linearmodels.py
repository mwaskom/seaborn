import numpy as np
import statsmodels.api as sm
import pandas as pd
import moss

import nose.tools as nt
import numpy.testing as npt

from .. import linearmodels as lm
from ..utils import color_palette

rs = np.random.RandomState(0)


class TestRegPlot(object):
    """Test internal functions that perform computation for regplot()."""
    x = rs.randn(50)
    x_discrete = np.repeat([0, 1], 25)
    y = 2 + 1.5 * 2 + rs.randn(50)
    grid = np.linspace(-3, 3, 30)
    n_boot = 20
    ci = 95
    bins_numeric = 3
    bins_given = [-1, 0, 1]
    units = None

    def test_regress_fast(self):
        """Validate fast regression fit and bootstrap."""

        # Fit with the "fast" function, which just does linear algebra
        fast = lm._regress_fast(self.grid, self.x, self.y, self.units,
                                self.ci, self.n_boot)
        yhat_fast, _ = fast

        # Fit using the statsmodels function with an OLS model
        smod = lm._regress_statsmodels(self.grid, self.x, self.y, self.units,
                                       sm.OLS, self.ci, self.n_boot)
        yhat_smod, _ = smod

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_fast, yhat_smod)

    def test_regress_poly(self):
        """Validate polyfit-based regression fit and bootstrap."""

        # Fit an first-order polynomial
        poly = lm._regress_poly(self.grid, self.x, self.y, self.units, 1,
                                self.ci, self.n_boot)
        yhat_poly, _ = poly

        # Fit using the statsmodels function with an OLS model
        smod = lm._regress_statsmodels(self.grid, self.x, self.y, self.units,
                                       sm.OLS, self.ci, self.n_boot)
        yhat_smod, _ = smod

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_poly, yhat_smod)

    def test_regress_n_boot(self):
        """Test correct bootstrap size for internal regression functions."""
        args = self.grid, self.x, self.y, self.units

        # Fast (linear algebra) version
        fast = lm._regress_fast(*args, ci=self.ci, n_boot=self.n_boot)
        _, boots_fast = fast
        npt.assert_equal(boots_fast.shape, (self.n_boot, self.grid.size))

        # Slower (np.polyfit) version
        poly = lm._regress_poly(*args, order=1, ci=self.ci, n_boot=self.n_boot)
        _, boots_poly = poly
        npt.assert_equal(boots_poly.shape, (self.n_boot, self.grid.size))

        # Slowest (statsmodels) version
        smod = lm._regress_statsmodels(*args, model=sm.OLS,
                                       ci=self.ci, n_boot=self.n_boot)
        _, boots_smod = smod
        npt.assert_equal(boots_smod.shape, (self.n_boot, self.grid.size))

    def test_regress_noboot(self):
        """Test that regression functions return None if not bootstrapping."""
        args = self.grid, self.x, self.y, self.units

        # Fast (linear algebra) version
        fast = lm._regress_fast(*args, ci=None, n_boot=self.n_boot)
        _, boots_fast = fast
        nt.assert_is(boots_fast, None)

        # Slower (np.polyfit) version
        poly = lm._regress_poly(*args, order=1, ci=None, n_boot=self.n_boot)
        _, boots_poly = poly
        nt.assert_is(boots_poly, None)

        # Slowest (statsmodels) version
        smod = lm._regress_statsmodels(*args, model=sm.OLS,
                                       ci=None, n_boot=self.n_boot)
        _, boots_smod = smod
        nt.assert_is(boots_smod, None)

    def test_numeric_bins(self):
        """Test discretizing x into `n` bins."""
        x_binned, bins = lm._bin_predictor(self.x, self.bins_numeric)
        npt.assert_equal(len(bins), self.bins_numeric)
        npt.assert_array_equal(np.unique(x_binned), bins)

    def test_provided_bins(self):
        """Test discretizing x into provided bins."""
        x_binned, bins = lm._bin_predictor(self.x, self.bins_given)
        npt.assert_array_equal(np.unique(x_binned), self.bins_given)

    def test_binning(self):
        """Test that the binning actually works."""
        x_binned, bins = lm._bin_predictor(self.x, self.bins_given)
        nt.assert_greater(self.x[x_binned == 0].min(),
                          self.x[x_binned == -1].max())
        nt.assert_greater(self.x[x_binned == 1].min(),
                          self.x[x_binned == 0].max())

    def test_point_est(self):
        """Test statistic estimation for discrete input data."""
        x_vals, points, cis = lm._point_est(self.x_discrete, self.y, np.mean,
                                            self.ci, self.units, self.n_boot)

        npt.assert_array_equal(x_vals, sorted(np.unique(self.x_discrete)))
        nt.assert_equal(len(points), np.unique(self.x_discrete).size)
        nt.assert_equal(np.shape(cis), (np.unique(self.x_discrete).size, 2))

    def test_point_ci(self):
        """Test the confidence interval in the point estimate function."""
        _, _, big_cis = lm._point_est(self.x_discrete, self.y,
                                      np.mean, 95, self.units, self.n_boot)
        _, _, wee_cis = lm._point_est(self.x_discrete, self.y,
                                      np.mean, 15, self.units, self.n_boot)
        npt.assert_array_less(np.diff(wee_cis), np.diff(big_cis))

        _, _, no_cis = lm._point_est(self.x_discrete, self.y,
                                     np.mean, None, self.units, self.n_boot)
        npt.assert_array_equal(no_cis, [None] * len(no_cis))


class TestDiscretePlotter(object):

    rs = np.random.RandomState(341)
    df = pd.DataFrame(dict(x=np.repeat(list("abc"), 30),
                           y=rs.randn(90),
                           g=np.tile(list("xy"), 45),
                           u=np.tile(list("123456"), 15)))

    def test_variables_from_frame(self):

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, units="u")

        npt.assert_array_equal(p.x, self.df.x)
        npt.assert_array_equal(p.y, self.df.y)
        npt.assert_array_equal(p.hue, self.df.g)
        npt.assert_array_equal(p.units, self.df.u)
        npt.assert_array_equal(p.data, self.df)

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
        npt.assert_array_equal(p.data, self.df)

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

    def test_variables_hue_as_x(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, palette="husl")
        npt.assert_array_equal(p.hue, self.df.x)
        npt.assert_array_equal(p.hue_order, np.sort(self.df.x.unique()))

    def test_palette(self):

        p = lm._DiscretePlotter("x", "y", data=self.df)
        nt.assert_equal(p.palette, [color_palette()[0]] * 3)

        p = lm._DiscretePlotter("x", "y", data=self.df, color="green")
        nt.assert_equal(p.palette, ["green"] * 3)

        p = lm._DiscretePlotter("x", "y", data=self.df, palette="husl")
        nt.assert_equal(p.palette, color_palette("husl", 3))

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

        p = lm._DiscretePlotter("x", data=self.df, kind="auto")
        nt.assert_equal(p.kind, "bar")

        p = lm._DiscretePlotter("x", np.ones(len(self.df)),
                                data=self.df, kind="auto")
        nt.assert_equal(p.kind, "point")

    def test_positions(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, kind="bar")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [0])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, kind="bar")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [-.2, .2])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df, kind="point")
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [0, 0])

        p = lm._DiscretePlotter("x", "y", "g", data=self.df,
                                kind="point", dodge=.4)
        npt.assert_array_equal(p.positions, [0, 1, 2])
        npt.assert_array_equal(p.offset, [-.2, .2])

    def test_plot_data(self):

        p = lm._DiscretePlotter("x", "y", data=self.df)
        nt.assert_equal(len(list(p.plot_data)), 1)
        pos, height, ci = p.plot_data.next()

        npt.assert_array_equal(pos, [0, 1, 2])

        height_want = self.df.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        get_cis = lambda x: moss.ci(moss.bootstrap(x, random_seed=0), 95)
        ci_want = np.array(self.df.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

        p = lm._DiscretePlotter("x", "y", "g", data=self.df)
        nt.assert_equal(len(list(p.plot_data)), 2)
        data_gen = p.plot_data

        first_hue = self.df[self.df.g == "x"]
        pos, height, ci = data_gen.next()

        npt.assert_array_equal(pos, [-.2, .8, 1.8])

        height_want = first_hue.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        ci_want = np.array(first_hue.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

        second_hue = self.df[self.df.g == "y"]
        pos, height, ci = data_gen.next()

        npt.assert_array_equal(pos, [.2, 1.2, 2.2])

        height_want = second_hue.groupby("x").y.mean()
        npt.assert_array_equal(height, height_want)

        ci_want = np.array(second_hue.groupby("x").y.apply(get_cis).tolist())
        npt.assert_array_almost_equal(np.squeeze(ci), ci_want, 1)

    def test_plot_cis(self):

        p = lm._DiscretePlotter("x", "y", data=self.df, ci=95)
        _, _, ci_big = p.plot_data.next()
        ci_big = np.diff(ci_big, axis=1)

        p = lm._DiscretePlotter("x", "y", data=self.df, ci=68)
        _, _, ci_wee = p.plot_data.next()
        ci_wee = np.diff(ci_wee, axis=1)

        npt.assert_array_less(ci_wee, ci_big)

    def test_plot_units(self):
        p = lm._DiscretePlotter("x", "y", data=self.df, units="u")
        _, _, ci_big = p.plot_data.next()
        ci_big = np.diff(ci_big, axis=1)

        p = lm._DiscretePlotter("x", "y", data=self.df)
        _, _, ci_wee = p.plot_data.next()
        ci_wee = np.diff(ci_wee, axis=1)

        npt.assert_array_less(ci_wee, ci_big)
