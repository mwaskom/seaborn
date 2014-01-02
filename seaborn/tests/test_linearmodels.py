import numpy as np
import statsmodels.api as sm

import nose.tools as nt
import numpy.testing as npt

from .. import linearmodels as lm

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

    def test_regress_fast(self):
        """Validate fast regression fit and bootstrap."""

        # Fit with the "fast" function, which just does linear algebra
        fast = lm._regress_fast(self.grid, self.x, self.y,
                                self.ci, self.n_boot)
        yhat_fast, _ = fast

        # Fit using the statsmodels function with an OLS model
        smod = lm._regress_statsmodels(self.grid, self.x, self.y, sm.OLS,
                                       self.ci, self.n_boot)
        yhat_smod, _ = smod

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_fast, yhat_smod)

    def test_regress_poly(self):
        """Validate polyfit-based regression fit and bootstrap."""

        # Fit an first-order polynomial
        poly = lm._regress_poly(self.grid, self.x, self.y, 1,
                                self.ci, self.n_boot)
        yhat_poly, _ = poly

        # Fit using the statsmodels function with an OLS model
        smod = lm._regress_statsmodels(self.grid, self.x, self.y, sm.OLS,
                                       self.ci, self.n_boot)
        yhat_smod, _ = smod

        # Compare the vector of y_hat values
        npt.assert_array_almost_equal(yhat_poly, yhat_smod)

    def test_regress_n_boot(self):
        """Test correct bootstrap size for internal regression functions."""
        args = self.grid, self.x, self.y

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
        args = self.grid, self.x, self.y

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
                                            self.ci, self.n_boot)

        npt.assert_array_equal(x_vals, sorted(np.unique(self.x_discrete)))
        nt.assert_equal(len(points), np.unique(self.x_discrete).size)
        nt.assert_equal(np.shape(cis), (np.unique(self.x_discrete).size, 2))

    def test_point_ci(self):
        """Test the confidence interval in the point estimate function."""
        _, _, big_cis = lm._point_est(self.x_discrete, self.y,
                                      np.mean, 95, self.n_boot)
        _, _, wee_cis = lm._point_est(self.x_discrete, self.y,
                                      np.mean, 15, self.n_boot)
        npt.assert_array_less(np.diff(wee_cis), np.diff(big_cis))
