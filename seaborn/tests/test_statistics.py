import numpy as np
from scipy import integrate

import pytest
from numpy.testing import assert_array_equal

from .._statistics import (
    KDE,
)


class TestKDE:

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
        assert integrate.trapz(density, support) == pytest.approx(1, abs=1e-5)

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

    def test_bivariate_cumulative(self, rng):

        x, y = rng.normal(0, 3, (2, 50))
        kde = KDE(gridsize=100, cumulative=True)
        density, _ = kde(x, y)

        assert density[0, 0] == pytest.approx(0, abs=1e-2)
        assert density[-1, -1] == pytest.approx(1, abs=1e-2)
