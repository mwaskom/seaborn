"""Tests for plotting utilities."""
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal
import nose
import nose.tools as nt
from nose.tools import assert_equal, raises

from .. import utils


a_norm = np.random.randn(100)


def test_pmf_hist_basics():
    """Test the function to return barplot args for pmf hist."""
    out = utils.pmf_hist(a_norm)
    assert_equal(len(out), 3)
    x, h, w = out
    assert_equal(len(x), len(h))

    # Test simple case
    a = np.arange(10)
    x, h, w = utils.pmf_hist(a, 10)
    nose.tools.assert_true(np.all(h == h[0]))


def test_pmf_hist_widths():
    """Test histogram width is correct."""
    x, h, w = utils.pmf_hist(a_norm)
    assert_equal(x[1] - x[0], w)


def test_pmf_hist_normalization():
    """Test that output data behaves like a PMF."""
    x, h, w = utils.pmf_hist(a_norm)
    nose.tools.assert_almost_equal(sum(h), 1)
    nose.tools.assert_less_equal(h.max(), 1)


def test_pmf_hist_bins():
    """Test bin specification."""
    x, h, w = utils.pmf_hist(a_norm, 20)
    assert_equal(len(x), 20)


def test_ci_to_errsize():
    """Test behavior of ci_to_errsize."""
    cis = [[.5, .5],
           [1.25, 1.5]]

    heights = [1, 1.5]

    actual_errsize = np.array([[.5, 1],
                               [.25, 0]])

    test_errsize = utils.ci_to_errsize(cis, heights)
    assert_array_equal(actual_errsize, test_errsize)


def test_desaturate():
    """Test color desaturation."""
    out1 = utils.desaturate("red", .5)
    assert_equal(out1, (.75, .25, .25))

    out2 = utils.desaturate("#00FF00", .5)
    assert_equal(out2, (.25, .75, .25))

    out3 = utils.desaturate((0, 0, 1), .5)
    assert_equal(out3, (.25, .25, .75))

    out4 = utils.desaturate("red", .5)
    assert_equal(out4, (.75, .25, .25))


@raises(ValueError)
def test_desaturation_prop():
    """Test that pct outside of [0, 1] raises exception."""
    utils.desaturate("blue", 50)


def test_saturate():
    """Test performance of saturation function."""
    out = utils.saturate((.75, .25, .25))
    assert_equal(out, (1, 0, 0))


def test_iqr():
    """Test the IQR function."""
    a = np.arange(5)
    iqr = utils.iqr(a)
    assert_equal(iqr, 2)


class TestSpineUtils(object):

    sides = ["left", "right", "bottom", "top"]
    outer_sides = ["top", "right"]
    inner_sides = ["left", "bottom"]

    def test_despine(self):
        f, ax = plt.subplots()
        for side in self.sides:
            nt.assert_true(ax.spines[side].get_visible())

        utils.despine()
        for side in self.outer_sides:
            nt.assert_true(~ax.spines[side].get_visible())
        for side in self.inner_sides:
            nt.assert_true(ax.spines[side].get_visible())

        utils.despine(**dict(zip(self.sides, [True] * 4)))
        for side in self.sides:
            nt.assert_true(~ax.spines[side].get_visible())

        plt.close("all")

    def test_despine_specific_axes(self):
        f, (ax1, ax2) = plt.subplots(2, 1)

        utils.despine(ax=ax2)

        for side in self.sides:
            nt.assert_true(ax1.spines[side].get_visible())

        for side in self.outer_sides:
            nt.assert_true(~ax2.spines[side].get_visible())
        for side in self.inner_sides:
            nt.assert_true(ax2.spines[side].get_visible())

        plt.close("all")

    def test_despine_trim_spines(self):
        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xlim(.75, 3.25)

        utils.despine(trim=True)
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            nt.assert_equal(bounds, (1, 3))

        plt.close("all")

    def test_offset_spines(self):
        f, ax = plt.subplots()

        for side in self.sides:
            nt.assert_equal(ax.spines[side].get_position(), ("outward", 0))

        utils.offset_spines(10)

        for side in self.sides:
            nt.assert_equal(ax.spines[side].get_position(), ("outward", 10))

        plt.close("all")

    def test_offset_spines_specific_axes(self):
        f, (ax1, ax2) = plt.subplots(2, 1)

        utils.offset_spines(10, ax=ax2)

        for side in self.sides:
            nt.assert_equal(ax1.spines[side].get_position(), ("outward", 0))
            nt.assert_equal(ax2.spines[side].get_position(), ("outward", 10))

        plt.close("all")
