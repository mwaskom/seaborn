"""Tests for plotting utilities."""
import warnings
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nose
import nose.tools as nt
from nose.tools import assert_equal, raises
import numpy.testing as npt
import pandas.util.testing as pdt

from distutils.version import LooseVersion
pandas_has_categoricals = LooseVersion(pd.__version__) >= "0.15"

from pandas.util.testing import network
from ..utils import get_dataset_names, load_dataset

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .. import utils, rcmod


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
    npt.assert_array_equal(actual_errsize, test_errsize)


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

    offset = 10
    original_position = ("outward", 0)
    offset_position = ("outward", offset)

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

    def test_despine_with_offset(self):
        f, ax = plt.subplots()

        for side in self.sides:
            nt.assert_equal(ax.spines[side].get_position(),
                            self.original_position)

        utils.despine(ax=ax, offset=self.offset)

        for side in self.sides:
            is_visible = ax.spines[side].get_visible()
            new_position = ax.spines[side].get_position()
            if is_visible:
                nt.assert_equal(new_position, self.offset_position)
            else:
                nt.assert_equal(new_position, self.original_position)

        plt.close("all")

    def test_despine_with_offset_specific_axes(self):
        f, (ax1, ax2) = plt.subplots(2, 1)

        utils.despine(offset=self.offset, ax=ax2)

        for side in self.sides:
            nt.assert_equal(ax1.spines[side].get_position(),
                            self.original_position)
            if ax2.spines[side].get_visible():
                nt.assert_equal(ax2.spines[side].get_position(),
                                self.offset_position)
            else:
                nt.assert_equal(ax2.spines[side].get_position(),
                                self.original_position)
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

    def test_despine_trim_inverted(self):

        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_ylim(.85, 3.15)
        ax.invert_yaxis()

        utils.despine(trim=True)
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            nt.assert_equal(bounds, (1, 3))

        plt.close("all")

    def test_despine_trim_noticks(self):

        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_yticks([])
        utils.despine(trim=True)
        nt.assert_equal(ax.get_yticks().size, 0)

    def test_offset_spines_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=UserWarning)

            f, ax = plt.subplots()
            utils.offset_spines(offset=self.offset)
            nt.assert_true('deprecated' in str(w[0].message))
            nt.assert_true(issubclass(w[0].category, UserWarning))

        plt.close('all')

    def test_offset_spines(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=UserWarning)
            f, ax = plt.subplots()

            for side in self.sides:
                nt.assert_equal(ax.spines[side].get_position(),
                                self.original_position)

            utils.offset_spines(offset=self.offset)

            for side in self.sides:
                nt.assert_equal(ax.spines[side].get_position(),
                                self.offset_position)

        plt.close("all")

    def test_offset_spines_specific_axes(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=UserWarning)
            f, (ax1, ax2) = plt.subplots(2, 1)

            utils.offset_spines(offset=self.offset, ax=ax2)

            for side in self.sides:
                nt.assert_equal(ax1.spines[side].get_position(),
                                self.original_position)
                nt.assert_equal(ax2.spines[side].get_position(),
                                self.offset_position)
        plt.close("all")


def test_ticklabels_overlap():

    rcmod.set()
    f, ax = plt.subplots(figsize=(2, 2))
    f.tight_layout()  # This gets the Agg renderer working

    assert not utils.axis_ticklabels_overlap(ax.get_xticklabels())

    big_strings = "abcdefgh", "ijklmnop"
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(big_strings)

    assert utils.axis_ticklabels_overlap(ax.get_xticklabels())

    x, y = utils.axes_ticklabels_overlap(ax)
    assert x
    assert not y


def test_categorical_order():

    x = ["a", "c", "c", "b", "a", "d"]
    order = ["a", "b", "c", "d"]

    out = utils.categorical_order(x)
    nt.assert_equal(out, ["a", "c", "b", "d"])

    out = utils.categorical_order(x, order)
    nt.assert_equal(out, order)

    out = utils.categorical_order(x, ["b", "a"])
    nt.assert_equal(out, ["b", "a"])

    out = utils.categorical_order(np.array(x))
    nt.assert_equal(out, ["a", "c", "b", "d"])

    out = utils.categorical_order(pd.Series(x))
    nt.assert_equal(out, ["a", "c", "b", "d"])

    if pandas_has_categoricals:
        x = pd.Categorical(x, order)
        out = utils.categorical_order(x)
        nt.assert_equal(out, list(x.categories))

        x = pd.Series(x)
        out = utils.categorical_order(x)
        nt.assert_equal(out, list(x.cat.categories))

        out = utils.categorical_order(x, ["b", "a"])
        nt.assert_equal(out, ["b", "a"])

    x = ["a", np.nan, "c", "c", "b", "a", "d"]
    out = utils.categorical_order(x)
    nt.assert_equal(out, ["a", "c", "b", "d"])


if LooseVersion(pd.__version__) >= "0.15":

    def check_load_dataset(name):
        ds = load_dataset(name, cache=False)
        assert(isinstance(ds, pd.DataFrame))

    def check_load_cached_dataset(name):
        # Test the cacheing using a temporary file.
        # With Python 3.2+, we could use the tempfile.TemporaryDirectory()
        # context manager instead of this try...finally statement
        tmpdir = tempfile.mkdtemp()
        try:
            # download and cache
            ds = load_dataset(name, cache=True, data_home=tmpdir)

            # use cached version
            ds2 = load_dataset(name, cache=True, data_home=tmpdir)
            pdt.assert_frame_equal(ds, ds2)

        finally:
            shutil.rmtree(tmpdir)

    @network(url="https://github.com/mwaskom/seaborn-data")
    def test_get_dataset_names():
        if not BeautifulSoup:
            raise nose.SkipTest("No BeautifulSoup available for parsing html")
        names = get_dataset_names()
        assert(len(names) > 0)
        assert(u"titanic" in names)

    @network(url="https://github.com/mwaskom/seaborn-data")
    def test_load_datasets():
        if not BeautifulSoup:
            raise nose.SkipTest("No BeautifulSoup available for parsing html")

        # Heavy test to verify that we can load all available datasets
        for name in get_dataset_names():
            # unfortunately @network somehow obscures this generator so it
            # does not get in effect, so we need to call explicitly
            # yield check_load_dataset, name
            check_load_dataset(name)

    @network(url="https://github.com/mwaskom/seaborn-data")
    def test_load_cached_datasets():
        if not BeautifulSoup:
            raise nose.SkipTest("No BeautifulSoup available for parsing html")

        # Heavy test to verify that we can load all available datasets
        for name in get_dataset_names():
            # unfortunately @network somehow obscures this generator so it
            # does not get in effect, so we need to call explicitly
            # yield check_load_dataset, name
            check_load_cached_dataset(name)
