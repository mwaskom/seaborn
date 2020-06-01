"""Tests for plotting utilities."""
import tempfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

import pytest
from numpy.testing import (
    assert_array_equal,
)
from pandas.testing import (
    assert_series_equal,
    assert_frame_equal,
)

from distutils.version import LooseVersion

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .. import utils, rcmod
from ..utils import (
    get_dataset_names,
    get_color_cycle,
    remove_na,
    load_dataset,
    _network,
)


a_norm = np.random.randn(100)


def test_pmf_hist_basics():
    """Test the function to return barplot args for pmf hist."""
    with pytest.warns(FutureWarning):
        out = utils.pmf_hist(a_norm)
    assert len(out) == 3
    x, h, w = out
    assert len(x) == len(h)

    # Test simple case
    a = np.arange(10)
    with pytest.warns(FutureWarning):
        x, h, w = utils.pmf_hist(a, 10)
    assert np.all(h == h[0])

    # Test width
    with pytest.warns(FutureWarning):
        x, h, w = utils.pmf_hist(a_norm)
    assert x[1] - x[0] == w

    # Test normalization
    with pytest.warns(FutureWarning):
        x, h, w = utils.pmf_hist(a_norm)
    assert sum(h) == pytest.approx(1)
    assert h.max() <= 1

    # Test bins
    with pytest.warns(FutureWarning):
        x, h, w = utils.pmf_hist(a_norm, 20)
    assert len(x) == 20


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
    assert out1 == (.75, .25, .25)

    out2 = utils.desaturate("#00FF00", .5)
    assert out2 == (.25, .75, .25)

    out3 = utils.desaturate((0, 0, 1), .5)
    assert out3 == (.25, .25, .75)

    out4 = utils.desaturate("red", .5)
    assert out4 == (.75, .25, .25)


def test_desaturation_prop():
    """Test that pct outside of [0, 1] raises exception."""
    with pytest.raises(ValueError):
        utils.desaturate("blue", 50)


def test_saturate():
    """Test performance of saturation function."""
    out = utils.saturate((.75, .25, .25))
    assert out == (1, 0, 0)


@pytest.mark.parametrize(
    "p,annot", [(.0001, "***"), (.001, "**"), (.01, "*"), (.09, "."), (1, "")]
)
def test_sig_stars(p, annot):
    """Test the sig stars function."""
    with pytest.warns(FutureWarning):
        stars = utils.sig_stars(p)
        assert stars == annot


def test_iqr():
    """Test the IQR function."""
    a = np.arange(5)
    with pytest.warns(FutureWarning):
        iqr = utils.iqr(a)
    assert iqr == 2


@pytest.mark.parametrize(
    "s,exp",
    [
        ("a", "a"),
        ("abc", "abc"),
        (b"a", "a"),
        (b"abc", "abc"),
        (bytearray("abc", "utf-8"), "abc"),
        (bytearray(), ""),
        (1, "1"),
        (0, "0"),
        ([], str([])),
    ],
)
def test_to_utf8(s, exp):
    """Test the to_utf8 function: object to string"""
    u = utils.to_utf8(s)
    assert type(u) == str
    assert u == exp


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
            assert ax.spines[side].get_visible()

        utils.despine()
        for side in self.outer_sides:
            assert ~ax.spines[side].get_visible()
        for side in self.inner_sides:
            assert ax.spines[side].get_visible()

        utils.despine(**dict(zip(self.sides, [True] * 4)))
        for side in self.sides:
            assert ~ax.spines[side].get_visible()

    def test_despine_specific_axes(self):
        f, (ax1, ax2) = plt.subplots(2, 1)

        utils.despine(ax=ax2)

        for side in self.sides:
            assert ax1.spines[side].get_visible()

        for side in self.outer_sides:
            assert ~ax2.spines[side].get_visible()
        for side in self.inner_sides:
            assert ax2.spines[side].get_visible()

    def test_despine_with_offset(self):
        f, ax = plt.subplots()

        for side in self.sides:
            pos = ax.spines[side].get_position()
            assert pos == self.original_position

        utils.despine(ax=ax, offset=self.offset)

        for side in self.sides:
            is_visible = ax.spines[side].get_visible()
            new_position = ax.spines[side].get_position()
            if is_visible:
                assert new_position == self.offset_position
            else:
                assert new_position == self.original_position

    def test_despine_side_specific_offset(self):

        f, ax = plt.subplots()
        utils.despine(ax=ax, offset=dict(left=self.offset))

        for side in self.sides:
            is_visible = ax.spines[side].get_visible()
            new_position = ax.spines[side].get_position()
            if is_visible and side == "left":
                assert new_position == self.offset_position
            else:
                assert new_position == self.original_position

    def test_despine_with_offset_specific_axes(self):
        f, (ax1, ax2) = plt.subplots(2, 1)

        utils.despine(offset=self.offset, ax=ax2)

        for side in self.sides:
            pos1 = ax1.spines[side].get_position()
            pos2 = ax2.spines[side].get_position()
            assert pos1 == self.original_position
            if ax2.spines[side].get_visible():
                assert pos2 == self.offset_position
            else:
                assert pos2 == self.original_position

    def test_despine_trim_spines(self):

        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xlim(.75, 3.25)

        utils.despine(trim=True)
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            assert bounds == (1, 3)

    def test_despine_trim_inverted(self):

        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_ylim(.85, 3.15)
        ax.invert_yaxis()

        utils.despine(trim=True)
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            assert bounds == (1, 3)

    def test_despine_trim_noticks(self):

        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_yticks([])
        utils.despine(trim=True)
        assert ax.get_yticks().size == 0

    def test_despine_trim_categorical(self):

        f, ax = plt.subplots()
        ax.plot(["a", "b", "c"], [1, 2, 3])

        utils.despine(trim=True)

        bounds = ax.spines["left"].get_bounds()
        assert bounds == (1, 3)

        bounds = ax.spines["bottom"].get_bounds()
        assert bounds == (0, 2)

    def test_despine_moved_ticks(self):

        f, ax = plt.subplots()
        for t in ax.yaxis.majorTicks:
            t.tick1line.set_visible(True)
        utils.despine(ax=ax, left=True, right=False)
        for t in ax.yaxis.majorTicks:
            assert t.tick2line.get_visible()
        plt.close(f)

        f, ax = plt.subplots()
        for t in ax.yaxis.majorTicks:
            t.tick1line.set_visible(False)
        utils.despine(ax=ax, left=True, right=False)
        for t in ax.yaxis.majorTicks:
            assert not t.tick2line.get_visible()
        plt.close(f)

        f, ax = plt.subplots()
        for t in ax.xaxis.majorTicks:
            t.tick1line.set_visible(True)
        utils.despine(ax=ax, bottom=True, top=False)
        for t in ax.xaxis.majorTicks:
            assert t.tick2line.get_visible()
        plt.close(f)

        f, ax = plt.subplots()
        for t in ax.xaxis.majorTicks:
            t.tick1line.set_visible(False)
        utils.despine(ax=ax, bottom=True, top=False)
        for t in ax.xaxis.majorTicks:
            assert not t.tick2line.get_visible()
        plt.close(f)


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


def test_locator_to_legend_entries():

    locator = mpl.ticker.MaxNLocator(nbins=3)
    limits = (0.09, 0.4)
    levels, str_levels = utils.locator_to_legend_entries(
        locator, limits, float
    )
    assert str_levels == ["0.00", "0.15", "0.30", "0.45"]

    limits = (0.8, 0.9)
    levels, str_levels = utils.locator_to_legend_entries(
        locator, limits, float
    )
    assert str_levels == ["0.80", "0.84", "0.88", "0.92"]

    limits = (1, 6)
    levels, str_levels = utils.locator_to_legend_entries(locator, limits, int)
    assert str_levels == ["0", "2", "4", "6"]

    locator = mpl.ticker.LogLocator(numticks=3)
    limits = (5, 1425)
    levels, str_levels = utils.locator_to_legend_entries(locator, limits, int)
    if LooseVersion(mpl.__version__) >= "3.1":
        assert str_levels == ['0', '1', '100', '10000', '1e+06']

    limits = (0.00003, 0.02)
    levels, str_levels = utils.locator_to_legend_entries(
        locator, limits, float
    )
    if LooseVersion(mpl.__version__) >= "3.1":
        assert str_levels == ['1e-07', '1e-05', '1e-03', '1e-01', '10']


def check_load_dataset(name):
    ds = load_dataset(name, cache=False)
    assert(isinstance(ds, pd.DataFrame))


def check_load_cached_dataset(name):
    # Test the cacheing using a temporary file.
    with tempfile.TemporaryDirectory() as tmpdir:
        # download and cache
        ds = load_dataset(name, cache=True, data_home=tmpdir)

        # use cached version
        ds2 = load_dataset(name, cache=True, data_home=tmpdir)
        assert_frame_equal(ds, ds2)


@_network(url="https://github.com/mwaskom/seaborn-data")
def test_get_dataset_names():
    if not BeautifulSoup:
        pytest.skip("No BeautifulSoup available for parsing html")
    names = get_dataset_names()
    assert(len(names) > 0)
    assert("titanic" in names)


@_network(url="https://github.com/mwaskom/seaborn-data")
def test_load_datasets():
    if not BeautifulSoup:
        raise pytest.skip("No BeautifulSoup available for parsing html")

    # Heavy test to verify that we can load all available datasets
    for name in get_dataset_names():
        # unfortunately @network somehow obscures this generator so it
        # does not get in effect, so we need to call explicitly
        # yield check_load_dataset, name
        check_load_dataset(name)


@_network(url="https://github.com/mwaskom/seaborn-data")
def test_load_cached_datasets():
    if not BeautifulSoup:
        raise pytest.skip("No BeautifulSoup available for parsing html")

    # Heavy test to verify that we can load all available datasets
    for name in get_dataset_names():
        # unfortunately @network somehow obscures this generator so it
        # does not get in effect, so we need to call explicitly
        # yield check_load_dataset, name
        check_load_cached_dataset(name)


def test_relative_luminance():
    """Test relative luminance."""
    out1 = utils.relative_luminance("white")
    assert out1 == 1

    out2 = utils.relative_luminance("#000000")
    assert out2 == 0

    out3 = utils.relative_luminance((.25, .5, .75))
    assert out3 == pytest.approx(0.201624536)

    rgbs = mpl.cm.RdBu(np.linspace(0, 1, 10))
    lums1 = [utils.relative_luminance(rgb) for rgb in rgbs]
    lums2 = utils.relative_luminance(rgbs)

    for lum1, lum2 in zip(lums1, lums2):
        assert lum1 == pytest.approx(lum2)


@pytest.mark.parametrize(
    "cycler,result",
    [
        (cycler(color=["y"]), ["y"]),
        (cycler(color=["k"]), ["k"]),
        (cycler(color=["k", "y"]), ["k", "y"]),
        (cycler(color=["y", "k"]), ["y", "k"]),
        (cycler(color=["b", "r"]), ["b", "r"]),
        (cycler(color=["r", "b"]), ["r", "b"]),
        (cycler(lw=[1, 2]), [".15"]),  # no color in cycle
    ],
)
def test_get_color_cycle(cycler, result):
    with mpl.rc_context(rc={"axes.prop_cycle": cycler}):
        assert get_color_cycle() == result


def test_remove_na():

    a_array = np.array([1, 2, np.nan, 3])
    a_array_rm = remove_na(a_array)
    assert_array_equal(a_array_rm, np.array([1, 2, 3]))

    a_series = pd.Series([1, 2, np.nan, 3])
    a_series_rm = remove_na(a_series)
    assert_series_equal(a_series_rm, pd.Series([1., 2, 3], [0, 1, 3]))
