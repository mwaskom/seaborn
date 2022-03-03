from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal

from seaborn._marks.base import Mark, Feature


class TestFeature:

    def mark(self, **features):

        @dataclass
        class MockMark(Mark):
            linewidth: float = Feature(rc="lines.linewidth")
            pointsize: float = Feature(4)
            color: str = Feature("C0")
            fillcolor: str = Feature(depend="color")
            alpha: float = Feature(1)
            fillalpha: float = Feature(depend="alpha")

        m = MockMark(**features)
        return m

    def test_repr(self):

        assert str(Feature(.5)) == "<0.5>"
        assert str(Feature("CO")) == "<'CO'>"
        assert str(Feature(rc="lines.linewidth")) == "<rc:lines.linewidth>"
        assert str(Feature(depend="color")) == "<depend:color>"

    def test_input_checks(self):

        with pytest.raises(AssertionError):
            Feature(rc="bogus.parameter")
        with pytest.raises(AssertionError):
            Feature(depend="nonexistent_feature")

    def test_value(self):

        val = 3
        m = self.mark(linewidth=val)
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_default(self):

        val = 3
        m = self.mark(linewidth=Feature(val))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_rcparam(self):

        param = "lines.linewidth"
        val = mpl.rcParams[param]

        m = self.mark(linewidth=Feature(rc=param))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_depends(self):

        val = 2
        df = pd.DataFrame(index=pd.RangeIndex(10))

        m = self.mark(pointsize=Feature(val), linewidth=Feature(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

        m = self.mark(pointsize=val * 2, linewidth=Feature(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val * 2
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val * 2))

    def test_mapped(self):

        values = {"a": 1, "b": 2, "c": 3}

        def f(x):
            return np.array([values[x_i] for x_i in x])

        m = self.mark(linewidth=Feature(2))
        m.scales = {"linewidth": f}

        assert m._resolve({"linewidth": "c"}, "linewidth") == 3

        df = pd.DataFrame({"linewidth": ["a", "b", "c"]})
        assert_array_equal(m._resolve(df, "linewidth"), np.array([1, 2, 3], float))

    def test_color(self):

        c, a = "C1", .5
        m = self.mark(color=c, alpha=a)

        assert m._resolve_color({}) == mpl.colors.to_rgba(c, a)

        df = pd.DataFrame(index=pd.RangeIndex(10))
        cs = [c] * len(df)
        assert_array_equal(m._resolve_color(df), mpl.colors.to_rgba_array(cs, a))

    def test_color_mapped_alpha(self):

        c = "r"
        values = {"a": .2, "b": .5, "c": .8}

        m = self.mark(color=c, alpha=Feature(1))
        m.scales = {"alpha": lambda s: np.array([values[s_i] for s_i in s])}

        assert m._resolve_color({"alpha": "b"}) == mpl.colors.to_rgba(c, .5)

        df = pd.DataFrame({"alpha": list(values.keys())})

        # Do this in two steps for mpl 3.2 compat
        expected = mpl.colors.to_rgba_array([c] * len(df))
        expected[:, 3] = list(values.values())

        assert_array_equal(m._resolve_color(df), expected)

    def test_color_scaled_as_strings(self):

        colors = ["C1", "dodgerblue", "#445566"]
        m = self.mark()
        m.scales = {"color": lambda s: colors}

        actual = m._resolve_color({"color": pd.Series(["a", "b", "c"])})
        expected = mpl.colors.to_rgba_array(colors)
        assert_array_equal(actual, expected)

    def test_fillcolor(self):

        c, a = "green", .8
        fa = .2
        m = self.mark(
            color=c, alpha=a,
            fillcolor=Feature(depend="color"), fillalpha=Feature(fa),
        )

        assert m._resolve_color({}) == mpl.colors.to_rgba(c, a)
        assert m._resolve_color({}, "fill") == mpl.colors.to_rgba(c, fa)

        df = pd.DataFrame(index=pd.RangeIndex(10))
        cs = [c] * len(df)
        assert_array_equal(m._resolve_color(df), mpl.colors.to_rgba_array(cs, a))
        assert_array_equal(
            m._resolve_color(df, "fill"), mpl.colors.to_rgba_array(cs, fa)
        )
