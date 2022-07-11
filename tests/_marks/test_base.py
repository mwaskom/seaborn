from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from numpy.testing import assert_array_equal

from seaborn._marks.base import Mark, Mappable, resolve_color


class TestMappable:

    def mark(self, **features):

        @dataclass
        class MockMark(Mark):
            linewidth: float = Mappable(rc="lines.linewidth")
            pointsize: float = Mappable(4)
            color: str = Mappable("C0")
            fillcolor: str = Mappable(depend="color")
            alpha: float = Mappable(1)
            fillalpha: float = Mappable(depend="alpha")

        m = MockMark(**features)
        return m

    def test_repr(self):

        assert str(Mappable(.5)) == "<0.5>"
        assert str(Mappable("CO")) == "<'CO'>"
        assert str(Mappable(rc="lines.linewidth")) == "<rc:lines.linewidth>"
        assert str(Mappable(depend="color")) == "<depend:color>"
        assert str(Mappable(auto=True)) == "<auto>"

    def test_input_checks(self):

        with pytest.raises(AssertionError):
            Mappable(rc="bogus.parameter")
        with pytest.raises(AssertionError):
            Mappable(depend="nonexistent_feature")

    def test_value(self):

        val = 3
        m = self.mark(linewidth=val)
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_default(self):

        val = 3
        m = self.mark(linewidth=Mappable(val))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_rcparam(self):

        param = "lines.linewidth"
        val = mpl.rcParams[param]

        m = self.mark(linewidth=Mappable(rc=param))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_depends(self):

        val = 2
        df = pd.DataFrame(index=pd.RangeIndex(10))

        m = self.mark(pointsize=Mappable(val), linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

        m = self.mark(pointsize=val * 2, linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val * 2
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val * 2))

    def test_mapped(self):

        values = {"a": 1, "b": 2, "c": 3}

        def f(x):
            return np.array([values[x_i] for x_i in x])

        m = self.mark(linewidth=Mappable(2))
        scales = {"linewidth": f}

        assert m._resolve({"linewidth": "c"}, "linewidth", scales) == 3

        df = pd.DataFrame({"linewidth": ["a", "b", "c"]})
        expected = np.array([1, 2, 3], float)
        assert_array_equal(m._resolve(df, "linewidth", scales), expected)

    def test_color(self):

        c, a = "C1", .5
        m = self.mark(color=c, alpha=a)

        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)

        df = pd.DataFrame(index=pd.RangeIndex(10))
        cs = [c] * len(df)
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))

    def test_color_mapped_alpha(self):

        c = "r"
        values = {"a": .2, "b": .5, "c": .8}

        m = self.mark(color=c, alpha=Mappable(1))
        scales = {"alpha": lambda s: np.array([values[s_i] for s_i in s])}

        assert resolve_color(m, {"alpha": "b"}, "", scales) == mpl.colors.to_rgba(c, .5)

        df = pd.DataFrame({"alpha": list(values.keys())})

        # Do this in two steps for mpl 3.2 compat
        expected = mpl.colors.to_rgba_array([c] * len(df))
        expected[:, 3] = list(values.values())

        assert_array_equal(resolve_color(m, df, "", scales), expected)

    def test_color_scaled_as_strings(self):

        colors = ["C1", "dodgerblue", "#445566"]
        m = self.mark()
        scales = {"color": lambda s: colors}

        actual = resolve_color(m, {"color": pd.Series(["a", "b", "c"])}, "", scales)
        expected = mpl.colors.to_rgba_array(colors)
        assert_array_equal(actual, expected)

    def test_fillcolor(self):

        c, a = "green", .8
        fa = .2
        m = self.mark(
            color=c, alpha=a,
            fillcolor=Mappable(depend="color"), fillalpha=Mappable(fa),
        )

        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)
        assert resolve_color(m, {}, "fill") == mpl.colors.to_rgba(c, fa)

        df = pd.DataFrame(index=pd.RangeIndex(10))
        cs = [c] * len(df)
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))
        assert_array_equal(
            resolve_color(m, df, "fill"), mpl.colors.to_rgba_array(cs, fa)
        )
