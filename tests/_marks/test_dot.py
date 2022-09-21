from matplotlib.colors import to_rgba, to_rgba_array

import pytest
from numpy.testing import assert_array_equal

from seaborn.palettes import color_palette
from seaborn._core.plot import Plot
from seaborn._marks.dot import Dot, Dots


@pytest.fixture(autouse=True)
def default_palette():
    with color_palette("deep"):
        yield


class DotBase:

    def check_offsets(self, points, x, y):

        offsets = points.get_offsets().T
        assert_array_equal(offsets[0], x)
        assert_array_equal(offsets[1], y)

    def check_colors(self, part, points, colors, alpha=None):

        rgba = to_rgba_array(colors, alpha)

        getter = getattr(points, f"get_{part}colors")
        assert_array_equal(getter(), rgba)


class TestDot(DotBase):

    def test_simple(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Dot()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [C0] * 3, 1)
        self.check_colors("edge", points, [C0] * 3, 1)

    def test_filled_unfilled_mix(self):

        x = [1, 2]
        y = [4, 5]
        marker = ["a", "b"]
        shapes = ["o", "x"]

        mark = Dot(edgecolor="w", stroke=2, edgewidth=1)
        p = Plot(x=x, y=y).add(mark, marker=marker).scale(marker=shapes).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [C0, to_rgba(C0, 0)], None)
        self.check_colors("edge", points, ["w", C0], 1)

        expected = [mark.edgewidth, mark.stroke]
        assert_array_equal(points.get_linewidths(), expected)

    def test_missing_coordinate_data(self):

        x = [1, float("nan"), 3]
        y = [5, 3, 4]

        p = Plot(x=x, y=y).add(Dot()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, [1, 3], [5, 4])

    @pytest.mark.parametrize("prop", ["color", "fill", "marker", "pointsize"])
    def test_missing_semantic_data(self, prop):

        x = [1, 2, 3]
        y = [5, 3, 4]
        z = ["a", float("nan"), "b"]

        p = Plot(x=x, y=y, **{prop: z}).add(Dot()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, [1, 3], [5, 4])


class TestDots(DotBase):

    def test_simple(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Dots()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [C0] * 3, .2)
        self.check_colors("edge", points, [C0] * 3, 1)

    def test_set_color(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        m = Dots(color=".25")
        p = Plot(x=x, y=y).add(m).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [m.color] * 3, .2)
        self.check_colors("edge", points, [m.color] * 3, 1)

    def test_map_color(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        p = Plot(x=x, y=y, color=c).add(Dots()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [C0, C1, C0], .2)
        self.check_colors("edge", points, [C0, C1, C0], 1)

    def test_fill(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        p = Plot(x=x, y=y, color=c).add(Dots(fill=False)).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [C0, C1, C0], 0)
        self.check_colors("edge", points, [C0, C1, C0], 1)

    def test_pointsize(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        s = 3
        p = Plot(x=x, y=y).add(Dots(pointsize=s)).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        assert_array_equal(points.get_sizes(), [s ** 2] * 3)

    def test_stroke(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        s = 3
        p = Plot(x=x, y=y).add(Dots(stroke=s)).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        assert_array_equal(points.get_linewidths(), [s] * 3)

    def test_filled_unfilled_mix(self):

        x = [1, 2]
        y = [4, 5]
        marker = ["a", "b"]
        shapes = ["o", "x"]

        mark = Dots(stroke=2)
        p = Plot(x=x, y=y).add(mark, marker=marker).scale(marker=shapes).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)
        self.check_colors("face", points, [to_rgba(C0, .2), to_rgba(C0, 0)], None)
        self.check_colors("edge", points, [C0, C0], 1)
        assert_array_equal(points.get_linewidths(), [mark.stroke] * 2)
