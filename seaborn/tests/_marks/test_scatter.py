from matplotlib.colors import to_rgba_array

from numpy.testing import assert_array_equal

from seaborn._core.plot import Plot
from seaborn._marks.scatter import Scatter


class TestScatter:

    def check_offsets(self, points, x, y):

        offsets = points.get_offsets().T
        assert_array_equal(offsets[0], x)
        assert_array_equal(offsets[1], y)

    def check_colors(self, part, points, colors, alpha=None):

        rgba = to_rgba_array(colors, alpha)

        getter = getattr(points, f"get_{part}colors")
        assert_array_equal(getter(), rgba)

    def test_simple(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Scatter()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        self.check_colors("face", points, ["C0"] * 3, .2)
        self.check_colors("edge", points, ["C0"] * 3, 1)

    def test_color_feature(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Scatter(color="g")).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        self.check_colors("face", points, ["g"] * 3, .2)
        self.check_colors("edge", points, ["g"] * 3, 1)

    def test_color_mapped(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        p = Plot(x=x, y=y, color=c).add(Scatter()).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        self.check_colors("face", points, ["C0", "C1", "C0"], .2)
        self.check_colors("edge", points, ["C0", "C1", "C0"], 1)

    def test_fill(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        p = Plot(x=x, y=y, color=c).add(Scatter(fill=False)).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        self.check_colors("face", points, ["C0", "C1", "C0"], 0)
        self.check_colors("edge", points, ["C0", "C1", "C0"], 1)

    def test_pointsize(self):

        x = [1, 2, 3]
        y = [4, 5, 2]
        s = 3
        p = Plot(x=x, y=y).add(Scatter(pointsize=s)).plot()
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, x, y)
        assert_array_equal(points.get_sizes(), [s ** 2] * 3)
