
import numpy as np
from matplotlib.colors import same_color, to_rgba

from numpy.testing import assert_array_equal

from seaborn._core.plot import Plot
from seaborn._marks.lines import Line, Path, Lines, Paths


class TestPath:

    def test_xy_data(self):

        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        p = Plot(x=x, y=y, group=g).add(Path()).plot()
        line1, line2 = p._figure.axes[0].get_lines()

        assert_array_equal(line1.get_xdata(), [1, 3, np.nan])
        assert_array_equal(line1.get_ydata(), [1, 2, np.nan])
        assert_array_equal(line2.get_xdata(), [5, 2])
        assert_array_equal(line2.get_ydata(), [4, 3])

    def test_shared_colors_direct(self):

        x = y = [1, 2, 3]
        m = Path(color="r")
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert same_color(line.get_color(), "r")
        assert same_color(line.get_markeredgecolor(), "r")
        assert same_color(line.get_markerfacecolor(), "r")

    def test_separate_colors_direct(self):

        x = y = [1, 2, 3]
        y = [1, 2, 3]
        m = Path(color="r", edgecolor="g", fillcolor="b")
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert same_color(line.get_color(), m.color)
        assert same_color(line.get_markeredgecolor(), m.edgecolor)
        assert same_color(line.get_markerfacecolor(), m.fillcolor)

    def test_shared_colors_mapped(self):

        x = y = [1, 2, 3, 4]
        c = ["a", "a", "b", "b"]
        m = Path()
        p = Plot(x=x, y=y, color=c).add(m).plot()
        ax = p._figure.axes[0]
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), f"C{i}")
            assert same_color(line.get_markeredgecolor(), f"C{i}")
            assert same_color(line.get_markerfacecolor(), f"C{i}")

    def test_separate_colors_mapped(self):

        x = y = [1, 2, 3, 4]
        c = ["a", "a", "b", "b"]
        d = ["x", "y", "x", "y"]
        m = Path()
        p = Plot(x=x, y=y, color=c, fillcolor=d).add(m).plot()
        ax = p._figure.axes[0]
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), f"C{i // 2}")
            assert same_color(line.get_markeredgecolor(), f"C{i // 2}")
            assert same_color(line.get_markerfacecolor(), f"C{i % 2}")

    def test_color_with_alpha(self):

        x = y = [1, 2, 3]
        m = Path(color=(.4, .9, .2, .5), fillcolor=(.2, .2, .3, .9))
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert same_color(line.get_color(), m.color)
        assert same_color(line.get_markeredgecolor(), m.color)
        assert same_color(line.get_markerfacecolor(), m.fillcolor)

    def test_color_and_alpha(self):

        x = y = [1, 2, 3]
        m = Path(color=(.4, .9, .2), fillcolor=(.2, .2, .3), alpha=.5)
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert same_color(line.get_color(), to_rgba(m.color, m.alpha))
        assert same_color(line.get_markeredgecolor(), to_rgba(m.color, m.alpha))
        assert same_color(line.get_markerfacecolor(), to_rgba(m.fillcolor, m.alpha))

    def test_other_props_direct(self):

        x = y = [1, 2, 3]
        m = Path(marker="s", linestyle="--", linewidth=3, pointsize=10, edgewidth=1)
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert line.get_marker() == m.marker
        assert line.get_linestyle() == m.linestyle
        assert line.get_linewidth() == m.linewidth
        assert line.get_markersize() == m.pointsize
        assert line.get_markeredgewidth() == m.edgewidth

    def test_other_props_mapped(self):

        x = y = [1, 2, 3, 4]
        g = ["a", "a", "b", "b"]
        m = Path()
        p = Plot(x=x, y=y, marker=g, linestyle=g, pointsize=g).add(m).plot()
        line1, line2 = p._figure.axes[0].get_lines()
        assert line1.get_marker() != line2.get_marker()
        # Matplotlib bug in storing linestyle from dash pattern
        # assert line1.get_linestyle() != line2.get_linestyle()
        assert line1.get_markersize() != line2.get_markersize()


class TestLine:

    # Most behaviors shared with Path and covered by above tests

    def test_xy_data(self):

        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        p = Plot(x=x, y=y, group=g).add(Line()).plot()
        line1, line2 = p._figure.axes[0].get_lines()

        assert_array_equal(line1.get_xdata(), [1, 3])
        assert_array_equal(line1.get_ydata(), [1, 2])
        assert_array_equal(line2.get_xdata(), [2, 5])
        assert_array_equal(line2.get_ydata(), [3, 4])


class TestPaths:

    def test_xy_data(self):

        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        p = Plot(x=x, y=y, group=g).add(Paths()).plot()
        lines, = p._figure.axes[0].collections

        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], [1, 3, np.nan])
        assert_array_equal(verts[1], [1, 2, np.nan])

        verts = lines.get_paths()[1].vertices.T
        assert_array_equal(verts[0], [5, 2])
        assert_array_equal(verts[1], [4, 3])

    def test_props_direct(self):

        x = y = [1, 2, 3]
        m = Paths(color="r", linewidth=1, linestyle=(3, 1))
        p = Plot(x=x, y=y).add(m).plot()
        lines, = p._figure.axes[0].collections

        assert same_color(lines.get_color().squeeze(), m.color)
        assert lines.get_linewidth().item() == m.linewidth
        assert lines.get_linestyle()[0] == (0, list(m.linestyle))

    def test_props_mapped(self):

        x = y = [1, 2, 3, 4]
        g = ["a", "a", "b", "b"]
        p = Plot(x=x, y=y, color=g, linewidth=g, linestyle=g).add(Paths()).plot()
        lines, = p._figure.axes[0].collections

        assert not np.array_equal(lines.get_colors()[0], lines.get_colors()[1])
        assert lines.get_linewidths()[0] != lines.get_linewidth()[1]
        assert lines.get_linestyle()[0] != lines.get_linestyle()[1]

    def test_color_with_alpha(self):

        x = y = [1, 2, 3]
        m = Paths(color=(.2, .6, .9, .5))
        p = Plot(x=x, y=y).add(m).plot()
        lines, = p._figure.axes[0].collections
        assert same_color(lines.get_colors().squeeze(), m.color)

    def test_color_and_alpha(self):

        x = y = [1, 2, 3]
        m = Paths(color=(.2, .6, .9), alpha=.5)
        p = Plot(x=x, y=y).add(m).plot()
        lines, = p._figure.axes[0].collections
        assert same_color(lines.get_colors().squeeze(), to_rgba(m.color, m.alpha))


class TestLines:

    def test_xy_data(self):

        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        p = Plot(x=x, y=y, group=g).add(Lines()).plot()
        lines, = p._figure.axes[0].collections

        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], [1, 3])
        assert_array_equal(verts[1], [1, 2])

        verts = lines.get_paths()[1].vertices.T
        assert_array_equal(verts[0], [2, 5])
        assert_array_equal(verts[1], [3, 4])
