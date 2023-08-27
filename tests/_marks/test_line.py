
import numpy as np
import matplotlib as mpl
from matplotlib.colors import same_color, to_rgba

from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.plot import Plot
from seaborn._core.moves import Dodge
from seaborn._marks.line import Dash, Line, Path, Lines, Paths, Range


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
        color = ".44"
        m = Path(color=color)
        p = Plot(x=x, y=y).add(m).plot()
        line, = p._figure.axes[0].get_lines()
        assert same_color(line.get_color(), color)
        assert same_color(line.get_markeredgecolor(), color)
        assert same_color(line.get_markerfacecolor(), color)

    def test_separate_colors_direct(self):

        x = y = [1, 2, 3]
        y = [1, 2, 3]
        m = Path(color=".22", edgecolor=".55", fillcolor=".77")
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
        colors = p._theme["axes.prop_cycle"].by_key()["color"]
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), colors[i])
            assert same_color(line.get_markeredgecolor(), colors[i])
            assert same_color(line.get_markerfacecolor(), colors[i])

    def test_separate_colors_mapped(self):

        x = y = [1, 2, 3, 4]
        c = ["a", "a", "b", "b"]
        d = ["x", "y", "x", "y"]
        m = Path()
        p = Plot(x=x, y=y, color=c, fillcolor=d).add(m).plot()
        ax = p._figure.axes[0]
        colors = p._theme["axes.prop_cycle"].by_key()["color"]
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), colors[i // 2])
            assert same_color(line.get_markeredgecolor(), colors[i // 2])
            assert same_color(line.get_markerfacecolor(), colors[i % 2])

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

    def test_capstyle(self):

        x = y = [1, 2]
        rc = {"lines.solid_capstyle": "projecting", "lines.dash_capstyle": "round"}

        p = Plot(x, y).add(Path()).theme(rc).plot()
        line, = p._figure.axes[0].get_lines()
        assert line.get_dash_capstyle() == "projecting"

        p = Plot(x, y).add(Path(linestyle="--")).theme(rc).plot()
        line, = p._figure.axes[0].get_lines()
        assert line.get_dash_capstyle() == "round"

        p = Plot(x, y).add(Path({"solid_capstyle": "butt"})).theme(rc).plot()
        line, = p._figure.axes[0].get_lines()
        assert line.get_solid_capstyle() == "butt"


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

    def test_set_properties(self):

        x = y = [1, 2, 3]
        m = Paths(color=".737", linewidth=1, linestyle=(3, 1))
        p = Plot(x=x, y=y).add(m).plot()
        lines, = p._figure.axes[0].collections

        assert same_color(lines.get_color().squeeze(), m.color)
        assert lines.get_linewidth().item() == m.linewidth
        assert lines.get_dashes()[0] == (0, list(m.linestyle))

    def test_mapped_properties(self):

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

    def test_capstyle(self):

        x = y = [1, 2]
        rc = {"lines.solid_capstyle": "projecting"}

        with mpl.rc_context(rc):
            p = Plot(x, y).add(Paths()).plot()
            lines = p._figure.axes[0].collections[0]
            assert lines.get_capstyle() == "projecting"

            p = Plot(x, y).add(Paths(linestyle="--")).plot()
            lines = p._figure.axes[0].collections[0]
            assert lines.get_capstyle() == "projecting"

            p = Plot(x, y).add(Paths({"capstyle": "butt"})).plot()
            lines = p._figure.axes[0].collections[0]
            assert lines.get_capstyle() == "butt"


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

    def test_single_orient_value(self):

        x = [1, 1, 1]
        y = [1, 2, 3]
        p = Plot(x, y).add(Lines()).plot()
        lines, = p._figure.axes[0].collections
        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], x)
        assert_array_equal(verts[1], y)


class TestRange:

    def test_xy_data(self):

        x = [1, 2]
        ymin = [1, 4]
        ymax = [2, 3]

        p = Plot(x=x, ymin=ymin, ymax=ymax).add(Range()).plot()
        lines, = p._figure.axes[0].collections

        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_equal(verts[0], [x[i], x[i]])
            assert_array_equal(verts[1], [ymin[i], ymax[i]])

    def test_auto_range(self):

        x = [1, 1, 2, 2, 2]
        y = [1, 2, 3, 4, 5]

        p = Plot(x=x, y=y).add(Range()).plot()
        lines, = p._figure.axes[0].collections
        paths = lines.get_paths()
        assert_array_equal(paths[0].vertices, [(1, 1), (1, 2)])
        assert_array_equal(paths[1].vertices, [(2, 3), (2, 5)])

    def test_mapped_color(self):

        x = [1, 2, 1, 2]
        ymin = [1, 4, 3, 2]
        ymax = [2, 3, 1, 4]
        group = ["a", "a", "b", "b"]

        p = Plot(x=x, ymin=ymin, ymax=ymax, color=group).add(Range()).plot()
        lines, = p._figure.axes[0].collections
        colors = p._theme["axes.prop_cycle"].by_key()["color"]

        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_equal(verts[0], [x[i], x[i]])
            assert_array_equal(verts[1], [ymin[i], ymax[i]])
            assert same_color(lines.get_colors()[i], colors[i // 2])

    def test_direct_properties(self):

        x = [1, 2]
        ymin = [1, 4]
        ymax = [2, 3]

        m = Range(color=".654", linewidth=4)
        p = Plot(x=x, ymin=ymin, ymax=ymax).add(m).plot()
        lines, = p._figure.axes[0].collections

        for i, path in enumerate(lines.get_paths()):
            assert same_color(lines.get_colors()[i], m.color)
            assert lines.get_linewidths()[i] == m.linewidth


class TestDash:

    def test_xy_data(self):

        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        p = Plot(x=x, y=y).add(Dash()).plot()
        lines, = p._figure.axes[0].collections

        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_almost_equal(verts[0], [x[i] - .4, x[i] + .4])
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_xy_data_grouped(self):

        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]
        color = ["a", "b", "a", "b"]

        p = Plot(x=x, y=y, color=color).add(Dash()).plot()
        lines, = p._figure.axes[0].collections

        idx = [0, 2, 1, 3]
        for i, path in zip(idx, lines.get_paths()):
            verts = path.vertices.T
            assert_array_almost_equal(verts[0], [x[i] - .4, x[i] + .4])
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_set_properties(self):

        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        m = Dash(color=".8", linewidth=4)
        p = Plot(x=x, y=y).add(m).plot()
        lines, = p._figure.axes[0].collections

        for color in lines.get_color():
            assert same_color(color, m.color)
        for linewidth in lines.get_linewidth():
            assert linewidth == m.linewidth

    def test_mapped_properties(self):

        x = [0, 1]
        y = [1, 2]
        color = ["a", "b"]
        linewidth = [1, 2]

        p = Plot(x=x, y=y, color=color, linewidth=linewidth).add(Dash()).plot()
        lines, = p._figure.axes[0].collections
        palette = p._theme["axes.prop_cycle"].by_key()["color"]

        for color, line_color in zip(palette, lines.get_color()):
            assert same_color(color, line_color)

        linewidths = lines.get_linewidths()
        assert linewidths[1] > linewidths[0]

    def test_width(self):

        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        p = Plot(x=x, y=y).add(Dash(width=.4)).plot()
        lines, = p._figure.axes[0].collections

        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_almost_equal(verts[0], [x[i] - .2, x[i] + .2])
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_dodge(self):

        x = [0, 1]
        y = [1, 2]
        group = ["a", "b"]

        p = Plot(x=x, y=y, group=group).add(Dash(), Dodge()).plot()
        lines, = p._figure.axes[0].collections

        paths = lines.get_paths()

        v0 = paths[0].vertices.T
        assert_array_almost_equal(v0[0], [-.4, 0])
        assert_array_equal(v0[1], [y[0], y[0]])

        v1 = paths[1].vertices.T
        assert_array_almost_equal(v1[0], [1, 1.4])
        assert_array_equal(v1[1], [y[1], y[1]])
