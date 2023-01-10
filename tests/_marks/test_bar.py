
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba, to_rgba_array

import pytest
from numpy.testing import assert_array_equal

from seaborn._core.plot import Plot
from seaborn._marks.bar import Bar, Bars


class TestBar:

    def plot_bars(self, variables, mark_kws, layer_kws):

        p = Plot(**variables).add(Bar(**mark_kws), **layer_kws).plot()
        ax = p._figure.axes[0]
        return [bar for barlist in ax.containers for bar in barlist]

    def check_bar(self, bar, x, y, width, height):

        assert bar.get_x() == pytest.approx(x)
        assert bar.get_y() == pytest.approx(y)
        assert bar.get_width() == pytest.approx(width)
        assert bar.get_height() == pytest.approx(height)

    def test_categorical_positions_vertical(self):

        x = ["a", "b"]
        y = [1, 2]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, i - w / 2, 0, w, y[i])

    def test_categorical_positions_horizontal(self):

        x = [1, 2]
        y = ["a", "b"]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, 0, i - w / 2, x[i], w)

    def test_numeric_positions_vertical(self):

        x = [1, 2]
        y = [3, 4]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, x[i] - w / 2, 0, w, y[i])

    def test_numeric_positions_horizontal(self):

        x = [1, 2]
        y = [3, 4]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {"orient": "h"})
        for i, bar in enumerate(bars):
            self.check_bar(bar, 0, y[i] - w / 2, x[i], w)

    def test_set_properties(self):

        x = ["a", "b", "c"]
        y = [1, 3, 2]

        mark = Bar(
            color=".8",
            alpha=.5,
            edgecolor=".3",
            edgealpha=.9,
            edgestyle=(2, 1),
            edgewidth=1.5,
        )

        p = Plot(x, y).add(mark).plot()
        ax = p._figure.axes[0]
        for bar in ax.patches:
            assert bar.get_facecolor() == to_rgba(mark.color, mark.alpha)
            assert bar.get_edgecolor() == to_rgba(mark.edgecolor, mark.edgealpha)
            # See comments in plotting method for why we need these adjustments
            assert bar.get_linewidth() == mark.edgewidth * 2
            expected_dashes = (mark.edgestyle[0] / 2, mark.edgestyle[1] / 2)
            assert bar.get_linestyle() == (0, expected_dashes)

    def test_mapped_properties(self):

        x = ["a", "b"]
        y = [1, 2]
        mark = Bar(alpha=.2)
        p = Plot(x, y, color=x, edgewidth=y).add(mark).plot()
        ax = p._figure.axes[0]
        colors = p._theme["axes.prop_cycle"].by_key()["color"]
        for i, bar in enumerate(ax.patches):
            assert bar.get_facecolor() == to_rgba(colors[i], mark.alpha)
            assert bar.get_edgecolor() == to_rgba(colors[i], 1)
        assert ax.patches[0].get_linewidth() < ax.patches[1].get_linewidth()

    def test_zero_height_skipped(self):

        p = Plot(["a", "b", "c"], [1, 0, 2]).add(Bar()).plot()
        ax = p._figure.axes[0]
        assert len(ax.patches) == 2

    def test_artist_kws_clip(self):

        p = Plot(["a", "b"], [1, 2]).add(Bar({"clip_on": False})).plot()
        patch = p._figure.axes[0].patches[0]
        assert patch.clipbox is None


class TestBars:

    @pytest.fixture
    def x(self):
        return pd.Series([4, 5, 6, 7, 8], name="x")

    @pytest.fixture
    def y(self):
        return pd.Series([2, 8, 3, 5, 9], name="y")

    @pytest.fixture
    def color(self):
        return pd.Series(["a", "b", "c", "a", "c"], name="color")

    def test_positions(self, x, y):

        p = Plot(x, y).add(Bars()).plot()
        ax = p._figure.axes[0]
        paths = ax.collections[0].get_paths()
        assert len(paths) == len(x)
        for i, path in enumerate(paths):
            verts = path.vertices
            assert verts[0, 0] == pytest.approx(x[i] - .5)
            assert verts[1, 0] == pytest.approx(x[i] + .5)
            assert verts[0, 1] == 0
            assert verts[3, 1] == y[i]

    def test_positions_horizontal(self, x, y):

        p = Plot(x=y, y=x).add(Bars(), orient="h").plot()
        ax = p._figure.axes[0]
        paths = ax.collections[0].get_paths()
        assert len(paths) == len(x)
        for i, path in enumerate(paths):
            verts = path.vertices
            assert verts[0, 1] == pytest.approx(x[i] - .5)
            assert verts[3, 1] == pytest.approx(x[i] + .5)
            assert verts[0, 0] == 0
            assert verts[1, 0] == y[i]

    def test_width(self, x, y):

        p = Plot(x, y).add(Bars(width=.4)).plot()
        ax = p._figure.axes[0]
        paths = ax.collections[0].get_paths()
        for i, path in enumerate(paths):
            verts = path.vertices
            assert verts[0, 0] == pytest.approx(x[i] - .2)
            assert verts[1, 0] == pytest.approx(x[i] + .2)

    def test_mapped_color_direct_alpha(self, x, y, color):

        alpha = .5
        p = Plot(x, y, color=color).add(Bars(alpha=alpha)).plot()
        ax = p._figure.axes[0]
        fcs = ax.collections[0].get_facecolors()
        C0, C1, C2, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        expected = to_rgba_array([C0, C1, C2, C0, C2], alpha)
        assert_array_equal(fcs, expected)

    def test_mapped_edgewidth(self, x, y):

        p = Plot(x, y, edgewidth=y).add(Bars()).plot()
        ax = p._figure.axes[0]
        lws = ax.collections[0].get_linewidths()
        assert_array_equal(np.argsort(lws), np.argsort(y))

    def test_auto_edgewidth(self):

        x0 = np.arange(10)
        x1 = np.arange(1000)

        p0 = Plot(x0, x0).add(Bars()).plot()
        p1 = Plot(x1, x1).add(Bars()).plot()

        lw0 = p0._figure.axes[0].collections[0].get_linewidths()
        lw1 = p1._figure.axes[0].collections[0].get_linewidths()

        assert (lw0 > lw1).all()

    def test_unfilled(self, x, y):

        p = Plot(x, y).add(Bars(fill=False, edgecolor="C4")).plot()
        ax = p._figure.axes[0]
        fcs = ax.collections[0].get_facecolors()
        ecs = ax.collections[0].get_edgecolors()
        colors = p._theme["axes.prop_cycle"].by_key()["color"]
        assert_array_equal(fcs, to_rgba_array([colors[0]] * len(x), 0))
        assert_array_equal(ecs, to_rgba_array([colors[4]] * len(x), 1))

    def test_log_scale(self):

        x = y = [1, 10, 100, 1000]
        p = Plot(x, y).add(Bars()).scale(x="log").plot()
        ax = p._figure.axes[0]

        paths = ax.collections[0].get_paths()
        for a, b in zip(paths, paths[1:]):
            assert a.vertices[1, 0] == pytest.approx(b.vertices[0, 0])
