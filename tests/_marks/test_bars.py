import pytest

from matplotlib.colors import to_rgba

from seaborn._core.plot import Plot
from seaborn._marks.bars import Bar


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

    def test_direct_properties(self):

        x = ["a", "b", "c"]
        y = [1, 3, 2]

        mark = Bar(
            color="C2",
            alpha=.5,
            edgecolor="k",
            edgealpha=.9,
            edgestyle=(2, 1),
            edgewidth=1.5,
        )

        p = Plot(x, y).add(mark).plot()
        ax = p._figure.axes[0]
        for bar in ax.patches:
            assert bar.get_facecolor() == to_rgba(mark.color, mark.alpha)
            assert bar.get_edgecolor() == to_rgba(mark.edgecolor, mark.edgealpha)
            assert bar.get_linewidth() == mark.edgewidth
            assert bar.get_linestyle() == (0, mark.edgestyle)

    def test_mapped_properties(self):

        x = ["a", "b"]
        y = [1, 2]
        mark = Bar(alpha=.2)
        p = Plot(x, y, color=x, edgewidth=y).add(mark).plot()
        ax = p._figure.axes[0]
        for i, bar in enumerate(ax.patches):
            assert bar.get_facecolor() == to_rgba(f"C{i}", mark.alpha)
            assert bar.get_edgecolor() == to_rgba(f"C{i}", 1)
        assert ax.patches[0].get_linewidth() < ax.patches[1].get_linewidth()
