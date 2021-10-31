import pytest

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

    def test_categorical_dodge_vertical(self):

        x = ["a", "a", "b", "b"]
        y = [1, 2, 3, 4]
        group = ["x", "y", "x", "y"]
        w = .8
        bars = self.plot_bars(
            {"x": x, "y": y, "group": group}, {"multiple": "dodge"}, {}
        )
        for i, bar in enumerate(bars[:2]):
            self.check_bar(bar, i - w / 2, 0, w / 2, y[i * 2])
        for i, bar in enumerate(bars[2:]):
            self.check_bar(bar, i, 0, w / 2, y[i * 2 + 1])

    def test_categorical_dodge_horizontal(self):

        x = [1, 2, 3, 4]
        y = ["a", "a", "b", "b"]
        group = ["x", "y", "x", "y"]
        w = .8
        bars = self.plot_bars(
            {"x": x, "y": y, "group": group}, {"multiple": "dodge"}, {}
        )
        for i, bar in enumerate(bars[:2]):
            self.check_bar(bar, 0, i - w / 2, x[i * 2], w / 2)
        for i, bar in enumerate(bars[2:]):
            self.check_bar(bar, 0, i, x[i * 2 + 1], w / 2)
