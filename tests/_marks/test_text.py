
from matplotlib.colors import to_rgba

from seaborn._core.plot import Plot
from seaborn._marks.text import Text


class TestText:

    def test_simple(self):

        x = y = [1, 2, 3]
        s = list("abc")

        p = Plot(x, y, text=s).add(Text()).plot()
        ax = p._figure.axes[0]
        for i, text in enumerate(ax.texts):
            x_, y_ = text.get_position()
            assert x_ == x[i]
            assert y_ == y[i]
            assert text.get_text() == s[i]
            assert text.get_horizontalalignment() == "center"
            assert text.get_verticalalignment() == "center_baseline"

    def test_set_properties(self):

        x = y = [1, 2, 3]
        s = list("abc")
        color = "red"
        alpha = .6
        fontsize = 6
        valign = "bottom"

        m = Text(color=color, alpha=alpha, fontsize=fontsize, valign=valign)
        p = Plot(x, y, text=s).add(m).plot()
        ax = p._figure.axes[0]
        for i, text in enumerate(ax.texts):
            assert text.get_text() == s[i]
            assert text.get_color() == to_rgba(m.color, m.alpha)
            assert text.get_fontsize() == m.fontsize
            assert text.get_verticalalignment() == m.valign

    def test_mapped_properties(self):

        x = y = [1, 2, 3]
        s = list("abc")
        color = list("aab")
        fontsize = [1, 2, 4]

        p = Plot(x, y, color=color, fontsize=fontsize, text=s).add(Text()).plot()
        ax = p._figure.axes[0]
        texts = ax.texts
        assert texts[0].get_color() == texts[1].get_color()
        assert texts[0].get_color() != texts[2].get_color()
        assert (
            texts[0].get_fontsize()
            < texts[1].get_fontsize()
            < texts[2].get_fontsize()
        )

    def test_identity_fontsize(self):

        x = y = [1, 2, 3]
        s = list("abc")
        fs = [5, 8, 12]
        p = Plot(x, y, text=s, fontsize=fs).add(Text()).scale(fontsize=None).plot()
        ax = p._figure.axes[0]
        for i, text in enumerate(ax.texts):
            assert text.get_fontsize() == fs[i]
