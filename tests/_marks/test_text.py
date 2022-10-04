
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.text import Text as MPLText

from numpy.testing import assert_array_almost_equal

from seaborn._core.plot import Plot
from seaborn._marks.text import Text


class TestText:

    def get_texts(self, ax):
        if ax.texts:
            return list(ax.texts)
        else:
            # Compatibility with matplotlib < 3.5 (I think)
            return [a for a in ax.artists if isinstance(a, MPLText)]

    def test_simple(self):

        x = y = [1, 2, 3]
        s = list("abc")

        p = Plot(x, y, text=s).add(Text()).plot()
        ax = p._figure.axes[0]
        for i, text in enumerate(self.get_texts(ax)):
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
        for i, text in enumerate(self.get_texts(ax)):
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
        texts = self.get_texts(ax)
        assert texts[0].get_color() == texts[1].get_color()
        assert texts[0].get_color() != texts[2].get_color()
        assert (
            texts[0].get_fontsize()
            < texts[1].get_fontsize()
            < texts[2].get_fontsize()
        )

    def test_mapped_alignment(self):

        x = [1, 2]
        p = Plot(x=x, y=x, halign=x, valign=x, text=x).add(Text()).plot()
        ax = p._figure.axes[0]
        t1, t2 = self.get_texts(ax)
        assert t1.get_horizontalalignment() == "left"
        assert t2.get_horizontalalignment() == "right"
        assert t1.get_verticalalignment() == "top"
        assert t2.get_verticalalignment() == "bottom"

    def test_identity_fontsize(self):

        x = y = [1, 2, 3]
        s = list("abc")
        fs = [5, 8, 12]
        p = Plot(x, y, text=s, fontsize=fs).add(Text()).scale(fontsize=None).plot()
        ax = p._figure.axes[0]
        for i, text in enumerate(self.get_texts(ax)):
            assert text.get_fontsize() == fs[i]

    def test_offset_centered(self):

        x = y = [1, 2, 3]
        s = list("abc")
        p = Plot(x, y, text=s).add(Text()).plot()
        ax = p._figure.axes[0]
        ax_trans = ax.transData.get_matrix()
        for text in self.get_texts(ax):
            assert_array_almost_equal(text.get_transform().get_matrix(), ax_trans)

    def test_offset_valign(self):

        x = y = [1, 2, 3]
        s = list("abc")
        m = Text(valign="bottom", fontsize=5, offset=.1)
        p = Plot(x, y, text=s).add(m).plot()
        ax = p._figure.axes[0]
        expected_shift_matrix = np.zeros((3, 3))
        expected_shift_matrix[1, -1] = m.offset * ax.figure.dpi / 72
        ax_trans = ax.transData.get_matrix()
        for text in self.get_texts(ax):
            shift_matrix = text.get_transform().get_matrix() - ax_trans
            assert_array_almost_equal(shift_matrix, expected_shift_matrix)

    def test_offset_halign(self):

        x = y = [1, 2, 3]
        s = list("abc")
        m = Text(halign="right", fontsize=10, offset=.5)
        p = Plot(x, y, text=s).add(m).plot()
        ax = p._figure.axes[0]
        expected_shift_matrix = np.zeros((3, 3))
        expected_shift_matrix[0, -1] = -m.offset * ax.figure.dpi / 72
        ax_trans = ax.transData.get_matrix()
        for text in self.get_texts(ax):
            shift_matrix = text.get_transform().get_matrix() - ax_trans
            assert_array_almost_equal(shift_matrix, expected_shift_matrix)
