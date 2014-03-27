import numpy as np
import matplotlib as mpl

import nose.tools as nt
import numpy.testing as npt

from .. import palettes, utils, rcmod


class TestColorPalettes(object):

    def test_current_palette(self):

        pal = palettes.color_palette(["red", "blue", "green"], 3)
        rcmod.set_palette(pal, 3)
        nt.assert_equal(pal, mpl.rcParams["axes.color_cycle"])
        rcmod.set()

    def test_palette_context(self):

        default_pal = palettes.color_palette()
        context_pal = palettes.color_palette("muted")

        with palettes.color_palette(context_pal):
            nt.assert_equal(mpl.rcParams["axes.color_cycle"], context_pal)

        nt.assert_equal(mpl.rcParams["axes.color_cycle"], default_pal)

    def test_big_palette_context(self):

        default_pal = palettes.color_palette()
        context_pal = palettes.color_palette("husl", 10)

        with palettes.color_palette(context_pal, 10):
            nt.assert_equal(mpl.rcParams["axes.color_cycle"], context_pal)

        nt.assert_equal(mpl.rcParams["axes.color_cycle"], default_pal)

    def test_seaborn_palettes(self):

        pals = "deep", "muted", "pastel", "bright", "dark", "colorblind"
        for name in pals:
            pal_out = palettes.color_palette(name)
            nt.assert_equal(len(np.unique(pal_out)), 6)

    def test_hls_palette(self):

        hls_pal1 = palettes.hls_palette()
        hls_pal2 = palettes.color_palette("hls")
        npt.assert_array_equal(hls_pal1, hls_pal2)

    def test_husl_palette(self):

        husl_pal1 = palettes.husl_palette()
        husl_pal2 = palettes.color_palette("husl")
        npt.assert_array_equal(husl_pal1, husl_pal2)

    def test_mpl_palette(self):

        mpl_pal1 = palettes.mpl_palette("Reds")
        mpl_pal2 = palettes.color_palette("Reds")
        npt.assert_array_equal(mpl_pal1, mpl_pal2)

    def test_mpl_dark_palette(self):

        mpl_pal1 = palettes.mpl_palette("Blues_d")
        mpl_pal2 = palettes.color_palette("Blues_d")
        npt.assert_array_equal(mpl_pal1, mpl_pal2)

    def test_bad_palette_name(self):

        with nt.assert_raises(ValueError):
            palettes.color_palette("IAmNotAPalette")

    def test_bad_palette_colors(self):

        pal = ["red", "blue", "iamnotacolor"]
        with nt.assert_raises(ValueError):
            palettes.color_palette(pal)

    def test_palette_desat(self):

        pal1 = palettes.husl_palette(6)
        pal1 = [utils.desaturate(c, .5) for c in pal1]
        pal2 = palettes.color_palette("husl", desat=.5)
        npt.assert_array_equal(pal1, pal2)

    def test_palette_is_list_of_tuples(self):

        pal_in = np.array(["red", "blue", "green"])
        pal_out = palettes.color_palette(pal_in, 3)

        nt.assert_is_instance(pal_out, list)
        nt.assert_is_instance(pal_out[0], tuple)
        nt.assert_is_instance(pal_out[0][0], float)
        nt.assert_equal(len(pal_out[0]), 3)

    def test_palette_cycles(self):

        deep = palettes.color_palette("deep")
        double_deep = palettes.color_palette("deep", 12)
        nt.assert_equal(double_deep, deep + deep)

    def test_hls_values(self):

        pal1 = palettes.hls_palette(6, h=0)
        pal2 = palettes.hls_palette(6, h=.5)
        pal2 = pal2[3:] + pal2[:3]
        npt.assert_array_almost_equal(pal1, pal2)

        pal_dark = palettes.hls_palette(5, l=.2)
        pal_bright = palettes.hls_palette(5, l=.8)
        npt.assert_array_less(list(map(sum, pal_dark)),
                              list(map(sum, pal_bright)))

        pal_flat = palettes.hls_palette(5, s=.1)
        pal_bold = palettes.hls_palette(5, s=.9)
        npt.assert_array_less(list(map(np.std, pal_flat)),
                              list(map(np.std, pal_bold)))

    def test_husl_values(self):

        pal1 = palettes.husl_palette(6, h=0)
        pal2 = palettes.husl_palette(6, h=.5)
        pal2 = pal2[3:] + pal2[:3]
        npt.assert_array_almost_equal(pal1, pal2)

        pal_dark = palettes.husl_palette(5, l=.2)
        pal_bright = palettes.husl_palette(5, l=.8)
        npt.assert_array_less(list(map(sum, pal_dark)),
                              list(map(sum, pal_bright)))

        pal_flat = palettes.husl_palette(5, s=.1)
        pal_bold = palettes.husl_palette(5, s=.9)
        npt.assert_array_less(list(map(np.std, pal_flat)),
                              list(map(np.std, pal_bold)))

    def test_cbrewer_qual(self):

        pal_short = palettes.mpl_palette("Set1", 4)
        pal_long = palettes.mpl_palette("Set1", 6)
        nt.assert_equal(pal_short, pal_long[:4])

        pal_full = palettes.mpl_palette("Set2", 8)
        pal_long = palettes.mpl_palette("Set2", 10)
        nt.assert_equal(pal_full, pal_long[:8])

    def test_mpl_reversal(self):

        pal_forward = palettes.mpl_palette("BuPu", 6)
        pal_reverse = palettes.mpl_palette("BuPu_r", 6)
        nt.assert_equal(pal_forward, pal_reverse[::-1])

    def test_dark_palette(self):

        pal_forward = palettes.dark_palette("red")
        pal_reverse = palettes.dark_palette("red", reverse=True)
        npt.assert_array_almost_equal(pal_forward, pal_reverse[::-1])

        pal_cmap = palettes.dark_palette("blue", as_cmap=True)
        nt.assert_is_instance(pal_cmap, mpl.colors.LinearSegmentedColormap)

    def test_blend_palette(self):

        colors = ["red", "yellow", "white"]
        pal_cmap = palettes.blend_palette(colors, as_cmap=True)
        nt.assert_is_instance(pal_cmap, mpl.colors.LinearSegmentedColormap)
