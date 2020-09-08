from distutils.version import LooseVersion

import numpy as np
import matplotlib as mpl
import nose
import matplotlib.pyplot as plt
import nose.tools as nt
import numpy.testing as npt

from .. import rcmod, palettes, utils


class RCParamTester(object):

    def flatten_list(self, orig_list):

        iter_list = map(np.atleast_1d, orig_list)
        flat_list = [item for sublist in iter_list for item in sublist]
        return flat_list

    def assert_rc_params(self, params):

        for k, v in params.items():
            # Various subtle issues in matplotlib lead to unexpected
            # values for the backend rcParam, which isn't relevant here
            if k == "backend":
                continue
            if isinstance(v, np.ndarray):
                npt.assert_array_equal(mpl.rcParams[k], v)
            else:
                nt.assert_equal((k, mpl.rcParams[k]), (k, v))

    def assert_rc_params_equal(self, params1, params2):

        for key, v1 in params1.items():
            # Various subtle issues in matplotlib lead to unexpected
            # values for the backend rcParam, which isn't relevant here
            if key == "backend":
                continue

            v2 = params2[key]
            if isinstance(v1, np.ndarray):
                npt.assert_array_equal(v1, v2)
            else:
                nt.assert_equal(v1, v2)


class TestAxesStyle(RCParamTester):

    styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

    def test_default_return(self):

        current = rcmod.axes_style()
        self.assert_rc_params(current)

    def test_key_usage(self):

        _style_keys = set(rcmod._style_keys)
        for style in self.styles:
            nt.assert_true(not set(rcmod.axes_style(style)) ^ _style_keys)

    def test_bad_style(self):

        with nt.assert_raises(ValueError):
            rcmod.axes_style("i_am_not_a_style")

    def test_rc_override(self):

        rc = {"axes.facecolor": "blue", "foo.notaparam": "bar"}
        out = rcmod.axes_style("darkgrid", rc)
        nt.assert_equal(out["axes.facecolor"], "blue")
        nt.assert_not_in("foo.notaparam", out)

    def test_set_style(self):

        for style in self.styles:

            style_dict = rcmod.axes_style(style)
            rcmod.set_style(style)
            self.assert_rc_params(style_dict)

    def test_style_context_manager(self):

        rcmod.set_style("darkgrid")
        orig_params = rcmod.axes_style()
        context_params = rcmod.axes_style("whitegrid")

        with rcmod.axes_style("whitegrid"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @rcmod.axes_style("whitegrid")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)

    def test_style_context_independence(self):

        nt.assert_true(set(rcmod._style_keys) ^ set(rcmod._context_keys))

    def test_set_rc(self):

        rcmod.set_theme(rc={"lines.linewidth": 4})
        nt.assert_equal(mpl.rcParams["lines.linewidth"], 4)
        rcmod.set_theme()

    def test_set_with_palette(self):

        rcmod.reset_orig()

        rcmod.set_theme(palette="deep")
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        rcmod.reset_orig()

        rcmod.set_theme(palette="deep", color_codes=False)
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        rcmod.reset_orig()

        pal = palettes.color_palette("deep")
        rcmod.set_theme(palette=pal)
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        rcmod.reset_orig()

        rcmod.set_theme(palette=pal, color_codes=False)
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)
        rcmod.reset_orig()

        rcmod.set_theme()

    def test_reset_defaults(self):

        rcmod.reset_defaults()
        self.assert_rc_params(mpl.rcParamsDefault)
        rcmod.set_theme()

    def test_reset_orig(self):

        rcmod.reset_orig()
        self.assert_rc_params(mpl.rcParamsOrig)
        rcmod.set_theme()

    def test_set_is_alias(self):

        rcmod.set_theme(context="paper", style="white")
        params1 = mpl.rcParams.copy()
        rcmod.reset_orig()

        rcmod.set_theme(context="paper", style="white")
        params2 = mpl.rcParams.copy()

        self.assert_rc_params_equal(params1, params2)

        rcmod.set_theme()


class TestPlottingContext(RCParamTester):

    contexts = ["paper", "notebook", "talk", "poster"]

    def test_default_return(self):

        current = rcmod.plotting_context()
        self.assert_rc_params(current)

    def test_key_usage(self):

        _context_keys = set(rcmod._context_keys)
        for context in self.contexts:
            missing = set(rcmod.plotting_context(context)) ^ _context_keys
            nt.assert_true(not missing)

    def test_bad_context(self):

        with nt.assert_raises(ValueError):
            rcmod.plotting_context("i_am_not_a_context")

    def test_font_scale(self):

        notebook_ref = rcmod.plotting_context("notebook")
        notebook_big = rcmod.plotting_context("notebook", 2)

        font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
                     "xtick.labelsize", "ytick.labelsize", "font.size"]

        if LooseVersion(mpl.__version__) >= "3.0":
            font_keys.append("legend.title_fontsize")

        for k in font_keys:
            nt.assert_equal(notebook_ref[k] * 2, notebook_big[k])

    def test_rc_override(self):

        key, val = "grid.linewidth", 5
        rc = {key: val, "foo": "bar"}
        out = rcmod.plotting_context("talk", rc=rc)
        nt.assert_equal(out[key], val)
        nt.assert_not_in("foo", out)

    def test_set_context(self):

        for context in self.contexts:

            context_dict = rcmod.plotting_context(context)
            rcmod.set_context(context)
            self.assert_rc_params(context_dict)

    def test_context_context_manager(self):

        rcmod.set_context("notebook")
        orig_params = rcmod.plotting_context()
        context_params = rcmod.plotting_context("paper")

        with rcmod.plotting_context("paper"):
            self.assert_rc_params(context_params)
        self.assert_rc_params(orig_params)

        @rcmod.plotting_context("paper")
        def func():
            self.assert_rc_params(context_params)
        func()
        self.assert_rc_params(orig_params)


class TestPalette(object):

    def test_set_palette(self):

        rcmod.set_palette("deep")
        assert utils.get_color_cycle() == palettes.color_palette("deep", 10)

        rcmod.set_palette("pastel6")
        assert utils.get_color_cycle() == palettes.color_palette("pastel6", 6)

        rcmod.set_palette("dark", 4)
        assert utils.get_color_cycle() == palettes.color_palette("dark", 4)

        rcmod.set_palette("Set2", color_codes=True)
        assert utils.get_color_cycle() == palettes.color_palette("Set2", 8)


class TestFonts(object):

    def test_set_font(self):

        rcmod.set_theme(font="Verdana")

        _, ax = plt.subplots()
        ax.set_xlabel("foo")

        try:
            nt.assert_equal(ax.xaxis.label.get_fontname(),
                            "Verdana")
        except AssertionError:
            if has_verdana():
                raise
            else:
                raise nose.SkipTest("Verdana font is not present")
        finally:
            rcmod.set_theme()

    def test_set_serif_font(self):

        rcmod.set_theme(font="serif")

        _, ax = plt.subplots()
        ax.set_xlabel("foo")

        nt.assert_in(ax.xaxis.label.get_fontname(),
                     mpl.rcParams["font.serif"])

        rcmod.set_theme()

    def test_different_sans_serif(self):

        rcmod.set_theme()
        rcmod.set_style(rc={"font.sans-serif": ["Verdana"]})

        _, ax = plt.subplots()
        ax.set_xlabel("foo")

        try:
            nt.assert_equal(ax.xaxis.label.get_fontname(),
                            "Verdana")
        except AssertionError:
            if has_verdana():
                raise
            else:
                raise nose.SkipTest("Verdana font is not present")
        finally:
            rcmod.set_theme()


def has_verdana():
    """Helper to verify if Verdana font is present"""
    # This import is relatively lengthy, so to prevent its import for
    # testing other tests in this module not requiring this knowledge,
    # import font_manager here
    import matplotlib.font_manager as mplfm
    try:
        verdana_font = mplfm.findfont('Verdana', fallback_to_default=False)
    except:  # noqa
        # if https://github.com/matplotlib/matplotlib/pull/3435
        # gets accepted
        return False
    # otherwise check if not matching the logic for a 'default' one
    try:
        unlikely_font = mplfm.findfont("very_unlikely_to_exist1234",
                                       fallback_to_default=False)
    except:  # noqa
        # if matched verdana but not unlikely, Verdana must exist
        return True
    # otherwise -- if they match, must be the same default
    return verdana_font != unlikely_font
