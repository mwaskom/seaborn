import numpy as np
import matplotlib as mpl
import nose.tools as nt

from .. import rcmod


class RCParamTester(object):

    def flatten_list(self, orig_list):

        iter_list = map(np.atleast_1d, orig_list)
        flat_list = [item for sublist in iter_list for item in sublist]
        return flat_list

    def mpl_matches(self, params):

        matches = [v == mpl.rcParams[k] for k, v in params.items()]
        return all(self.flatten_list(matches))


class TestAxesStyle(RCParamTester):

    styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

    def test_default_return(self):

        current = rcmod.axes_style()
        nt.assert_true(self.mpl_matches(current))

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

    def test_back_compat(self):

        nogrid = rcmod.axes_style("nogrid")
        white = rcmod.axes_style("white")
        nt.assert_equal(nogrid, white)

    def test_set_style(self):

        for style in self.styles:

            style_dict = rcmod.axes_style(style)
            rcmod.set_style(style)
            nt.assert_true(self.mpl_matches(style_dict))

    def test_style_context_manager(self):

        rcmod.set_style("darkgrid")
        orig_params = rcmod.axes_style()
        with rcmod.axes_style("whitegrid"):
            context_params = rcmod.axes_style("whitegrid")
            nt.assert_true(self.mpl_matches(context_params))
        nt.assert_true(self.mpl_matches(orig_params))

    def test_style_context_independence(self):

        nt.assert_true(set(rcmod._style_keys) ^ set(rcmod._context_keys))

    def test_set_rc(self):

        rcmod.set(rc={"lines.linewidth": 4})
        nt.assert_equal(mpl.rcParams["lines.linewidth"], 4)
        rcmod.set()

    def test_reset_defaults(self):

        rcmod.reset_defaults()
        nt.assert_equal(mpl.rcParamsDefault, mpl.rcParams)
        rcmod.set()

    def test_reset_orig(self):

        rcmod.reset_orig()
        nt.assert_equal(mpl.rcParamsOrig, mpl.rcParams)
        rcmod.set()


class TestPlottingContext(RCParamTester):

    contexts = ["paper", "notebook", "talk", "poster"]

    def test_default_return(self):

        current = rcmod.plotting_context()
        nt.assert_true(self.mpl_matches(current))

    def test_key_usage(self):

        _context_keys = set(rcmod._context_keys)
        for context in self.contexts:
            missing = set(rcmod.plotting_context(context)) ^ _context_keys
            nt.assert_true(not missing)

    def test_bad_context(self):

        with nt.assert_raises(ValueError):
            rcmod.plotting_context("i_am_not_a_context")

    def test_rc_override(self):

        key, val = "grid.linewidth", 5
        rc = {key: val, "foo": "bar"}
        out = rcmod.plotting_context("talk", rc)
        nt.assert_equal(out[key], val)
        nt.assert_not_in("foo", out)

    def test_set_context(self):

        for context in self.contexts:

            context_dict = rcmod.plotting_context(context)
            rcmod.set_context(context)
            nt.assert_true(self.mpl_matches(context_dict))

    def test_context_context_manager(self):

        rcmod.set_context("notebook")
        orig_params = rcmod.plotting_context()
        with rcmod.plotting_context("paper"):
            context_params = rcmod.plotting_context("paper")
            nt.assert_true(self.mpl_matches(context_params))
        nt.assert_true(self.mpl_matches(orig_params))
