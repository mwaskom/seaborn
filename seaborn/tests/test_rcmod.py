import matplotlib as mpl
import nose.tools as nt

from .. import rcmod


class TestAxesStyle(object):

    styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

    def mpl_matches(self, params):

        matches = [v == mpl.rcParams[k] for k, v in params.items()]
        return all(matches)

    def test_default_return(self):

        current = rcmod.axes_style()
        nt.assert_true(self.mpl_matches(current))

    def test_key_usage(self):

        style_keys = set(rcmod.style_keys)
        for style in self.styles:
            nt.assert_true(not set(rcmod.axes_style(style)) ^ style_keys)

    def test_rc_override(self):

        rc = {"axes.facecolor": "blue", "foo.notaparam": "bar"}
        out = rcmod.axes_style("darkgrid", rc)
        nt.assert_equal(out["axes.facecolor"], "blue")
        nt.assert_not_in("foo.notaparam", out)

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
