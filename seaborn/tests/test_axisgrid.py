import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import nose.tools as nt
import numpy.testing as npt

from .. import axisgrid as ag
from ..palettes import color_palette
from ..distributions import kdeplot
from ..linearmodels import pointplot

rs = np.random.RandomState(0)


class TestFacetGrid(object):

    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           y=rs.gamma(4, size=60),
                           a=np.repeat(list("abc"), 20),
                           b=np.tile(list("mn"), 30),
                           c=rs.choice(list("tuv"), 60),
                           d=np.tile(list("abcdefghij"), 6)))

    def test_self_data(self):

        g = ag.FacetGrid(self.df)
        nt.assert_is(g.data, self.df)
        plt.close("all")

    def test_self_fig(self):

        g = ag.FacetGrid(self.df)
        nt.assert_is_instance(g.fig, plt.Figure)
        plt.close("all")

    def test_self_axes(self):

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        for ax in g.axes.flat:
            nt.assert_is_instance(ax, plt.Axes)

        plt.close("all")

    def test_axes_array_size(self):

        g1 = ag.FacetGrid(self.df)
        nt.assert_equal(g1.axes.shape, (1, 1))

        g2 = ag.FacetGrid(self.df, row="a")
        nt.assert_equal(g2.axes.shape, (3, 1))

        g3 = ag.FacetGrid(self.df, col="b")
        nt.assert_equal(g3.axes.shape, (1, 2))

        g4 = ag.FacetGrid(self.df, hue="c")
        nt.assert_equal(g4.axes.shape, (1, 1))

        g5 = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        nt.assert_equal(g5.axes.shape, (3, 2))

        for ax in g5.axes.flat:
            nt.assert_is_instance(ax, plt.Axes)

        plt.close("all")

    def test_col_wrap(self):

        g = ag.FacetGrid(self.df, col="d")
        nt.assert_equal(g.axes.shape, (1, 10))
        nt.assert_is(g.facet_axis(0, 8), g.axes[0, 8])

        g_wrap = ag.FacetGrid(self.df, col="d", col_wrap=4)
        nt.assert_equal(g_wrap.axes.shape, (10,))
        nt.assert_is(g_wrap.facet_axis(0, 8), g_wrap.axes[8])
        nt.assert_equal(g_wrap._ncol, 4)
        nt.assert_equal(g_wrap._nrow, 3)

        plt.close("all")

    def test_normal_axes(self):

        null = np.empty(0, object).flat

        g = ag.FacetGrid(self.df)
        npt.assert_array_equal(g._bottom_axes, g.axes.flat)
        npt.assert_array_equal(g._not_bottom_axes, null)
        npt.assert_array_equal(g._left_axes, g.axes.flat)
        npt.assert_array_equal(g._not_left_axes, null)
        npt.assert_array_equal(g._inner_axes, null)

        g = ag.FacetGrid(self.df, col="c")
        npt.assert_array_equal(g._bottom_axes, g.axes.flat)
        npt.assert_array_equal(g._not_bottom_axes, null)
        npt.assert_array_equal(g._left_axes, g.axes[:, 0].flat)
        npt.assert_array_equal(g._not_left_axes, g.axes[:, 1:].flat)
        npt.assert_array_equal(g._inner_axes, null)

        g = ag.FacetGrid(self.df, row="c")
        npt.assert_array_equal(g._bottom_axes, g.axes[-1, :].flat)
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:-1, :].flat)
        npt.assert_array_equal(g._left_axes, g.axes.flat)
        npt.assert_array_equal(g._not_left_axes, null)
        npt.assert_array_equal(g._inner_axes, null)

        g = ag.FacetGrid(self.df, col="a", row="c")
        npt.assert_array_equal(g._bottom_axes, g.axes[-1, :].flat)
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:-1, :].flat)
        npt.assert_array_equal(g._left_axes, g.axes[:, 0].flat)
        npt.assert_array_equal(g._not_left_axes, g.axes[:, 1:].flat)
        npt.assert_array_equal(g._inner_axes, g.axes[:-1, 1:].flat)

        plt.close("all")

    def test_wrapped_axes(self):

        null = np.empty(0, object).flat

        g = ag.FacetGrid(self.df, col="a", col_wrap=2)
        npt.assert_array_equal(g._bottom_axes,
                               g.axes[np.array([1, 2])].flat)
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:1].flat)
        npt.assert_array_equal(g._left_axes, g.axes[np.array([0, 2])].flat)
        npt.assert_array_equal(g._not_left_axes, g.axes[np.array([1])].flat)
        npt.assert_array_equal(g._inner_axes, null)

        plt.close("all")

    def test_figure_size(self):

        g = ag.FacetGrid(self.df, row="a", col="b")
        npt.assert_array_equal(g.fig.get_size_inches(), (6, 9))

        g = ag.FacetGrid(self.df, row="a", col="b", size=6)
        npt.assert_array_equal(g.fig.get_size_inches(), (12, 18))

        g = ag.FacetGrid(self.df, col="c", size=4, aspect=.5)
        npt.assert_array_equal(g.fig.get_size_inches(), (6, 4))

        plt.close("all")

    def test_figure_size_with_legend(self):

        g1 = ag.FacetGrid(self.df, col="a", hue="c", size=4, aspect=.5)
        npt.assert_array_equal(g1.fig.get_size_inches(), (6, 4))
        g1.set_legend()
        nt.assert_greater(g1.fig.get_size_inches()[0], 6)

        g2 = ag.FacetGrid(self.df, col="a", hue="c", size=4, aspect=.5,
                          legend_out=False)
        npt.assert_array_equal(g2.fig.get_size_inches(), (6, 4))
        g2.set_legend()
        npt.assert_array_equal(g2.fig.get_size_inches(), (6, 4))

        plt.close("all")

    def test_legend_data(self):

        g1 = ag.FacetGrid(self.df, hue="a")
        g1.map(plt.plot, "x", "y")
        palette = color_palette("husl", 3)

        nt.assert_equal(g1._legend.get_title().get_text(), "a")

        a_levels = sorted(self.df.a.unique())

        lines = g1._legend.get_lines()
        nt.assert_equal(len(lines), len(a_levels))

        for line, hue in zip(lines, palette):
            nt.assert_equal(line.get_color(), hue)

        labels = g1._legend.get_texts()
        nt.assert_equal(len(labels), len(a_levels))

        for label, level in zip(labels, a_levels):
            nt.assert_equal(label.get_text(), level)

        plt.close("all")

    def test_get_boolean_legend_data(self):

        self.df["b_bool"] = self.df.b == "m"
        g1 = ag.FacetGrid(self.df, hue="b_bool")
        g1.map(plt.plot, "x", "y")
        palette = color_palette("husl", 2)

        nt.assert_equal(g1._legend.get_title().get_text(), "b_bool")

        b_levels = sorted(map(str, self.df.b_bool.unique()))

        lines = g1._legend.get_lines()
        nt.assert_equal(len(lines), len(b_levels))

        for line, hue in zip(lines, palette):
            nt.assert_equal(line.get_color(), hue)

        labels = g1._legend.get_texts()
        nt.assert_equal(len(labels), len(b_levels))

        for label, level in zip(labels, b_levels):
            nt.assert_equal(label.get_text(), level)

        plt.close("all")

    def test_legend_options(self):

        g1 = ag.FacetGrid(self.df, hue="b")
        g1.map(plt.plot, "x", "y")

    def test_data_generator(self):

        g = ag.FacetGrid(self.df, row="a")
        d = list(g.facet_data())
        nt.assert_equal(len(d), 3)

        tup, data = d[0]
        nt.assert_equal(tup, (0, 0, 0))
        nt.assert_true((data["a"] == "a").all())

        tup, data = d[1]
        nt.assert_equal(tup, (1, 0, 0))
        nt.assert_true((data["a"] == "b").all())

        g = ag.FacetGrid(self.df, row="a", col="b")
        d = list(g.facet_data())
        nt.assert_equal(len(d), 6)

        tup, data = d[0]
        nt.assert_equal(tup, (0, 0, 0))
        nt.assert_true((data["a"] == "a").all())
        nt.assert_true((data["b"] == "m").all())

        tup, data = d[1]
        nt.assert_equal(tup, (0, 1, 0))
        nt.assert_true((data["a"] == "a").all())
        nt.assert_true((data["b"] == "n").all())

        tup, data = d[2]
        nt.assert_equal(tup, (1, 0, 0))
        nt.assert_true((data["a"] == "b").all())
        nt.assert_true((data["b"] == "m").all())

        g = ag.FacetGrid(self.df, hue="c")
        d = list(g.facet_data())
        nt.assert_equal(len(d), 3)
        tup, data = d[1]
        nt.assert_equal(tup, (0, 0, 1))
        nt.assert_true((data["c"] == "u").all())

        plt.close("all")

    def test_map(self):

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        g.map(plt.plot, "x", "y", linewidth=3)

        lines = g.axes[0, 0].lines
        nt.assert_equal(len(lines), 3)

        line1, _, _ = lines
        nt.assert_equal(line1.get_linewidth(), 3)
        x, y = line1.get_data()
        mask = (self.df.a == "a") & (self.df.b == "m") & (self.df.c == "t")
        npt.assert_array_equal(x, self.df.x[mask])
        npt.assert_array_equal(y, self.df.y[mask])

    def test_map_dataframe(self):

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        plot = lambda x, y, data=None, **kws: plt.plot(data[x], data[y], **kws)
        g.map_dataframe(plot, "x", "y", linestyle="--")

        lines = g.axes[0, 0].lines
        nt.assert_equal(len(lines), 3)

        line1, _, _ = lines
        nt.assert_equal(line1.get_linestyle(), "--")
        x, y = line1.get_data()
        mask = (self.df.a == "a") & (self.df.b == "m") & (self.df.c == "t")
        npt.assert_array_equal(x, self.df.x[mask])
        npt.assert_array_equal(y, self.df.y[mask])

    def test_set(self):

        g = ag.FacetGrid(self.df, row="a", col="b")
        xlim = (-2, 5)
        ylim = (3, 6)
        xticks = [-2, 0, 3, 5]
        yticks = [3, 4.5, 6]
        g.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
        for ax in g.axes.flat:
            npt.assert_array_equal(ax.get_xlim(), xlim)
            npt.assert_array_equal(ax.get_ylim(), ylim)
            npt.assert_array_equal(ax.get_xticks(), xticks)
            npt.assert_array_equal(ax.get_yticks(), yticks)

        plt.close("all")

    def test_set_titles(self):

        g = ag.FacetGrid(self.df, row="a", col="b")
        g.map(plt.plot, "x", "y")

        # Test the default titles
        nt.assert_equal(g.axes[0, 0].get_title(), "a = a | b = m")
        nt.assert_equal(g.axes[0, 1].get_title(), "a = a | b = n")
        nt.assert_equal(g.axes[1, 0].get_title(), "a = b | b = m")

        # Test a provided title
        g.set_titles("{row_var} == {row_name} \/ {col_var} == {col_name}")
        nt.assert_equal(g.axes[0, 0].get_title(), "a == a \/ b == m")
        nt.assert_equal(g.axes[0, 1].get_title(), "a == a \/ b == n")
        nt.assert_equal(g.axes[1, 0].get_title(), "a == b \/ b == m")

        # Test a single row
        g = ag.FacetGrid(self.df,  col="b")
        g.map(plt.plot, "x", "y")

        # Test the default titles
        nt.assert_equal(g.axes[0, 0].get_title(), "b = m")
        nt.assert_equal(g.axes[0, 1].get_title(), "b = n")

        plt.close("all")

    def test_set_titles_margin_titles(self):

        g = ag.FacetGrid(self.df, row="a", col="b", margin_titles=True)
        g.map(plt.plot, "x", "y")

        # Test the default titles
        nt.assert_equal(g.axes[0, 0].get_title(), "b = m")
        nt.assert_equal(g.axes[0, 1].get_title(), "b = n")
        nt.assert_equal(g.axes[1, 0].get_title(), "")

        # Test a provided title
        g.set_titles(col_template="{col_var} == {col_name}")
        nt.assert_equal(g.axes[0, 0].get_title(), "b == m")
        nt.assert_equal(g.axes[0, 1].get_title(), "b == n")
        nt.assert_equal(g.axes[1, 0].get_title(), "")

        plt.close("all")

    def test_set_ticklabels(self):

        g = ag.FacetGrid(self.df, row="a", col="b")
        g.map(plt.plot, "x", "y")
        xlab = [l.get_text() + "h" for l in g.axes[1, 0].get_xticklabels()]
        ylab = [l.get_text() for l in g.axes[1, 0].get_yticklabels()]

        g.set_xticklabels(xlab)
        g.set_yticklabels(rotation=90)

        got_x = [l.get_text() + "h" for l in g.axes[1, 1].get_xticklabels()]
        got_y = [l.get_text() for l in g.axes[0, 0].get_yticklabels()]
        npt.assert_array_equal(got_x, xlab)
        npt.assert_array_equal(got_y, ylab)

        x, y = np.arange(10), np.arange(10)
        df = pd.DataFrame(np.c_[x, y], columns=["x", "y"])
        g = ag.FacetGrid(df).map(pointplot, "x", "y")
        g.set_xticklabels(step=2)
        got_x = [int(l.get_text()) for l in g.axes[0, 0].get_xticklabels()]
        npt.assert_array_equal(x[::2], got_x)

    def test_subplot_kws(self):

        g = ag.FacetGrid(self.df, row="a", col="b", xlim=(0, 4), ylim=(-2, 3))
        nt.assert_equal(g.axes[0, 0].get_xlim(), (0, 4))
        nt.assert_equal(g.axes[0, 0].get_ylim(), (-2, 3))
        plt.close("all")

    def test_data_orders(self):

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")

        nt.assert_equal(g.row_names, list("abc"))
        nt.assert_equal(g.col_names, list("mn"))
        nt.assert_equal(g.hue_names, list("tuv"))

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c",
                         row_order=list("bca"),
                         col_order=list("nm"),
                         hue_order=list("vtu"))

        nt.assert_equal(g.row_names, list("bca"))
        nt.assert_equal(g.col_names, list("nm"))
        nt.assert_equal(g.hue_names, list("vtu"))
        plt.close("all")

    def test_palette(self):

        g = ag.FacetGrid(self.df, hue="c")
        nt.assert_equal(g._colors, color_palette("husl", 3))

        g = ag.FacetGrid(self.df, hue="c", palette="Set2")
        nt.assert_equal(g._colors, color_palette("Set2", 3))

        dict_pal = dict(t="red", u="green", v="blue")
        list_pal = color_palette(["red", "green", "blue"], 3)
        g = ag.FacetGrid(self.df, hue="c", palette=dict_pal)
        nt.assert_equal(g._colors, list_pal)

        list_pal = color_palette(["green", "blue", "red"], 3)
        g = ag.FacetGrid(self.df, hue="c", hue_order=list("uvt"),
                         palette=dict_pal)
        nt.assert_equal(g._colors, list_pal)

        plt.close("all")

    def test_dropna(self):

        df = self.df.copy()
        hasna = pd.Series(np.tile(np.arange(6), 10))
        hasna[hasna == 5] = np.nan
        df["hasna"] = hasna
        g = ag.FacetGrid(df, dropna=False, row="hasna")
        nt.assert_equal(g._not_na.sum(), 60)

        g = ag.FacetGrid(df, dropna=True, row="hasna")
        nt.assert_equal(g._not_na.sum(), 50)

        plt.close("all")

    @classmethod
    def teardown_class(cls):
        """Ensure that all figures are closed on exit."""
        plt.close("all")


class TestJointGrid(object):

    rs = np.random.RandomState(sum(map(ord, "JointGrid")))
    x = rs.randn(100)
    y = rs.randn(100)
    x_na = x.copy()
    x_na[10] = np.nan
    x_na[20] = np.nan
    data = pd.DataFrame(dict(x=x, y=y, x_na=x_na))

    def test_margin_grid_from_arrays(self):

        g = ag.JointGrid(self.x, self.y)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)
        plt.close("all")

    def test_margin_grid_from_series(self):

        g = ag.JointGrid(self.data.x, self.data.y)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)
        plt.close("all")

    def test_margin_grid_from_dataframe(self):

        g = ag.JointGrid("x", "y", self.data)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)
        plt.close("all")

    def test_margin_grid_axis_labels(self):

        g = ag.JointGrid("x", "y", self.data)

        xlabel, ylabel = g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()
        nt.assert_equal(xlabel, "x")
        nt.assert_equal(ylabel, "y")

        g.set_axis_labels("x variable", "y variable")
        xlabel, ylabel = g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()
        nt.assert_equal(xlabel, "x variable")
        nt.assert_equal(ylabel, "y variable")
        plt.close("all")

    def test_dropna(self):

        g = ag.JointGrid("x_na", "y", self.data, dropna=False)
        nt.assert_equal(len(g.x), len(self.x_na))

        g = ag.JointGrid("x_na", "y", self.data, dropna=True)
        nt.assert_equal(len(g.x), pd.notnull(self.x_na).sum())
        plt.close("all")

    def test_axlims(self):

        lim = (-3, 3)
        g = ag.JointGrid("x", "y", self.data, xlim=lim, ylim=lim)

        nt.assert_equal(g.ax_joint.get_xlim(), lim)
        nt.assert_equal(g.ax_joint.get_ylim(), lim)

        nt.assert_equal(g.ax_marg_x.get_xlim(), lim)
        nt.assert_equal(g.ax_marg_y.get_ylim(), lim)

    def test_marginal_ticks(self):

        g = ag.JointGrid("x", "y", self.data)
        nt.assert_true(~len(g.ax_marg_x.get_xticks()))
        nt.assert_true(~len(g.ax_marg_y.get_yticks()))
        plt.close("all")

    def test_bivariate_plot(self):

        g = ag.JointGrid("x", "y", self.data)
        g.plot_joint(plt.plot)

        x, y = g.ax_joint.lines[0].get_xydata().T
        npt.assert_array_equal(x, self.x)
        npt.assert_array_equal(y, self.y)
        plt.close("all")

    def test_univariate_plot(self):

        g = ag.JointGrid("x", "x", self.data)
        g.plot_marginals(kdeplot)

        _, y1 = g.ax_marg_x.lines[0].get_xydata().T
        y2, _ = g.ax_marg_y.lines[0].get_xydata().T
        npt.assert_array_equal(y1, y2)
        plt.close("all")

    def test_plot(self):

        g = ag.JointGrid("x", "x", self.data)
        g.plot(plt.plot, kdeplot)

        x, y = g.ax_joint.lines[0].get_xydata().T
        npt.assert_array_equal(x, self.x)
        npt.assert_array_equal(y, self.x)

        _, y1 = g.ax_marg_x.lines[0].get_xydata().T
        y2, _ = g.ax_marg_y.lines[0].get_xydata().T
        npt.assert_array_equal(y1, y2)

        plt.close("all")

    def test_annotate(self):

        g = ag.JointGrid("x", "y", self.data)
        rp = stats.pearsonr(self.x, self.y)

        g.annotate(stats.pearsonr)
        annotation = g.ax_joint.legend_.texts[0].get_text()
        nt.assert_equal(annotation, "pearsonr = %.2g; p = %.2g" % rp)

        g.annotate(stats.pearsonr, stat="correlation")
        annotation = g.ax_joint.legend_.texts[0].get_text()
        nt.assert_equal(annotation, "correlation = %.2g; p = %.2g" % rp)

        def rsquared(x, y):
            return stats.pearsonr(x, y)[0] ** 2

        r2 = rsquared(self.x, self.y)
        g.annotate(rsquared)
        annotation = g.ax_joint.legend_.texts[0].get_text()
        nt.assert_equal(annotation, "rsquared = %.2g" % r2)

        template = "{stat} = {val:.3g} (p = {p:.3g})"
        g.annotate(stats.pearsonr, template=template)
        annotation = g.ax_joint.legend_.texts[0].get_text()
        nt.assert_equal(annotation, template.format(stat="pearsonr",
                                                    val=rp[0], p=rp[1]))

        plt.close("all")

    def test_space(self):

        g = ag.JointGrid("x", "y", self.data, space=0)

        joint_bounds = g.ax_joint.bbox.bounds
        marg_x_bounds = g.ax_marg_x.bbox.bounds
        marg_y_bounds = g.ax_marg_y.bbox.bounds

        nt.assert_equal(joint_bounds[2], marg_x_bounds[2])
        nt.assert_equal(joint_bounds[3], marg_y_bounds[3])

    @classmethod
    def teardown_class(cls):
        """Ensure that all figures are closed on exit."""
        plt.close("all")
