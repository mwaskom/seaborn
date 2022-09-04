from pathlib import Path
import warnings

from jinja2 import Environment
import yaml

import numpy as np
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so


TEMPLATE = """
:notoc:

.. _tutorial:

User guide and tutorial
=======================
{% for section in sections %}
{{ section.header }}
{% for page in section.pages %}
.. grid:: 1
  :gutter: 2

  .. grid-item-card::

    .. grid:: 2

      .. grid-item::
        :columns: 3

        .. image:: ./tutorial/{{ page }}.svg
          :target: ./tutorial/{{ page }}.html

      .. grid-item::
        :columns: 9
        :margin: auto

        .. toctree::
          :maxdepth: 2

          tutorial/{{ page }}
{% endfor %}
{% endfor %}
"""


def main(app):

    content_yaml = Path(app.builder.srcdir) / "tutorial.yaml"
    tutorial_rst = Path(app.builder.srcdir) / "tutorial.rst"

    tutorial_dir = Path(app.builder.srcdir) / "tutorial"
    tutorial_dir.mkdir(exist_ok=True)

    with open(content_yaml) as fid:
        sections = yaml.load(fid, yaml.BaseLoader)

    for section in sections:
        title = section["title"]
        section["header"] = "\n".join([title, "-" * len(title)]) if title else ""

    env = Environment().from_string(TEMPLATE)
    content = env.render(sections=sections)

    with open(tutorial_rst, "w") as fid:
        fid.write(content)

    for section in sections:
        for page in section["pages"]:
            if (
                not (svg_path := tutorial_dir / f"{page}.svg").exists()
                or svg_path.stat().st_mtime < Path(__file__).stat().st_mtime
            ):
                write_thumbnail(svg_path, page)


def write_thumbnail(svg_path, page):

    with (
        sns.axes_style("dark"),
        sns.plotting_context("notebook"),
        sns.color_palette("deep")
    ):
        fig = globals()[page]()
        for ax in fig.axes:
            ax.set(xticklabels=[], yticklabels=[], xlabel="", ylabel="", title="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        fig.savefig(svg_path, format="svg")


def introduction():

    tips = sns.load_dataset("tips")
    fmri = sns.load_dataset("fmri").query("region == 'parietal'")
    penguins = sns.load_dataset("penguins")

    f = mpl.figure.Figure(figsize=(5, 5))
    with sns.axes_style("whitegrid"):
        f.subplots(2, 2)

    sns.scatterplot(
        tips, x="total_bill", y="tip", hue="sex", size="size",
        alpha=.75, palette=["C0", ".5"], legend=False, ax=f.axes[0],
    )
    sns.kdeplot(
        tips.query("size != 5"), x="total_bill", hue="size",
        palette="blend:C0,.5", fill=True, linewidth=.5,
        legend=False, common_norm=False, ax=f.axes[1],
    )
    sns.lineplot(
        fmri, x="timepoint", y="signal", hue="event",
        errorbar=("se", 2), legend=False, palette=["C0", ".5"], ax=f.axes[2],
    )
    sns.boxplot(
        penguins, x="bill_depth_mm", y="species", hue="sex",
        whiskerprops=dict(linewidth=1.5), medianprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5), capprops=dict(linewidth=0),
        width=.5, palette=["C0", ".8"], whis=5, ax=f.axes[3],
    )
    f.axes[3].legend_ = None
    for ax in f.axes:
        ax.set(xticks=[], yticks=[])
    return f


def function_overview():

    from matplotlib.patches import FancyBboxPatch

    f = mpl.figure.Figure(figsize=(7, 5))
    with sns.axes_style("white"):
        ax = f.subplots()
    f.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))

    deep = sns.color_palette("deep")
    colors = dict(relational=deep[0], distributions=deep[1], categorical=deep[2])
    dark = sns.color_palette("dark")
    text_colors = dict(relational=dark[0], distributions=dark[1], categorical=dark[2])

    functions = dict(
        relational=["scatterplot", "lineplot"],
        distributions=["histplot", "kdeplot", "ecdfplot", "rugplot"],
        categorical=[
            "stripplot", "swarmplot", "boxplot", "violinplot", "pointplot", "barplot"
        ],
    )
    pad, w, h = .06, .2, .15
    xs, y = np.arange(0, 1, 1 / 3) + pad * 1.05, .7
    for x, mod in zip(xs, functions):
        color = colors[mod] + (.2,)
        text_color = text_colors[mod]
        ax.add_artist(FancyBboxPatch((x, y), w, h, f"round,pad={pad}", color="white"))
        ax.add_artist(FancyBboxPatch(
            (x, y), w, h, f"round,pad={pad}",
            linewidth=1, edgecolor=text_color, facecolor=color,
        ))
        ax.text(
            x + w / 2, y + h / 2, f"{mod[:3]}plot\n({mod})",
            ha="center", va="center", size=20, color=text_color
        )
        for i, func in enumerate(functions[mod]):
            x_i, y_i = x + w / 2, y - i * .1 - h / 2 - pad
            xy = x_i - w / 2, y_i - pad / 3
            ax.add_artist(
                FancyBboxPatch(xy, w, h / 4, f"round,pad={pad / 3}", color="white")
            )
            ax.add_artist(FancyBboxPatch(
                xy, w, h / 4, f"round,pad={pad / 3}",
                linewidth=1, edgecolor=text_color, facecolor=color
            ))
            ax.text(x_i, y_i, func, ha="center", va="center", size=16, color=text_color)
        ax.plot([x_i, x_i], [y, y_i], zorder=-100, color=text_color, lw=1)
    return f


def data_structure():

    f = mpl.figure.Figure(figsize=(7, 5))
    gs = mpl.gridspec.GridSpec(
        figure=f, ncols=6, nrows=2, height_ratios=(1, 20),
        left=0, right=.35, bottom=0, top=.9, wspace=.1, hspace=.01
    )
    colors = [c + (.5,) for c in sns.color_palette("deep")]
    f.add_subplot(gs[0, :], facecolor=".8")
    for i in range(gs.ncols):
        f.add_subplot(gs[1:, i], facecolor=colors[i])

    gs = mpl.gridspec.GridSpec(
        figure=f, ncols=2, nrows=2, height_ratios=(1, 8), width_ratios=(1, 11),
        left=.4, right=1, bottom=.2, top=.8, wspace=.015, hspace=.02
    )
    f.add_subplot(gs[0, 1:], facecolor=colors[2])
    f.add_subplot(gs[1:, 0], facecolor=colors[1])
    f.add_subplot(gs[1, 1], facecolor=colors[0])
    return f


def error_bars():

    diamonds = sns.load_dataset("diamonds")
    with sns.axes_style("whitegrid"):
        g = sns.catplot(
            diamonds, x="carat", y="clarity", hue="clarity", kind="point",
            errorbar=("sd", .5), join=False, legend=False, facet_kws={"despine": False},
            palette="ch:s=-.2,r=-.2,d=.4,l=.6_r", scale=.75, capsize=.3,
        )
    g.ax.yaxis.set_inverted(False)
    return g.figure


def properties():

    f = mpl.figure.Figure(figsize=(5, 5))

    x = np.arange(1, 11)
    y = np.zeros_like(x)

    p = so.Plot(x, y)
    ps = 14
    plots = [
        p.add(so.Dot(pointsize=ps), color=map(str, x)),
        p.add(so.Dot(color=".3", pointsize=ps), alpha=x),
        p.add(so.Dot(color=".9", pointsize=ps, edgewidth=2), edgecolor=x),
        p.add(so.Dot(color=".3"), pointsize=x).scale(pointsize=(4, 18)),
        p.add(so.Dot(pointsize=ps, color=".9", edgecolor=".2"), edgewidth=x),
        p.add(so.Dot(pointsize=ps, color=".3"), marker=map(str, x)),
        p.add(so.Dot(pointsize=ps, color=".3", marker="x"), stroke=x),
    ]

    with sns.axes_style("ticks"):
        axs = f.subplots(len(plots))
    for p, ax in zip(plots, axs):
        p.on(ax).plot()
        ax.set(xticks=x, yticks=[], xticklabels=[], ylim=(-.2, .3))
        sns.despine(ax=ax, left=True)
    f.legends = []
    return f


def objects_interface():

    f = mpl.figure.Figure(figsize=(5, 4))
    C = sns.color_palette("deep")
    ax = f.subplots()
    fontsize = 22
    rects = [((.135, .50), .69), ((.275, .38), .26), ((.59, .38), .40)]
    for i, (xy, w) in enumerate(rects):
        ax.add_artist(mpl.patches.Rectangle(xy, w, .09, color=C[i], alpha=.2, lw=0))
    ax.text(0, .52, "Plot(data, 'x', 'y', color='var1')", size=fontsize, color=".2")
    ax.text(0, .40, ".add(Dot(alpha=.5), marker='var2')", size=fontsize, color=".2")
    annots = [
        ("Mapped\nin all layers", (.48, .62), (0, 55)),
        ("Set directly", (.41, .35), (0, -55)),
        ("Mapped\nin this layer", (.80, .35), (0, -55)),
    ]
    for i, (text, xy, xytext) in enumerate(annots):
        ax.annotate(
            text, xy, xytext,
            textcoords="offset points", fontsize=18, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color=C[i]), color=C[i],
        )
    ax.set_axis_off()
    f.subplots_adjust(0, 0, 1, 1)

    return f


def relational():

    mpg = sns.load_dataset("mpg")
    with sns.axes_style("ticks"):
        g = sns.relplot(
            data=mpg, x="horsepower", y="mpg", size="displacement", hue="weight",
            sizes=(50, 500), hue_norm=(2000, 4500), alpha=.75, legend=False,
            palette="ch:start=-.5,rot=.7,dark=.3,light=.7_r",
        )
    g.figure.set_size_inches(5, 5)
    return g.figure


def distributions():

    penguins = sns.load_dataset("penguins").dropna()
    with sns.axes_style("white"):
        g = sns.displot(
            penguins, x="flipper_length_mm", row="island",
            binwidth=4, kde=True, line_kws=dict(linewidth=2), legend=False,
        )
    sns.despine(left=True)
    g.figure.set_size_inches(5, 5)
    return g.figure


def categorical():

    penguins = sns.load_dataset("penguins").dropna()
    with sns.axes_style("whitegrid"):
        g = sns.catplot(
            penguins, x="sex", y="body_mass_g", hue="island", col="sex",
            kind="box", whis=np.inf, legend=False, sharex=False,
        )
    sns.despine(left=True)
    g.figure.set_size_inches(5, 5)
    return g.figure


def regression():

    anscombe = sns.load_dataset("anscombe")
    with sns.axes_style("white"):
        g = sns.lmplot(
            anscombe, x="x", y="y", hue="dataset", col="dataset", col_wrap=2,
            scatter_kws=dict(edgecolor=".2", facecolor=".7", s=80),
            line_kws=dict(lw=4), ci=None,
        )
    g.set(xlim=(2, None), ylim=(2, None))
    g.figure.set_size_inches(5, 5)
    return g.figure


def axis_grids():

    penguins = sns.load_dataset("penguins").sample(200, random_state=0)
    with sns.axes_style("ticks"):
        g = sns.pairplot(
            penguins.drop("flipper_length_mm", axis=1),
            diag_kind="kde", diag_kws=dict(fill=False),
            plot_kws=dict(s=40, fc="none", ec="C0", alpha=.75, linewidth=.75),
        )
    g.figure.set_size_inches(5, 5)
    return g.figure


def aesthetics():

    f = mpl.figure.Figure(figsize=(5, 5))
    for i, style in enumerate(["darkgrid", "white", "ticks", "whitegrid"], 1):
        with sns.axes_style(style):
            ax = f.add_subplot(2, 2, i)
        ax.set(xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
    sns.despine(ax=f.axes[1])
    sns.despine(ax=f.axes[2])
    return f


def color_palettes():

    f = mpl.figure.Figure(figsize=(5, 5))
    palettes = ["deep", "husl", "gray", "ch:", "mako", "vlag", "icefire"]
    axs = f.subplots(len(palettes))
    x = np.arange(10)
    for ax, name in zip(axs, palettes):
        cmap = mpl.colors.ListedColormap(sns.color_palette(name, x.size))
        ax.pcolormesh(x[None, :], linewidth=.5, edgecolor="w", alpha=.8, cmap=cmap)
        ax.set_axis_off()
    return f


def setup(app):
    app.connect("builder-inited", main)
