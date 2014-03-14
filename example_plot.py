import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import moss

def lmplot(ax):

    n = 80
    c = "#222222"
    rs = np.random.RandomState(5)
    x = rs.normal(4, 1, n)
    y = 2 + 1.5 * x + rs.normal(0, 3, n)
    ax.plot(x, y, "o", c=c, alpha=.8)

    xx = np.linspace(1 + 1e-9, 7 - 1e-9, 100)
    lmpred = lambda x, y: np.polyval(np.polyfit(x, y, 1), xx)
    yy = lmpred(x, y)
    ax.plot(xx, yy, c=c)
    boots = moss.bootstrap(x, y, func=lmpred, n_boot=100)
    ci = moss.percentiles(boots, [2.5, 97.5], 0)
    ax.fill_between(xx, *ci, alpha=.15, color=c)
    ax.set_title("lmplot()")


def tsplot(ax):

    n = 20
    t = 10
    ax.set_xlim(0, t)
    x = np.linspace(0, t, 100)
    s = np.array([stats.gamma.pdf(x, a) for a in [3, 5, 7]])
    d = s[:, np.newaxis, :]
    rs = np.random.RandomState(24)

    d = d * np.array([1, -1])[rs.binomial(1, .3, 3)][:, np.newaxis, np.newaxis]
    d = d + rs.normal(0, .15, (3, n))[:, :, np.newaxis]
    d = d + rs.uniform(0, .25, 3)[:, np.newaxis, np.newaxis]
    d *= 10
    d = d.transpose((1, 2, 0))

    sns.tsplot(d, time=x, ax=ax)
    ax.set_title("tsplot()")


def violinplot(ax):

    n = 40
    p = 8
    rs = np.random.RandomState(8)
    d = rs.normal(0, 1, (n, p))
    d += np.log(np.arange(1, p + 1)) * -5 + 10

    sns.violinplot(d, inner="points", ax=ax)
    ax.set_title("violinplot()")


def interactplot(ax):

    rs = np.random.RandomState(11)

    n = 80
    x1 = rs.randn(n)
    x2 = x1 / 5 + rs.randn(n)
    b0, b1, b2, b3 = 1.5, 4, -1, 3
    y = b0  + b1 * x1 + b2 * x2 + b3 * x1 * x2 + rs.randn(n)

    sns.interactplot(x1, x2, y, colorbar=False, ax=ax)
    ax.set_title("interactplot()")


def corrplot(ax):

    rs = np.random.RandomState(0)
    x0, x1 = rs.randn(2, 60)
    x2, x3 = rs.multivariate_normal([0, 0], [(1, -.5), (-.5, 1)], 60).T
    x2 += x0 / 8
    x4 = x1 + rs.randn(60) * 2
    data = np.c_[x0, x1, x2, x3, x4]

    sns.corrplot(data, ax=ax)
    ax.set_title("corrplot()", verticalalignment="top")


def distplot_hist(ax):

    ax.set_xlim(0, 1)
    ax.set_xticklabels([])

    g = sns.color_palette("Set2", desat=.75)[0]
    n = 1000
    rs = np.random.RandomState(0)
    d = rs.beta(8, 13, n)

    sns.distplot(d, color=g, ax=ax)
    sns.despine(ax=ax)
    ax.set_title("distplot()")


def distplot_kde(ax):

    ax.set_xlim(0, 1)

    p = sns.color_palette("Set2", desat=.75)[2]
    n = 80
    rs = np.random.RandomState(0)
    d = rs.beta(50, 25, n)

    sns.distplot(d, hist=False, rug=True, color=p,
                 kde_kws=dict(shade=True), ax=ax)
    sns.despine(ax=ax)


if __name__ == "__main__":

    f = plt.figure(figsize=(10, 12))
    gs = plt.GridSpec(6, 2)
    np.random.seed(0)

    # Linear regression
    ax = plt.subplot(gs[:2, 0])
    lmplot(ax)

    # Timeseries plot
    ax = plt.subplot(gs[:2, 1])
    tsplot(ax)

    # Violin plots
    sns.set(style="whitegrid")
    ax = plt.subplot(gs[2:4, 0])
    violinplot(ax)

    # Continuous interaction
    sns.set(style="darkgrid")
    ax = plt.subplot(gs[2:4, 1])
    interactplot(ax)

    # Correlation matrix
    ax = plt.subplot(gs[4:, 0])
    corrplot(ax)

    # Beta distributions
    sns.set(style="nogrid")
    ax = plt.subplot(gs[4, 1])
    distplot_hist(ax)

    ax = plt.subplot(gs[5, 1], sharey=ax)
    distplot_kde(ax)

    # Save the plot
    f.tight_layout()
    png_template = os.path.dirname(__file__) + "/%s.png"
    f.savefig(png_template % "example_plot")

    # Carousel images
    plots = ["lmplot", "tsplot", "violinplot", "interactplot",
             "corrplot", "distplot"]
    styles = {p: "darkgrid" for p in plots}
    styles["violinplot"] = "whitegrid"
    styles["distplot"] = "nogrid"

    figsize = (6, 4.2)
    for plot in plots:
        sns.set_axes_style(styles[plot], "notebook")

        if plot == "distplot":
            f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=figsize)
            distplot_hist(ax1)
            distplot_kde(ax2)
        else:
            plot_func = locals()[plot]
            f, ax = plt.subplots(figsize=figsize)
            plot_func(ax)

        f.tight_layout()
        f.savefig(png_template % plot)
