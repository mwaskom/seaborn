"""Regression tests for axis labels with hidden tick labels (GH #3530)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import seaborn as sns


TICK_LABELS_HIDDEN = {
    "xtick.labelbottom": False,
    "xtick.labeltop": False,
    "ytick.labelleft": False,
    "ytick.labelright": False,
}
DATA = pd.DataFrame({
    "x": [0, 1, 2, 0, 1, 2],
    "y": [1, 2, 3, 4, 5, 6],
    "group": ["A"] * 3 + ["B"] * 3,
})


@pytest.mark.parametrize("hide_rc_ticklabels", [True, False])
def test_scatterplot_single_axis_labels_visible_without_tick_labels(
    hide_rc_ticklabels,
):
    rc = TICK_LABELS_HIDDEN if hide_rc_ticklabels else {}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots()
        try:
            sns.scatterplot(DATA, x="x", y="y", ax=ax)
            fig.canvas.draw()
            assert ax.get_xlabel() == "x"
            assert ax.get_ylabel() == "y"
            assert ax.xaxis.label.get_visible()
            assert ax.yaxis.label.get_visible()
        finally:
            plt.close(fig)


@pytest.mark.parametrize("hide_rc_ticklabels", [True, False])
def test_scatterplot_shared_axes_keep_only_outer_labels(hide_rc_ticklabels):
    rc = TICK_LABELS_HIDDEN if hide_rc_ticklabels else {}
    with mpl.rc_context(rc):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        try:
            for ax in axes.flat:
                sns.scatterplot(DATA, x="x", y="y", ax=ax)
            fig.canvas.draw()
            for row in range(2):
                for col in range(2):
                    ax = axes[row, col]
                    assert ax.xaxis.label.get_visible() is (row == 1)
                    assert ax.yaxis.label.get_visible() is (col == 0)
        finally:
            plt.close(fig)


@pytest.mark.parametrize("hide_rc_ticklabels", [True, False])
def test_relplot_facet_outer_axis_labels_visible(hide_rc_ticklabels):
    rc = TICK_LABELS_HIDDEN if hide_rc_ticklabels else {}
    with mpl.rc_context(rc):
        grid = sns.relplot(DATA, x="x", y="y", col="group", kind="scatter")
        try:
            grid.figure.canvas.draw()
            assert all(ax.xaxis.label.get_visible() for ax in grid.axes.flat)
            assert grid.axes.flat[0].yaxis.label.get_visible()
            assert not grid.axes.flat[1].yaxis.label.get_visible()
        finally:
            plt.close(grid.figure)


def test_scatterplot_labels_visible_with_no_ticks():
    fig, ax = plt.subplots()
    try:
        ax.set(xticks=[], yticks=[])
        sns.scatterplot(DATA, x="x", y="y", ax=ax)
        fig.canvas.draw()
        assert ax.xaxis.label.get_visible()
        assert ax.yaxis.label.get_visible()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("func", [sns.lineplot, sns.boxplot])
def test_other_axes_level_plots_preserve_labels_with_hidden_ticks(func):
    with mpl.rc_context(TICK_LABELS_HIDDEN):
        fig, ax = plt.subplots()
        try:
            func(DATA, x="x", y="y", ax=ax)
            fig.canvas.draw()
            assert ax.xaxis.label.get_visible()
            assert ax.yaxis.label.get_visible()
        finally:
            plt.close(fig)


def test_independent_facet_axes_preserve_both_labels_with_hidden_ticks():
    with mpl.rc_context(TICK_LABELS_HIDDEN):
        grid = sns.relplot(
            DATA, x="x", y="y", col="group", kind="scatter",
            facet_kws={"sharex": False, "sharey": False},
        )
        try:
            grid.figure.canvas.draw()
            for ax in grid.axes.flat:
                assert ax.xaxis.label.get_visible()
            # FacetGrid suppresses duplicate y-label text even when axes
            # are independent, without necessarily hiding the text artist.
            assert grid.axes.flat[0].yaxis.label.get_visible()
            assert grid.axes.flat[1].get_ylabel() == ""
        finally:
            plt.close(grid.figure)
