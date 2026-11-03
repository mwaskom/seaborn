"""Subset semantic levels should filter scatter points (issue #3601)."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import seaborn as sns


@pytest.fixture
def ordered_scatter_data():
    return pd.DataFrame({
        "x": [0, 1, 2, 3, 4],
        "y": [1, 2, 3, 4, 5],
        "category": ["a", "b", "c", "a", "b"],
    })


@pytest.mark.parametrize(
    "semantic,order_key",
    [("hue", "hue_order"), ("style", "style_order"),
     ("size", "size_order")],
)
def test_scatterplot_restricted_semantic_order_filters_points(
    ordered_scatter_data, semantic, order_key,
):
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            data=ordered_scatter_data,
            x="x",
            y="y",
            ax=ax,
            **{semantic: "category", order_key: ["a", "c"]},
        )
        points = ax.collections[0].get_offsets()
        assert points[:, 0].tolist() == [0.0, 2.0, 3.0]
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_scatterplot_restricted_hue_and_style_intersection(ordered_scatter_data):
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            data=ordered_scatter_data, x="x", y="y", ax=ax,
            hue="category", style="category",
            hue_order=["a", "b"], style_order=["b", "c"],
        )
        assert ax.collections[0].get_offsets()[:, 0].tolist() == [1.0, 4.0]
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_scatterplot_complete_order_retains_all_observations(ordered_scatter_data):
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            data=ordered_scatter_data, x="x", y="y", ax=ax,
            hue="category", style="category",
            hue_order=["a", "b", "c"], style_order=["c", "b", "a"],
        )
        assert len(ax.collections[0].get_offsets()) == 5
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_scatterplot_numeric_continuous_hue_preserves_all_observations(
    ordered_scatter_data,
):
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            data=ordered_scatter_data.assign(numeric=[0, 1, 2, 3, 4]),
            x="x", y="y", hue="numeric", ax=ax,
        )
        assert len(ax.collections[0].get_offsets()) == 5
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_relplot_scatter_restricted_style_filters_points(ordered_scatter_data):
    chart = sns.relplot(
        data=ordered_scatter_data,
        x="x", y="y", style="category", style_order=["a", "c"],
        kind="scatter",
    )
    try:
        assert chart.ax.collections[0].get_offsets()[:, 0].tolist() == [
            0.0, 2.0, 3.0,
        ]
        chart.figure.canvas.draw()
    finally:
        plt.close(chart.figure)


@pytest.mark.parametrize("semantic,order_key", [
    ("hue", "hue_order"), ("style", "style_order"), ("size", "size_order"),
])
def test_scatterplot_order_excludes_all_observations(
    ordered_scatter_data, semantic, order_key,
):
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            data=ordered_scatter_data, x="x", y="y", ax=ax,
            **{semantic: "category", order_key: ["absent"]},
        )
        assert not any(len(c.get_offsets()) for c in ax.collections)
    finally:
        plt.close(fig)


def test_scatterplot_all_missing_style_does_not_fail():
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(
            x=[0, 1], y=[1, 2], style=[None, None],
            style_order=["a"], ax=ax,
        )
        assert not any(len(c.get_offsets()) for c in ax.collections)
    finally:
        plt.close(fig)
