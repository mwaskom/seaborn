"""Regression coverage for nominal color scales with mixed int/float layers.

GH #3375: concatenation can upcast integer categories to floats when a
layer adds missing values; both must resolve to the same category unit.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn.objects as so


@pytest.mark.parametrize("categories", [[1, 2, 3], [0, 1, 2], [2, 4, 6]])
@pytest.mark.parametrize("palette", ["deep", so.Nominal()])
def test_nominal_color_multiple_layers_with_integer_categories(categories, palette):
    fig = None
    try:
        plotter = (
            so.Plot(
                x=[0, 1, 2],
                y=[1, 2, 3],
                color=categories,
            )
            .add(so.Dot())
            .add(so.Dot(), x=[0], y=[1], color=[categories[0]])
            .scale(color=palette)
            .plot()
        )
        fig = plotter._figure
        fig.canvas.draw()
        assert len(fig.axes[0].collections) >= 2
        colors = np.asarray(fig.axes[0].collections[0].get_facecolors())
        assert len(colors) == 3
        assert len(np.unique(colors, axis=0)) == 3
    finally:
        if fig is not None:
            plt.close(fig)


def test_nominal_color_explicit_integer_order_with_multiple_layers():
    fig = None
    try:
        plotter = (
            so.Plot(x=[0, 1, 2], y=[1, 2, 3], color=[1, 2, 3])
            .add(so.Dot())
            .add(so.Dot(), x=[0], y=[1], color=[1])
            .scale(color=so.Nominal(order=[3, 2, 1]))
            .plot()
        )
        fig = plotter._figure
        fig.canvas.draw()
        assert len(fig.axes[0].collections) >= 2
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize("category_values", [["1", "2", "3"], [1.0, 2.0, 3.0]])
def test_nominal_multilayer_text_and_float_categories_unchanged(category_values):
    fig = None
    try:
        plotter = (
            so.Plot(x=[0, 1, 2], y=[1, 2, 3], color=category_values)
            .add(so.Dot())
            .add(so.Dot(), x=[0], y=[1], color=[category_values[0]])
            .scale(color="deep")
            .plot()
        )
        fig = plotter._figure
        fig.canvas.draw()
    finally:
        if fig is not None:
            plt.close(fig)


def test_nominal_color_aggregation_layers_with_integer_subjects():
    """Reproduce the original so.Lines + so.Agg bug with fourteen subjects."""
    subjects = np.repeat(np.arange(1, 15, dtype=int), 12)
    timepoint = np.tile(np.arange(4), 42)
    signals = (
        .2 + subjects * .01 + timepoint * .025
        + (np.arange(len(subjects)) % 3) * .005
    )
    data = pd.DataFrame({
        "timepoint": timepoint,
        "signal": signals,
        "subject": subjects,
    })

    fig = None
    try:
        plotter = (
            so.Plot(data, x="timepoint", y="signal")
            .add(so.Lines(alpha=.7), so.Agg(), color="subject")
            .scale(color="deep")
            .add(so.Line(linewidth=3), so.Agg())
            .plot()
        )
        fig = plotter._figure
        fig.canvas.draw()
        assert len(fig.axes[0].collections) > 0
    finally:
        if fig is not None:
            plt.close(fig)
