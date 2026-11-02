"""Regression tests for sparse categorical histogram stacks (GH #3232)."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import seaborn as sns


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("common_bins", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_categorical_stack_has_finite_nonoverlapping_segments(
    orient, common_bins, weighted,
):
    data = pd.DataFrame({
        "category": ["A", "A", "A", "B", "B"],
        "group": ["a", "b", "c", "d", "a"],
    })
    if weighted:
        data["weight"] = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots()
    try:
        sns.histplot(
            data=data,
            hue="group",
            weights="weight" if weighted else None,
            multiple="stack",
            common_bins=common_bins,
            discrete=True,
            ax=ax,
            **{orient: "category"},
        )
        position = (lambda p: p.get_x()) if orient == "x" else (
            lambda p: p.get_y()
        )
        height = (lambda p: p.get_height()) if orient == "x" else (
            lambda p: p.get_width()
        )
        baseline = (lambda p: p.get_y()) if orient == "x" else (
            lambda p: p.get_x()
        )

        assert ax.patches
        for patch in ax.patches:
            assert math.isfinite(position(patch))
            assert math.isfinite(baseline(patch))
            assert math.isfinite(height(patch))
            assert height(patch) >= 0

        totals = (6, 9) if weighted else (3, 2)
        for location, expected in zip((-0.5, 0.5), totals):
            segments = [
                patch for patch in ax.patches
                if math.isclose(position(patch), location, abs_tol=1e-10)
                and height(patch) > 1e-10
            ]
            segments.sort(key=baseline)
            assert math.isclose(sum(map(height, segments)), expected)
            cursor = 0.0
            for patch in segments:
                assert math.isclose(baseline(patch), cursor, abs_tol=1e-10)
                cursor += height(patch)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("orient", ["x", "y"])
@pytest.mark.parametrize("common_bins", [True, False])
@pytest.mark.parametrize("share_categorical_axis", [True, False])
def test_faceted_stack_categorical_bins_are_finite(
    orient, common_bins, share_categorical_axis,
):
    data = pd.DataFrame({
        "category": ["C"] * 6 + ["D"] * 5 + ["E"] * 4 + ["F"] * 3,
        "group": [
            "hx", "ax", "bx", "cx", "dx", "ex", "fx", "gx", "hx",
            "ax", "bx", "cx", "dx", "ex", "fx", "gx", "hx", "ax",
        ],
        "facet": [1] * 11 + [2] * 7,
    })
    grid = sns.displot(
        data=data, hue="group", col="facet",
        multiple="stack", common_bins=common_bins,
        facet_kws={
            "sharex" if orient == "x" else "sharey": share_categorical_axis,
        },
        **{orient: "category"},
    )
    try:
        patches = [p for ax in grid.axes.flat for p in ax.patches]
        assert patches
        for p in patches:
            assert math.isfinite(p.get_x())
            assert math.isfinite(p.get_y())
            assert math.isfinite(p.get_width())
            assert math.isfinite(p.get_height())
    finally:
        plt.close(grid.figure)

@pytest.mark.parametrize(
    "stat", ["count", "frequency", "density", "probability", "percent"]
)
@pytest.mark.parametrize("common_norm", [False, True])
@pytest.mark.parametrize("orient", ["x", "y"])
def test_categorical_stack_statistics_remain_finite(stat, common_norm, orient):
    """Normalized histogram statistics must not acquire missing geometry."""
    data = pd.DataFrame({
        "category": ["B", "A", "A", "B", "C", "B", "A", "C"],
        "group": pd.Categorical(
            ["high", "low", "high", "low", "high", "high", "low", "low"],
            categories=["low", "high", "unused"],
            ordered=True,
        ),
    })
    fig, ax = plt.subplots()
    try:
        sns.histplot(
            data=data,
            hue="group",
            hue_order=["high", "low", "unused"],
            multiple="stack",
            common_bins=False,
            common_norm=common_norm,
            stat=stat,
            discrete=True,
            ax=ax,
            **{orient: "category"},
        )
        assert ax.patches
        for patch in ax.patches:
            assert all(math.isfinite(value) for value in (
                patch.get_x(),
                patch.get_y(),
                patch.get_width(),
                patch.get_height(),
            ))
            assert patch.get_width() >= 0
            assert patch.get_height() >= 0
    finally:
        plt.close(fig)
