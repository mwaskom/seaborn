import pandas as pd

import pytest

import matplotlib.pyplot as plt

from seaborn._core.plot import Plot


def axis_grid_is_visible(axis):

    return any(line.get_visible() for line in axis.get_gridlines())


def expected_limits_from_ticks(ticks, inverted=False):

    if not len(ticks):
        return None

    start = float(ticks[0]) - 0.5
    end = float(ticks[-1]) + 0.5
    return (end, start) if inverted else (start, end)


def test_nominal_x_axis_limits_and_margin():

    plot = Plot(x=["a", "b", "c"], y=[0, 1, 2]).plot()
    ax, = plot._figure.axes

    x_margin, _ = ax.margins()
    assert x_margin == 0
    expected = expected_limits_from_ticks(ax.get_xticks())
    assert tuple(ax.get_xlim()) == pytest.approx(expected)


def test_nominal_y_axis_limits_and_inversion():

    plot = Plot(x=[0, 1, 2], y=["a", "b", "c"]).plot()
    ax, = plot._figure.axes

    _, y_margin = ax.margins()
    assert y_margin == 0
    assert ax.yaxis_inverted()
    expected = expected_limits_from_ticks(ax.get_yticks(), inverted=True)
    assert tuple(ax.get_ylim()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "plot_kwargs, axis",
    [
        (dict(x=["a", "b", "c"], y=[0, 1, 2]), "x"),
        (dict(x=[0, 1, 2], y=["a", "b", "c"]), "y"),
    ],
)
def test_nominal_axes_suppress_grid_by_default(plot_kwargs, axis):

    plot = Plot(**plot_kwargs).plot()
    ax, = plot._figure.axes
    axis_obj = getattr(ax, f"{axis}axis")

    assert axis_grid_is_visible(axis_obj) is False


@pytest.mark.parametrize(
    "plot_kwargs, axis",
    [
        (dict(x=["a", "b", "c"], y=[0, 1, 2]), "x"),
        (dict(x=[0, 1, 2], y=["a", "b", "c"]), "y"),
    ],
)
def test_nominal_axes_respect_theme_grid(plot_kwargs, axis):

    plot = Plot(**plot_kwargs).theme({"axes.grid": True}).plot()
    ax, = plot._figure.axes
    axis_obj = getattr(ax, f"{axis}axis")

    assert axis_grid_is_visible(axis_obj) is True


def test_nominal_axes_preserve_behavior_in_facets():

    df = pd.DataFrame(
        {
            "category": list("abcabc"),
            "level": list("uvwxyz"),
            "group": ["left"] * 3 + ["right"] * 3,
        }
    )

    plot = Plot(df, x="category", y="level").facet(col="group").plot()

    axes = plot._figure.axes
    assert len(axes) == 2

    for ax in axes:
        expected_x = expected_limits_from_ticks(ax.get_xticks())
        expected_y = expected_limits_from_ticks(ax.get_yticks(), inverted=True)
        assert tuple(ax.get_xlim()) == pytest.approx(expected_x)
        x_margin, y_margin = ax.margins()
        assert x_margin == 0
        assert ax.yaxis_inverted()
        assert tuple(ax.get_ylim()) == pytest.approx(expected_y)
        assert y_margin == 0
        assert axis_grid_is_visible(ax.xaxis) is False
        assert axis_grid_is_visible(ax.yaxis) is False


def test_axes_reuse_clears_nominal_settings():

    fig, ax = plt.subplots()

    Plot(x=["a", "b", "c"], y=["u", "v", "w"]).on(ax).plot()

    ax.cla()

    Plot(x=[0, 1, 2], y=[0, 1, 2]).theme({"axes.grid": True}).on(ax).plot()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert xlim[0] > -0.2
    assert xlim[1] < 2.2
    assert ylim[0] < ylim[1]
    assert not ax.yaxis_inverted()

    assert axis_grid_is_visible(ax.xaxis) is True
    assert axis_grid_is_visible(ax.yaxis) is True

    for axis in ("x", "y"):
        gid = f"_sb_nominal_anchor_{axis}"
        assert all(line.get_gid() != gid for line in ax.lines)

    plt.close(fig)


def test_axes_respect_existing_inversion():

    fig, ax = plt.subplots()

    ax.invert_yaxis()

    Plot(x=[0, 1, 2], y=[0, 1, 2]).on(ax).plot()

    assert ax.yaxis_inverted()

    plt.close(fig)
