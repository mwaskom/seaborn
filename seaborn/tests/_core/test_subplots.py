import itertools
import numpy as np
import pytest

from seaborn._core.data import PlotData
from seaborn._core.subplots import Subplots
from seaborn._core.rules import categorical_order


class TestSpecificationChecks:

    def test_both_facets_and_wrap(self, long_df):

        data = PlotData(long_df, dict(col="a", row="b"))
        err = "Cannot wrap facets when specifying both `col` and `row`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {"wrap": 3}, {}, data)

    def test_cartesian_xy_pairing_and_wrap(self, long_df):

        data = PlotData(long_df, {})
        err = "Cannot wrap subplots when pairing on both `x` and `y`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y"], "y": ["a", "b"], "wrap": 3}, data)

    def test_col_facets_and_x_pairing(self, long_df):

        data = PlotData(long_df, {"col": "a"})
        err = "Cannot facet the columns while pairing on `x`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y"]}, data)

    def test_wrapped_columns_and_y_pairing(self, long_df):

        data = PlotData(long_df, {"col": "a"})
        err = "Cannot wrap the columns while pairing on `y`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {"wrap": 2}, {"y": ["x", "y"]}, data)

    def test_wrapped_x_pairing_and_facetd_rows(self, long_df):

        data = PlotData(long_df, {"row": "a"})
        err = "Cannot wrap the columns while faceting the rows."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y", "z"], "wrap": 2}, data)


class TestSubplotSpec:

    def test_single_subplot(self, long_df):

        data = PlotData(long_df, {"x": "x", "y": "y"})
        s = Subplots({}, {}, {}, data)

        assert s.n_subplots == 1
        assert s.subplot_spec["ncols"] == 1
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_single_facet(self, long_df):

        key = "a"
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_two_facets(self, long_df):

        col_key = "a"
        row_key = "b"
        data = PlotData(long_df, {"col": col_key, "row": row_key})
        s = Subplots({}, {}, {}, data)

        n_cols = len(categorical_order(long_df[col_key]))
        n_rows = len(categorical_order(long_df[row_key]))
        assert s.n_subplots == n_cols * n_rows
        assert s.subplot_spec["ncols"] == n_cols
        assert s.subplot_spec["nrows"] == n_rows
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped(self, long_df):

        key = "b"
        wrap = 3
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == n_levels // wrap + 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_row_facet_wrapped(self, long_df):

        key = "b"
        wrap = 3
        data = PlotData(long_df, {"row": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels // wrap + 1
        assert s.subplot_spec["nrows"] == wrap
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped_single_row(self, long_df):

        key = "b"
        n_levels = len(categorical_order(long_df[key]))
        wrap = n_levels + 2
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_x_and_y_paired(self, long_df):

        x = ["x", "y", "z"]
        y = ["a", "b"]
        data = PlotData({}, {})
        s = Subplots({}, {}, {"x": x, "y": y}, data)

        assert s.n_subplots == len(x) * len(y)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] == "row"

    def test_x_paired(self, long_df):

        x = ["x", "y", "z"]
        data = PlotData(long_df, {"y": "a"})
        s = Subplots({}, {}, {"x": x}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] is True

    def test_y_paired(self, long_df):

        y = ["x", "y", "z"]
        data = PlotData(long_df, {"x": "a"})
        s = Subplots({}, {}, {"y": y}, data)

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == 1
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] == "row"

    def test_x_paired_and_wrapped(self, long_df):

        x = ["a", "b", "x", "y", "z"]
        wrap = 3
        data = PlotData(long_df, {"y": "t"})
        s = Subplots({}, {}, {"x": x, "wrap": wrap}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is True

    def test_y_paired_and_wrapped(self, long_df):

        y = ["a", "b", "x", "y", "z"]
        wrap = 2
        data = PlotData(long_df, {"x": "a"})
        s = Subplots({}, {}, {"y": y, "wrap": wrap}, data)

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == len(y) // wrap + 1
        assert s.subplot_spec["nrows"] == wrap
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is False

    def test_col_faceted_y_paired(self, long_df):

        y = ["x", "y", "z"]
        key = "a"
        data = PlotData(long_df, {"x": "f", "col": key})
        s = Subplots({}, {}, {"y": y}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels * len(y)
        assert s.subplot_spec["ncols"] == n_levels
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] == "row"

    def test_row_faceted_x_paired(self, long_df):

        x = ["f", "s"]
        key = "a"
        data = PlotData(long_df, {"y": "z", "row": key})
        s = Subplots({}, {}, {"x": x}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels * len(x)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == n_levels
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] is True

    def test_x_any_y_paired_non_cartesian(self, long_df):

        x = ["a", "b", "c"]
        y = ["x", "y", "z"]

        data = PlotData(long_df, {})
        s = Subplots({}, {}, {"x": x, "y": y, "cartesian": False}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == len(y)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is False

    def test_x_any_y_paired_non_cartesian_wrapped(self, long_df):

        x = ["a", "b", "c"]
        y = ["x", "y", "z"]
        wrap = 2

        data = PlotData(long_df, {})
        s = Subplots({}, {}, {"x": x, "y": y, "cartesian": False, "wrap": wrap}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is False

    def test_forced_unshared_facets(self, long_df):

        data = PlotData(long_df, {"col": "a", "row": "f"})
        s = Subplots({"sharex": False, "sharey": "row"}, {}, {}, data)
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] == "row"


class TestSubplotElements:

    def test_single_subplot(self, long_df):

        data = PlotData(long_df, {"x": "x", "y": "y"})
        s = Subplots({}, {}, {}, data)
        f = s.init_figure(False)

        assert len(s) == 1
        for i, e in enumerate(s):
            for side in ["left", "right", "bottom", "top"]:
                assert e[side]
            for dim in ["col", "row"]:
                assert e[dim] is None
            for axis in "xy":
                assert e[axis] == axis
            assert e["ax"] == f.axes[i]

    @pytest.mark.parametrize("dim", ["col", "row"])
    def test_single_facet_dim(self, long_df, dim):

        key = "a"
        data = PlotData(long_df, {"x": "x", "y": "y", dim: key})
        s = Subplots({}, {}, {}, data)
        s.init_figure(False)

        levels = categorical_order(long_df[key])
        assert len(s) == len(levels)

        for i, e in enumerate(s):
            assert e[dim] == levels[i]
            for axis in "xy":
                assert e[axis] == axis
            assert e["top"] == (dim == "col" or i == 0)
            assert e["bottom"] == (dim == "col" or i == len(levels) - 1)
            assert e["left"] == (dim == "row" or i == 0)
            assert e["right"] == (dim == "row" or i == len(levels) - 1)

    @pytest.mark.parametrize("dim", ["col", "row"])
    def test_single_facet_dim_wrapped(self, long_df, dim):

        key = "b"
        levels = categorical_order(long_df[key])
        wrap = len(levels) - 1
        data = PlotData(long_df, {"x": "x", "y": "y", dim: key})
        s = Subplots({}, {"wrap": wrap}, {}, data)
        s.init_figure(False)

        assert len(s) == len(levels)

        for i, e in enumerate(s):
            assert e[dim] == levels[i]
            for axis in "xy":
                assert e[axis] == axis

            sides = {
                "col": ["top", "bottom", "left", "right"],
                "row": ["left", "right", "top", "bottom"],
            }
            tests = (
                i < wrap,
                i >= wrap or i >= len(s) % wrap,
                i % wrap == 0,
                i % wrap == wrap - 1 or i + 1 == len(s),
            )

            for side, expected in zip(sides[dim], tests):
                assert e[side] == expected

    def test_both_facet_dims(self, long_df):

        x = "f"
        y = "z"
        col = "a"
        row = "b"
        data = PlotData(long_df, {"x": x, "y": y, "col": col, "row": row})
        s = Subplots({}, {}, {}, data)
        s.init_figure(False)

        col_levels = categorical_order(long_df[col])
        row_levels = categorical_order(long_df[row])
        n_cols = len(col_levels)
        n_rows = len(row_levels)
        assert len(s) == n_cols * n_rows
        es = list(s)

        for e in es[:n_cols]:
            assert e["top"]
        for e in es[::n_cols]:
            assert e["left"]
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        for e in es[-n_cols:]:
            assert e["bottom"]

        for e, (row_, col_) in zip(es, itertools.product(row_levels, col_levels)):
            assert e["col"] == col_
            assert e["row"] == row_

        for e in es:
            assert e["x"] == "x"
            assert e["y"] == "y"

    @pytest.mark.parametrize("var", ["x", "y"])
    def test_single_paired_var(self, long_df, var):

        other_var = {"x": "y", "y": "x"}[var]
        variables = {other_var: "a"}
        pair_spec = {var: ["x", "y", "z"]}

        data = PlotData(long_df, variables)
        s = Subplots({}, {}, pair_spec, data)
        s.init_figure(False)

        assert len(s) == len(pair_spec[var])

        for i, e in enumerate(s):
            assert e[var] == f"{var}{i}"
            assert e[other_var] == other_var
            assert e["col"] is e["row"] is None

        tests = i == 0, True, True, i == len(s) - 1
        sides = {
            "x": ["left", "right", "top", "bottom"],
            "y": ["top", "bottom", "left", "right"],
        }

        for side, expected in zip(sides[var], tests):
            assert e[side] == expected

    @pytest.mark.parametrize("var", ["x", "y"])
    def test_single_paired_var_wrapped(self, long_df, var):

        other_var = {"x": "y", "y": "x"}[var]
        variables = {other_var: "a"}
        pairings = ["x", "y", "z", "a", "b"]
        wrap = len(pairings) - 2
        pair_spec = {var: pairings, "wrap": wrap}
        data = PlotData(long_df, variables)
        s = Subplots({}, {}, pair_spec, data)
        s.init_figure(False)

        assert len(s) == len(pairings)

        for i, e in enumerate(s):
            assert e[var] == f"{var}{i}"
            assert e[other_var] == other_var
            assert e["col"] is e["row"] is None

            tests = (
                i < wrap,
                i >= wrap or i >= len(s) % wrap,
                i % wrap == 0,
                i % wrap == wrap - 1 or i + 1 == len(s),
            )
            sides = {
                "x": ["top", "bottom", "left", "right"],
                "y": ["left", "right", "top", "bottom"],
            }
            for side, expected in zip(sides[var], tests):
                assert e[side] == expected

    def test_both_paired_variables(self, long_df):

        x = ["a", "b"]
        y = ["x", "y", "z"]
        data = PlotData(long_df, {})
        s = Subplots({}, {}, {"x": x, "y": y}, data)
        s.init_figure(False)

        n_cols = len(x)
        n_rows = len(y)
        assert len(s) == n_cols * n_rows
        es = list(s)

        for e in es[:n_cols]:
            assert e["top"]
        for e in es[::n_cols]:
            assert e["left"]
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        for e in es[-n_cols:]:
            assert e["bottom"]

        for e in es:
            assert e["col"] is e["row"] is None

        for i in range(len(y)):
            for j in range(len(x)):
                e = es[i * len(x) + j]
                assert e["x"] == f"x{j}"
                assert e["y"] == f"y{i}"

    def test_both_paired_non_cartesian(self, long_df):

        pair_spec = {"x": ["a", "b", "c"], "y": ["x", "y", "z"], "cartesian": False}
        data = PlotData(long_df, {})
        s = Subplots({}, {}, pair_spec, data)
        s.init_figure(False)

        for i, e in enumerate(s):
            assert e["x"] == f"x{i}"
            assert e["y"] == f"y{i}"
            assert e["col"] is e["row"] is None
            assert e["left"] == (i == 0)
            assert e["right"] == (i == (len(s) - 1))
            assert e["top"]
            assert e["bottom"]

    @pytest.mark.parametrize("dim,var", [("col", "y"), ("row", "x")])
    def test_one_facet_one_paired(self, long_df, dim, var):

        other_var = {"x": "y", "y": "x"}[var]
        other_dim = {"col": "row", "row": "col"}[dim]

        variables = {other_var: "z", dim: "s"}
        pairings = ["x", "y", "t"]
        pair_spec = {var: pairings}

        data = PlotData(long_df, variables)
        s = Subplots({}, {}, pair_spec, data)
        s.init_figure(False)

        levels = categorical_order(long_df[variables[dim]])
        n_cols = len(levels) if dim == "col" else len(pairings)
        n_rows = len(levels) if dim == "row" else len(pairings)

        assert len(s) == len(levels) * len(pairings)

        es = list(s)

        for e in es[:n_cols]:
            assert e["top"]
        for e in es[::n_cols]:
            assert e["left"]
        for e in es[n_cols - 1::n_cols]:
            assert e["right"]
        for e in es[-n_cols:]:
            assert e["bottom"]

        if dim == "row":
            es = np.reshape(es, (n_rows, n_cols)).T.ravel()

        for i, e in enumerate(es):
            assert e[dim] == levels[i % len(pairings)]
            assert e[other_dim] is None
            assert e[var] == f"{var}{i // len(levels)}"
            assert e[other_var] == other_var
