import itertools

import numpy as np
import pytest

from seaborn._core.subplots import Subplots


class TestSpecificationChecks:

    def test_both_facets_and_wrap(self):

        err = "Cannot wrap facets when specifying both `col` and `row`."
        facet_spec = {"wrap": 3, "variables": {"col": "a", "row": "b"}}
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, facet_spec, {})

    def test_cross_xy_pairing_and_wrap(self):

        err = "Cannot wrap subplots when pairing on both `x` and `y`."
        pair_spec = {"wrap": 3, "structure": {"x": ["a", "b"], "y": ["y", "z"]}}
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, pair_spec)

    def test_col_facets_and_x_pairing(self):

        err = "Cannot facet the columns while pairing on `x`."
        facet_spec = {"variables": {"col": "a"}}
        pair_spec = {"structure": {"x": ["x", "y"]}}
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, facet_spec, pair_spec)

    def test_wrapped_columns_and_y_pairing(self):

        err = "Cannot wrap the columns while pairing on `y`."
        facet_spec = {"variables": {"col": "a"}, "wrap": 2}
        pair_spec = {"structure": {"y": ["x", "y"]}}
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, facet_spec, pair_spec)

    def test_wrapped_x_pairing_and_facetd_rows(self):

        err = "Cannot wrap the columns while faceting the rows."
        facet_spec = {"variables": {"row": "a"}}
        pair_spec = {"structure": {"x": ["x", "y"]}, "wrap": 2}
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, facet_spec, pair_spec)


class TestSubplotSpec:

    def test_single_subplot(self):

        s = Subplots({}, {}, {})

        assert s.n_subplots == 1
        assert s.subplot_spec["ncols"] == 1
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_single_facet(self):

        key = "a"
        order = list("abc")
        spec = {"variables": {"col": key}, "structure": {"col": order}}
        s = Subplots({}, spec, {})

        assert s.n_subplots == len(order)
        assert s.subplot_spec["ncols"] == len(order)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_two_facets(self):

        col_key = "a"
        row_key = "b"
        col_order = list("xy")
        row_order = list("xyz")
        spec = {
            "variables": {"col": col_key, "row": row_key},
            "structure": {"col": col_order, "row": row_order},

        }
        s = Subplots({}, spec, {})

        assert s.n_subplots == len(col_order) * len(row_order)
        assert s.subplot_spec["ncols"] == len(col_order)
        assert s.subplot_spec["nrows"] == len(row_order)
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped(self):

        key = "b"
        wrap = 3
        order = list("abcde")
        spec = {"variables": {"col": key}, "structure": {"col": order}, "wrap": wrap}
        s = Subplots({}, spec, {})

        assert s.n_subplots == len(order)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(order) // wrap + 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_row_facet_wrapped(self):

        key = "b"
        wrap = 3
        order = list("abcde")
        spec = {"variables": {"row": key}, "structure": {"row": order}, "wrap": wrap}
        s = Subplots({}, spec, {})

        assert s.n_subplots == len(order)
        assert s.subplot_spec["ncols"] == len(order) // wrap + 1
        assert s.subplot_spec["nrows"] == wrap
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_col_facet_wrapped_single_row(self):

        key = "b"
        order = list("abc")
        wrap = len(order) + 2
        spec = {"variables": {"col": key}, "structure": {"col": order}, "wrap": wrap}
        s = Subplots({}, spec, {})

        assert s.n_subplots == len(order)
        assert s.subplot_spec["ncols"] == len(order)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is True

    def test_x_and_y_paired(self):

        x = ["x", "y", "z"]
        y = ["a", "b"]
        s = Subplots({}, {}, {"structure": {"x": x, "y": y}})

        assert s.n_subplots == len(x) * len(y)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] == "row"

    def test_x_paired(self):

        x = ["x", "y", "z"]
        s = Subplots({}, {}, {"structure": {"x": x}})

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] is True

    def test_y_paired(self):

        y = ["x", "y", "z"]
        s = Subplots({}, {}, {"structure": {"y": y}})

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == 1
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] == "row"

    def test_x_paired_and_wrapped(self):

        x = ["a", "b", "x", "y", "z"]
        wrap = 3
        s = Subplots({}, {}, {"structure": {"x": x}, "wrap": wrap})

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is True

    def test_y_paired_and_wrapped(self):

        y = ["a", "b", "x", "y", "z"]
        wrap = 2
        s = Subplots({}, {}, {"structure": {"y": y}, "wrap": wrap})

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == len(y) // wrap + 1
        assert s.subplot_spec["nrows"] == wrap
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is False

    def test_y_paired_and_wrapped_single_row(self):

        y = ["x", "y", "z"]
        wrap = 1
        s = Subplots({}, {}, {"structure": {"y": y}, "wrap": wrap})

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == len(y)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] is False

    def test_col_faceted_y_paired(self):

        y = ["x", "y", "z"]
        key = "a"
        order = list("abc")
        facet_spec = {"variables": {"col": key}, "structure": {"col": order}}
        pair_spec = {"structure": {"y": y}}
        s = Subplots({}, facet_spec, pair_spec)

        assert s.n_subplots == len(order) * len(y)
        assert s.subplot_spec["ncols"] == len(order)
        assert s.subplot_spec["nrows"] == len(y)
        assert s.subplot_spec["sharex"] is True
        assert s.subplot_spec["sharey"] == "row"

    def test_row_faceted_x_paired(self):

        x = ["f", "s"]
        key = "a"
        order = list("abc")
        facet_spec = {"variables": {"row": key}, "structure": {"row": order}}
        pair_spec = {"structure": {"x": x}}
        s = Subplots({}, facet_spec, pair_spec)

        assert s.n_subplots == len(order) * len(x)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == len(order)
        assert s.subplot_spec["sharex"] == "col"
        assert s.subplot_spec["sharey"] is True

    def test_x_any_y_paired_non_cross(self):

        x = ["a", "b", "c"]
        y = ["x", "y", "z"]
        spec = {"structure": {"x": x, "y": y}, "cross": False}
        s = Subplots({}, {}, spec)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == len(y)
        assert s.subplot_spec["nrows"] == 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is False

    def test_x_any_y_paired_non_cross_wrapped(self):

        x = ["a", "b", "c"]
        y = ["x", "y", "z"]
        wrap = 2
        spec = {"structure": {"x": x, "y": y}, "cross": False, "wrap": wrap}
        s = Subplots({}, {}, spec)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] is False

    def test_forced_unshared_facets(self):

        s = Subplots({"sharex": False, "sharey": "row"}, {}, {})
        assert s.subplot_spec["sharex"] is False
        assert s.subplot_spec["sharey"] == "row"


class TestSubplotElements:

    def test_single_subplot(self):

        s = Subplots({}, {}, {})
        f = s.init_figure({}, {})

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
    def test_single_facet_dim(self, dim):

        key = "a"
        order = list("abc")
        spec = {"variables": {dim: key}, "structure": {dim: order}}
        s = Subplots({}, spec, {})
        s.init_figure(spec, {})

        assert len(s) == len(order)

        for i, e in enumerate(s):
            assert e[dim] == order[i]
            for axis in "xy":
                assert e[axis] == axis
            assert e["top"] == (dim == "col" or i == 0)
            assert e["bottom"] == (dim == "col" or i == len(order) - 1)
            assert e["left"] == (dim == "row" or i == 0)
            assert e["right"] == (dim == "row" or i == len(order) - 1)

    @pytest.mark.parametrize("dim", ["col", "row"])
    def test_single_facet_dim_wrapped(self, dim):

        key = "b"
        order = list("abc")
        wrap = len(order) - 1
        spec = {"variables": {dim: key}, "structure": {dim: order}, "wrap": wrap}
        s = Subplots({}, spec, {})
        s.init_figure(spec, {})

        assert len(s) == len(order)

        for i, e in enumerate(s):
            assert e[dim] == order[i]
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

    def test_both_facet_dims(self):

        col = "a"
        row = "b"
        col_order = list("ab")
        row_order = list("xyz")
        facet_spec = {
            "variables": {"col": col, "row": row},
            "structure": {"col": col_order, "row": row_order},
        }
        s = Subplots({}, facet_spec, {})
        s.init_figure(facet_spec, {})

        n_cols = len(col_order)
        n_rows = len(row_order)
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

        for e, (row_, col_) in zip(es, itertools.product(row_order, col_order)):
            assert e["col"] == col_
            assert e["row"] == row_

        for e in es:
            assert e["x"] == "x"
            assert e["y"] == "y"

    @pytest.mark.parametrize("var", ["x", "y"])
    def test_single_paired_var(self, var):

        other_var = {"x": "y", "y": "x"}[var]
        pairings = ["x", "y", "z"]
        pair_spec = {
            "variables": {f"{var}{i}": v for i, v in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i, _ in enumerate(pairings)]},
        }

        s = Subplots({}, {}, pair_spec)
        s.init_figure(pair_spec)

        assert len(s) == len(pair_spec["structure"][var])

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
    def test_single_paired_var_wrapped(self, var):

        other_var = {"x": "y", "y": "x"}[var]
        pairings = ["x", "y", "z", "a", "b"]
        wrap = len(pairings) - 2
        pair_spec = {
            "variables": {f"{var}{i}": val for i, val in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i, _ in enumerate(pairings)]},
            "wrap": wrap
        }
        s = Subplots({}, {}, pair_spec)
        s.init_figure(pair_spec)

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

    def test_both_paired_variables(self):

        x = ["x0", "x1"]
        y = ["y0", "y1", "y2"]
        pair_spec = {"structure": {"x": x, "y": y}}
        s = Subplots({}, {}, pair_spec)
        s.init_figure(pair_spec)

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

    def test_both_paired_non_cross(self):

        pair_spec = {
            "structure": {"x": ["x0", "x1", "x2"], "y": ["y0", "y1", "y2"]},
            "cross": False
        }
        s = Subplots({}, {}, pair_spec)
        s.init_figure(pair_spec)

        for i, e in enumerate(s):
            assert e["x"] == f"x{i}"
            assert e["y"] == f"y{i}"
            assert e["col"] is e["row"] is None
            assert e["left"] == (i == 0)
            assert e["right"] == (i == (len(s) - 1))
            assert e["top"]
            assert e["bottom"]

    @pytest.mark.parametrize("dim,var", [("col", "y"), ("row", "x")])
    def test_one_facet_one_paired(self, dim, var):

        other_var = {"x": "y", "y": "x"}[var]
        other_dim = {"col": "row", "row": "col"}[dim]
        order = list("abc")
        facet_spec = {"variables": {dim: "s"}, "structure": {dim: order}}

        pairings = ["x", "y", "t"]
        pair_spec = {
            "variables": {f"{var}{i}": val for i, val in enumerate(pairings)},
            "structure": {var: [f"{var}{i}" for i, _ in enumerate(pairings)]},
        }

        s = Subplots({}, facet_spec, pair_spec)
        s.init_figure(pair_spec)

        n_cols = len(order) if dim == "col" else len(pairings)
        n_rows = len(order) if dim == "row" else len(pairings)

        assert len(s) == len(order) * len(pairings)

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
            assert e[dim] == order[i % len(pairings)]
            assert e[other_dim] is None
            assert e[var] == f"{var}{i // len(order)}"
            assert e[other_var] == other_var
