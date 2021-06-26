import functools
import numpy as np
import pandas as pd
import matplotlib as mpl

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from seaborn._core.plot import Plot
from seaborn._core.rules import categorical_order
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat

assert_vector_equal = functools.partial(assert_series_equal, check_names=False)


class MockStat(Stat):

    def __call__(self, data):

        return data


class MockMark(Mark):

    # TODO we need to sort out the stat application, it is broken right now
    # default_stat = MockStat
    grouping_vars = ["hue"]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.passed_keys = []
        self.passed_data = []
        self.passed_axes = []
        self.n_splits = 0

    def _plot_split(self, keys, data, ax, mappings, kws):

        self.n_splits += 1
        self.passed_keys.append(keys)
        self.passed_data.append(data)
        self.passed_axes.append(ax)


class TestPlot:

    def test_init_empty(self):

        p = Plot()
        assert p._data._source_data is None
        assert p._data._source_vars == {}

    def test_init_data_only(self, long_df):

        p = Plot(long_df)
        assert p._data._source_data is long_df
        assert p._data._source_vars == {}

    def test_init_df_and_named_variables(self, long_df):

        variables = {"x": "a", "y": "z"}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], long_df[col])
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_df_and_mixed_variables(self, long_df):

        variables = {"x": "a", "y": long_df["z"]}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            if isinstance(col, str):
                assert_vector_equal(p._data.frame[var], long_df[col])
            else:
                assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_vector_variables_only(self, long_df):

        variables = {"x": long_df["a"], "y": long_df["z"]}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_vector_variables_no_index(self, long_df):

        variables = {"x": long_df["a"].to_numpy(), "y": long_df["z"].to_list()}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], pd.Series(col))
            assert p._data.names[var] is None
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_scales(self, long_df):

        p = Plot(long_df, x="x", y="y")
        for var in "xy":
            assert var in p._scales
            assert p._scales[var].type == "unknown"

    def test_add_without_data(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark())
        layer, = p._layers
        assert_frame_equal(p._data.frame, layer.data.frame)

    def test_add_with_new_variable_by_name(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y="y")
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_new_variable_by_vector(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y=long_df["y"])
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_late_data_definition(self, long_df):

        p = Plot().add(MockMark(), data=long_df, x="x", y="y")
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_new_data_definition(self, long_df):

        long_df_sub = long_df.sample(frac=.5)

        p = Plot(long_df, x="x", y="y").add(MockMark(), data=long_df_sub)
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(
                layer.data.frame[var], long_df_sub[var].reindex(long_df.index)
            )

    def test_add_drop_variable(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark(), y=None)
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x"]
        assert "y" not in layer
        assert_vector_equal(layer.data.frame["x"], long_df["x"])

    def test_add_stat_default(self):

        class MarkWithDefaultStat(Mark):
            default_stat = MockStat

        p = Plot().add(MarkWithDefaultStat())
        layer, = p._layers
        assert layer.stat.__class__ is MockStat

    def test_add_stat_nondefault(self):

        class MarkWithDefaultStat(Mark):
            default_stat = MockStat

        class OtherMockStat(MockStat):
            pass

        p = Plot().add(MarkWithDefaultStat(), OtherMockStat())
        layer, = p._layers
        assert layer.stat.__class__ is OtherMockStat

    def test_axis_scale_inference(self, long_df):

        for col, scale_type in zip("zat", ["numeric", "categorical", "datetime"]):
            p = Plot(long_df, x=col, y=col).add(MockMark())
            for var in "xy":
                assert p._scales[var].type == "unknown"
            p._setup_scales()
            for var in "xy":
                assert p._scales[var].type == scale_type

    def test_axis_scale_inference_concatenates(self):

        p = Plot(x=[1, 2, 3]).add(MockMark(), x=["a", "b", "c"])
        p._setup_scales()
        assert p._scales["x"].type == "categorical"

    def test_axis_scale_categorical_explicit_order(self):

        p = Plot(x=["b", "c", "a"]).scale_categorical("x", order=["c", "a", "b"])

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series(["c", "a", "b"])).cat.codes.to_list() == [0, 1, 2]

    def test_axis_scale_numeric_as_categorical(self):

        p = Plot(x=[2, 1, 3]).scale_categorical("x")

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series([1, 2, 3])).cat.codes.to_list() == [0, 1, 2]

    def test_axis_scale_numeric_as_categorical_explicit_order(self):

        p = Plot(x=[1, 2, 3]).scale_categorical("x", order=[2, 1, 3])

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series([2, 1, 3])).cat.codes.to_list() == [0, 1, 2]

    def test_axis_scale_numeric_as_datetime(self):

        p = Plot(x=[1, 2, 3]).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"

        numbers = [2, 1, 3]
        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        assert_series_equal(
            scl.cast(pd.Series(numbers)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    @pytest.mark.xfail
    def test_axis_scale_categorical_as_numeric(self):

        # TODO marked as expected fail because we have not implemented this yet
        # see notes in ScaleWrapper.cast

        strings = ["2", "1", "3"]
        p = Plot(x=strings).scale_numeric("x")
        scl = p._scales["x"]
        assert scl.type == "numeric"
        assert_series_equal(
            scl.cast(pd.Series(strings)),
            pd.Series(strings).astype(float)
        )

    def test_axis_scale_categorical_as_datetime(self):

        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        p = Plot(x=dates).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"
        assert_series_equal(
            scl.cast(pd.Series(dates, dtype=object)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    def test_axis_scale_mark_data_log_transform(self, long_df):

        col = "z"
        m = MockMark()
        Plot(long_df, x=col).scale_numeric("x", "log").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df[col])

    def test_axis_scale_mark_data_log_transfrom_with_stat(self, long_df):

        class Mean(Stat):
            def __call__(self, data):
                return data.mean()

        col = "z"
        grouper = "a"
        m = MockMark()
        s = Mean()

        Plot(long_df, x=grouper, y=col).scale_numeric("y", "log").add(m, s).plot()

        expected = (
            long_df[col]
            .pipe(np.log)
            .groupby(long_df[grouper], sort=False)
            .mean()
            .pipe(np.exp)
            .reset_index(drop=True)
        )
        assert_vector_equal(m.passed_data[0]["y"], expected)

    def test_axis_scale_mark_data_from_categorical(self, long_df):

        col = "a"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        levels = categorical_order(long_df[col])
        level_map = {x: float(i) for i, x in enumerate(levels)}
        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(level_map))

    def test_axis_scale_mark_data_from_datetime(self, long_df):

        col = "t"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(mpl.dates.date2num))

    def test_figure_setup_creates_matplotlib_objects(self):

        p = Plot()
        p._setup_figure()
        assert isinstance(p._figure, mpl.figure.Figure)
        assert isinstance(p._ax, mpl.axes.Axes)

    @pytest.mark.parametrize(
        "arg,expected",
        [("x", "x"), ("y", "y"), ("v", "x"), ("h", "y")],
    )
    def test_orient(self, arg, expected):

        class MockMarkTrackOrient(MockMark):
            def _adjust(self, data):
                self.orient_at_adjust = self.orient
                return data

        class MockStatTrackOrient(MockStat):
            def setup(self, data):
                super().setup(data)
                self.orient_at_setup = self.orient
                return self

        m = MockMarkTrackOrient()
        s = MockStatTrackOrient()
        Plot(x=[1, 2, 3], y=[1, 2, 3]).add(m, s, orient=arg).plot()

        assert m.orient == expected
        assert m.orient_at_adjust == expected
        assert s.orient == expected
        assert s.orient_at_setup == expected

    def test_empty_plot(self):

        m = MockMark()
        Plot().plot()
        assert m.n_splits == 0

    def test_plot_split_single(self, long_df):

        m = MockMark()
        p = Plot(long_df, x="f", y="z").add(m).plot()
        assert m.n_splits == 1

        assert m.passed_keys[0] == {}
        assert m.passed_axes[0] is p._ax
        assert_frame_equal(m.passed_data[0], p._data.frame)

    def check_splits_single_var(self, plot, mark, split_var, split_keys):

        assert mark.n_splits == len(split_keys)
        assert mark.passed_keys == [{split_var: key} for key in split_keys]

        full_data = plot._data.frame
        for i, key in enumerate(split_keys):

            split_data = full_data[full_data[split_var] == key]
            assert_frame_equal(mark.passed_data[i], split_data)

    @pytest.mark.parametrize(
        "split_var", [
            "hue",  # explicitly declared on the Mark
            "group",  # implicitly used for all Mark classes
        ])
    def test_plot_split_one_grouping_variable(self, long_df, split_var):

        split_col = "a"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        assert m.passed_axes == [p._ax for _ in split_keys]
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_plot_split_across_facets_no_subgroups(self, long_df):

        split_var = "col"
        split_col = "b"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        assert m.passed_axes == list(p._figure.axes)
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_plot_adjustments(self, long_df):

        orig_df = long_df.copy(deep=True)

        class AdjustableMockMark(MockMark):
            def _adjust(self, data):
                data["x"] = data["x"] + 1
                return data

        m = AdjustableMockMark()
        Plot(long_df, x="z", y="z").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] + 1)
        assert_vector_equal(m.passed_data[0]["y"], long_df["z"])

        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    def test_plot_adjustments_log_scale(self, long_df):

        class AdjustableMockMark(MockMark):
            def _adjust(self, data):
                data["x"] = data["x"] - 1
                return data

        m = AdjustableMockMark()
        Plot(long_df, x="z", y="z").scale_numeric("x", "log").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] / 10)

    # TODO Current untested includes:
    # - anything having to do with semantic mapping
    # - much having to do with faceting
    # - interaction with existing matplotlib objects
    # - any important corner cases in the original test_core suite
