
import datetime as pydt

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.scale import LinearScale

import pytest
from pandas.testing import assert_series_equal

from seaborn._compat import scale_factory
from seaborn._core.scales import (
    NumericScale,
    CategoricalScale,
    DateTimeScale,
    get_default_scale,
)


class TestNumeric:

    @pytest.fixture
    def scale(self):
        return LinearScale("x")

    def test_cast_to_float(self, scale):

        x = pd.Series(["1", "2", "3"], name="x")
        s = NumericScale(scale, None)
        assert_series_equal(s.cast(x), x.astype(float))

    def test_convert(self, scale):

        x = pd.Series([1., 2., 3.], name="x")
        s = NumericScale(scale, None).setup(x)
        assert_series_equal(s.convert(x), x)

    def test_normalize_default(self, scale):

        x = pd.Series([1, 2, 3, 4])
        s = NumericScale(scale, None).setup(x)
        assert_series_equal(s.normalize(x), (x - 1) / 3)

    def test_normalize_tuple(self, scale):

        x = pd.Series([1, 2, 3, 4])
        s = NumericScale(scale, (2, 4)).setup(x)
        assert_series_equal(s.normalize(x), (x - 2) / 2)

    def test_normalize_missing(self, scale):

        x = pd.Series([1, 2, np.nan, 5])
        s = NumericScale(scale, None).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([0., .25, np.nan, 1.]))

    def test_normalize_object_uninit(self, scale):

        x = pd.Series([1, 2, 3, 4])
        norm = Normalize()
        s = NumericScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), (x - 1) / 3)
        assert not norm.scaled()

    def test_normalize_object_parinit(self, scale):

        x = pd.Series([1, 2, 3, 4])
        norm = Normalize(2)
        s = NumericScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), (x - 2) / 2)
        assert not norm.scaled()

    def test_normalize_object_fullinit(self, scale):

        x = pd.Series([1, 2, 3, 4])
        norm = Normalize(2, 5)
        s = NumericScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), (x - 2) / 3)
        assert norm.vmax == 5

    def test_normalize_by_full_range(self, scale):

        x = pd.Series([1, 2, 3, 4])
        norm = Normalize()
        s = NumericScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x[:3]), (x[:3] - 1) / 3)
        assert not norm.scaled()

    def test_norm_from_scale(self):

        x = pd.Series([1, 10, 100])
        scale = scale_factory("log", "x")
        s = NumericScale(scale, None).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([0, .5, 1]))

    def test_forward(self):

        x = pd.Series([1., 10., 100.])
        scale = scale_factory("log", "x")
        s = NumericScale(scale, None).setup(x)
        assert_series_equal(s.forward(x), pd.Series([0., 1., 2.]))

    def test_reverse(self):

        x = pd.Series([1., 10., 100.])
        scale = scale_factory("log", "x")
        s = NumericScale(scale, None).setup(x)
        y = pd.Series(np.log10(x))
        assert_series_equal(s.reverse(y), x)

    def test_bad_norm(self, scale):

        norm = "not_a_norm"
        err = "`norm` must be a Normalize object or tuple, not <class 'str'>"
        with pytest.raises(TypeError, match=err):
            scale = NumericScale(scale, norm=norm)


class TestCategorical:

    @pytest.fixture
    def scale(self):
        return LinearScale("x")

    def test_cast_numbers(self, scale):

        x = pd.Series([1, 2, 3])
        s = CategoricalScale(scale, None, format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["1", "2", "3"]))

    def test_cast_formatter(self, scale):

        x = pd.Series([1, 2, 3]) / 3
        s = CategoricalScale(scale, None, "{:.2f}".format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["0.33", "0.67", "1.00"]))

    def test_cast_string(self, scale):

        x = pd.Series(["a", "b", "c"])
        s = CategoricalScale(scale, None, format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["a", "b", "c"]))

    def test_cast_string_with_order(self, scale):

        x = pd.Series(["a", "b", "c"])
        order = ["b", "a", "c"]
        s = CategoricalScale(scale, order, format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["a", "b", "c"]))
        assert s.order == order

    def test_cast_categories(self, scale):

        x = pd.Series(pd.Categorical(["a", "b", "c"], ["b", "a", "c"]))
        s = CategoricalScale(scale, None, format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["a", "b", "c"]))

    def test_cast_drop_categories(self, scale):

        x = pd.Series(["a", "b", "c"])
        order = ["b", "a"]
        s = CategoricalScale(scale, order, format).setup(x)
        assert_series_equal(s.cast(x), pd.Series(["a", "b", np.nan]))

    def test_cast_with_missing(self, scale):

        x = pd.Series(["a", "b", np.nan])
        s = CategoricalScale(scale, None, format).setup(x)
        assert_series_equal(s.cast(x), x)

    def test_convert_strings(self, scale):

        x = pd.Series(["a", "b", "c"])
        s = CategoricalScale(scale, None, format).setup(x)
        y = pd.Series(["b", "a", "c"])
        assert_series_equal(s.convert(y), pd.Series([1., 0., 2.]))

    def test_convert_categories(self, scale):

        x = pd.Series(pd.Categorical(["a", "b", "c"], ["b", "a", "c"]))
        s = CategoricalScale(scale, None, format).setup(x)
        assert_series_equal(s.convert(x), pd.Series([1., 0., 2.]))

    def test_convert_numbers(self, scale):

        x = pd.Series([2, 1, 3])
        s = CategoricalScale(scale, None, format).setup(x)
        y = pd.Series([3, 1, 2])
        assert_series_equal(s.convert(y), pd.Series([2., 0., 1.]))

    def test_convert_ordered_numbers(self, scale):

        x = pd.Series([2, 1, 3])
        order = [3, 2, 1]
        s = CategoricalScale(scale, order, format).setup(x)
        y = pd.Series([3, 1, 2])
        assert_series_equal(s.convert(y), pd.Series([0., 2., 1.]))

    @pytest.mark.xfail(reason="'Nice' formatting for numbers not implemented yet")
    def test_convert_ordered_numbers_mixed_types(self, scale):

        x = pd.Series([2., 1., 3.])
        order = [3, 2, 1]
        s = CategoricalScale(scale, order, format).setup(x)
        assert_series_equal(s.convert(x), pd.Series([1., 2., 0.]))


class TestDateTime:

    @pytest.fixture
    def scale(self):
        return mpl.scale.LinearScale("x")

    def test_cast_strings(self, scale):

        x = pd.Series(["2020-01-01", "2020-03-04", "2020-02-02"])
        s = DateTimeScale(scale).setup(x)
        assert_series_equal(s.cast(x), pd.to_datetime(x))

    def test_cast_numbers(self, scale):

        x = pd.Series([1., 2., 3.])
        s = DateTimeScale(scale).setup(x)
        expected = x.apply(pd.to_datetime, unit="D")
        assert_series_equal(s.cast(x), expected)

    def test_cast_dates(self, scale):

        x = pd.Series(np.array([0, 1, 2], "datetime64[D]"))
        s = DateTimeScale(scale).setup(x)
        assert_series_equal(s.cast(x), x.astype("datetime64[ns]"))

    def test_normalize_default(self, scale):

        x = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        s = DateTimeScale(scale).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([0., .5, 1.]))

    def test_normalize_tuple_of_strings(self, scale):

        x = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        norm = ("2020-01-01", "2020-01-05")
        s = DateTimeScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([0., .25, .5]))

    def test_normalize_tuple_of_dates(self, scale):

        x = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        norm = (
            pydt.datetime.fromisoformat("2020-01-01"),
            pydt.datetime.fromisoformat("2020-01-05"),
        )
        s = DateTimeScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([0., .25, .5]))

    def test_normalize_object(self, scale):

        x = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"])
        norm = mpl.colors.Normalize()
        norm(mpl.dates.datestr2num(x) + 1)
        s = DateTimeScale(scale, norm).setup(x)
        assert_series_equal(s.normalize(x), pd.Series([-.5, 0., .5]))

    def test_forward(self, scale):

        x = pd.Series(["1970-01-04", "1970-01-05", "1970-01-06"])
        s = DateTimeScale(scale).setup(x)
        # Broken prior to matplotlib epoch reset in 3.3
        # expected = pd.Series([3., 4., 5.])
        expected = pd.Series(mpl.dates.datestr2num(x))
        assert_series_equal(s.forward(x), expected)

    def test_reverse(self, scale):

        x = pd.Series(["1970-01-04", "1970-01-05", "1970-01-06"])
        s = DateTimeScale(scale).setup(x)
        y = pd.Series([10., 11., 12.])
        assert_series_equal(s.reverse(y), y)

    def test_convert(self, scale):

        x = pd.Series(["1970-01-04", "1970-01-05", "1970-01-06"])
        s = DateTimeScale(scale).setup(x)
        # Broken prior to matplotlib epoch reset in 3.3
        # expected = pd.Series([3., 4., 5.])
        expected = pd.Series(mpl.dates.datestr2num(x))
        assert_series_equal(s.convert(x), expected)

    def test_convert_with_axis(self, scale):

        x = pd.Series(["1970-01-04", "1970-01-05", "1970-01-06"])
        s = DateTimeScale(scale).setup(x)
        # Broken prior to matplotlib epoch reset in 3.3
        # expected = pd.Series([3., 4., 5.])
        expected = pd.Series(mpl.dates.datestr2num(x))
        ax = mpl.figure.Figure().subplots()
        assert_series_equal(s.convert(x, ax.xaxis), expected)


class TestDefaultScale:

    def test_numeric(self):
        s = pd.Series([1, 2, 3])
        assert isinstance(get_default_scale(s), NumericScale)

    def test_datetime(self):
        s = pd.Series(["2000", "2010", "2020"]).map(pd.to_datetime)
        assert isinstance(get_default_scale(s), DateTimeScale)

    def test_categorical(self):
        s = pd.Series(["1", "2", "3"])
        assert isinstance(get_default_scale(s), CategoricalScale)
