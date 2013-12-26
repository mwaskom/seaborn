import numpy as np
import pandas as pd
import nose.tools as nt
import numpy.testing as npt

from .. import distributions as dist


class TestBoxReshaping(object):
    """Tests for function that preps boxplot/violinplot data."""
    n_total = 60
    rs = np.random.RandomState(0)
    x = rs.randn(n_total / 3, 3)
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    y = pd.Series(rs.randn(n_total), name="y_data")
    g = pd.Series(rs.choice(list("abc"), n_total), name="small")
    df = pd.DataFrame(dict(y=y, g=g))

    def test_1d_values(self):
        """Test boxplot prep for 1D data in various forms."""
        x_1d_array = self.x.ravel()
        vals_1d_array = dist._box_reshape(x_1d_array, None, None, None)[0]
        nt.assert_equal(len(vals_1d_array), 1)
        nt.assert_equal(len(vals_1d_array[0]), self.n_total)

        x_1d_list = x_1d_array.tolist()
        vals_1d_list = dist._box_reshape(x_1d_list, None, None, None)[0]
        nt.assert_equal(len(vals_1d_list), 1)
        nt.assert_equal(len(vals_1d_list[0]), self.n_total)

    def test_list_of_array_values(self):
        """Test boxplot prep for 2D data that is a list of arrays."""
        x_list = self.x.T.tolist()
        vals = dist._box_reshape(x_list, None, None, None)[0]
        nt.assert_equal(len(vals), 3)
        lengths = [len(v_i) for v_i in vals]
        nt.assert_equal(lengths, [self.n_total / 3] * 3)

    def test_array_values(self):
        """Test boxplot prep for a basic array input."""
        vals = dist._box_reshape(self.x, None, None, None)[0]
        nt.assert_equal(np.shape(vals), (3, self.n_total / 3))
        npt.assert_array_equal(vals, self.x.T)

    def test_dataframe_values(self):
        """Test boxplot prep for a DataFrame input."""
        vals = dist._box_reshape(self.x_df, None, None, None)[0]
        nt.assert_equal(np.shape(vals), (3, self.n_total / 3))
        npt.assert_array_equal(vals, self.x.T)

    def test_series_groupby(self):
        """Test boxplot groupby using a series of data labels."""
        vals = dist._box_reshape(self.df.y, self.df.g, None, None)[0]
        nt.assert_equal(len(vals), 3)
        want_lengths = pd.value_counts(self.df.g)[["a", "b", "c"]]
        got_lengths = [len(a) for a in vals]
        npt.assert_array_equal(want_lengths, got_lengths)

    def test_series_groupby_order(self):
        """Test a series-based groupby with a forced ordering."""
        order = ["c", "a", "b"]
        vals = dist._box_reshape(self.df.y, self.df.g, None, order)[0]
        want_lengths = pd.value_counts(self.df.g)[order]
        got_lengths = [len(a) for a in vals]
        npt.assert_array_equal(want_lengths, got_lengths)

    def test_function_groupby(self):
        """Test boxplot groupby using a grouping function."""
        grouper = lambda ix: self.df.y.ix[ix] > 0
        vals = dist._box_reshape(self.df.y, grouper, None, None)[0]
        nt.assert_equal(len(vals), 2)
        low, high = vals
        nt.assert_true(low.max() <= 0)
        nt.assert_true(high.min() > 0)

    def test_dict_groupby(self):
        """Test boxplot groupby using a dictionary."""
        grouper = {i: "A" if i % 2 else "B" for i in self.df.y.index}
        vals = dist._box_reshape(self.df.y, grouper, None, None)[0]
        nt.assert_equal(len(vals), 2)
        a, b = vals
        npt.assert_array_equal(self.df.y.iloc[1::2], a)
        npt.assert_array_equal(self.df.y.iloc[::2], b)

    def test_1d_labels(self):
        """Test boxplot labels for 1D data."""
        x_1d_array = self.x.ravel()
        vals, xlabel, ylabel, names = dist._box_reshape(x_1d_array,
                                                        None, None, None)
        nt.assert_is(xlabel, None)
        nt.assert_is(ylabel, None)
        nt.assert_equal(names, [1])

        vals, xlabel, ylabel, names = dist._box_reshape(x_1d_array,
                                                        None, ["A"], None)

    def test_array_labels(self):
        """Test boxplot labels for a basic array."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.x,
                                                        None, None, None)
        nt.assert_is(xlabel, None)
        nt.assert_is(ylabel, None)
        nt.assert_equal(names, list(range(1, 4)))

        want_names = list("ABC")
        vals, xlabel, ylabel, names = dist._box_reshape(self.x,
                                                        None, want_names, None)
        nt.assert_equal(names, want_names)

    def test_dataframe_labels(self):
        """Test boxplot labels with DataFrame."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df,
                                                        None, None, None)
        nt.assert_equal(xlabel, self.x_df.columns.name)
        nt.assert_equal(ylabel, None)
        npt.assert_array_equal(names, self.x_df.columns)

    def test_ordered_dataframe_labels(self):
        """Test boxplot labels with DataFrame and specified order."""
        order = list("ZYX")
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df,
                                                        None, None, order)
        nt.assert_equal(xlabel, self.x_df.columns.name)
        npt.assert_array_equal(names, order)

    def test_groupby_labels(self):
        """Test labels with groupby vals."""
        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        None, None)
        nt.assert_equal(xlabel, self.g.name)
        nt.assert_equal(ylabel, self.y.name)
        npt.assert_array_equal(names, sorted(self.g.unique()))

    def test_ordered_groupby_labels(self):
        """Test labels with groupby vals and specified order."""
        order = list("BAC")
        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        order, None)
        nt.assert_equal(xlabel, self.g.name)
        nt.assert_equal(ylabel, self.y.name)
        npt.assert_array_equal(names, order)

    def test_pandas_names_override(self):
        """Test that names can override those inferred from Pandas objects."""
        want_names = ["ex", "why", "zee"]
        vals, xlabel, ylabel, names = dist._box_reshape(self.x_df, None,
                                                        want_names, None)
        nt.assert_equal(names, want_names)

        vals, xlabel, ylabel, names = dist._box_reshape(self.y, self.g,
                                                        want_names, None)
        nt.assert_equal(names, want_names)

    def test_bad_order_length(self):
        """Test for error when order and names lengths mismatch."""
        with nt.assert_raises(ValueError):
            dist._box_reshape(self.x_df, None, range(5), range(6))

    def test_bad_order_type(self):
        """Test for error when trying to order with a vanilla array."""
        with nt.assert_raises(ValueError):
            dist._box_reshape(self.x, None, None, range(5))

