"""Tests for timeseries plotting utilities."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

import nose
import nose.tools as nt
"""
from . import PlotTestCase
from .. import timeseries_new as tsn
from .. import utils
"""
from seaborn.tests import PlotTestCase
from seaborn.timeseries_new import TimeSeriesPlotter
import seaborn.utils




class TestTimeSeriesPlotter(PlotTestCase):

    # TODO: what about `unit is not None`?

    def test_init_from_df(self):

        data = pd.DataFrame({'condition': ['a', 'a', 'a', 'a', 'b', 'b'],
                             'unit': [0, 0, 1, 1, 0, 0],
                             'time': [0, 1, 0, 1, 0, 1],
                             'value': [1, 1, 2, 2, 3, 3]})

        tsp = TimeSeriesPlotter(data, value='value', time='time', unit='unit', condition='condition',
                                legend=True)

        pdt.assert_frame_equal(data, tsp.data)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=True, legend_title='condition') == tsp.legend
        assert dict(xlabel='time', ylabel='value') == tsp.labels

    def test_init_from_array_1d_0(self):

        data = np.array([1, 1])
        time = None
        condition = None

        data_expected = pd.DataFrame({'condition': [0, 0],
                                      'unit': [0, 0],
                                      'time': [0, 1],
                                      'value': [1, 1]})

        tsp = TimeSeriesPlotter(data, time=time, condition=condition, legend=False)

        pdt.assert_frame_equal(data_expected, tsp.data)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=False, legend_title=None) == tsp.legend
        assert dict(xlabel=None, ylabel=None) == tsp.labels

    def test_init_from_array_1d_1(self):

        data = np.array([1, 1])
        time = pd.Series([0, 2], name='tiempo')
        condition = pd.Series(['a'], name='condicion')

        data_expected = pd.DataFrame({'condition': ['a', 'a'],
                                      'unit': [0, 0],
                                      'time': [0, 2],
                                      'value': [1, 1]})

        tsp = TimeSeriesPlotter(data, time=time, condition=condition, value='valor', legend=True)

        pdt.assert_frame_equal(data_expected, tsp.data)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=True, legend_title='condicion') == tsp.legend
        assert dict(xlabel='tiempo', ylabel='valor') == tsp.labels


    def test_init_from_array_2d(self):

        data = np.array([[1, 1], [2, 2]])
        time = None
        condition = None

        data_expected = pd.DataFrame({'condition': [0, 0, 0, 0],
                                      'unit': [0, 0, 1, 1],
                                      'time': [0, 1, 0, 1],
                                      'value': [1, 1, 2, 2]})
        # TODO: is there a configuration I should try?
        tsp = TimeSeriesPlotter(data, time=time, condition=condition, legend=False)

        pdt.assert_frame_equal(data_expected, tsp.data)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=False, legend_title=None) == tsp.legend
        assert dict(xlabel=None, ylabel=None) == tsp.labels


    def test_init_from_array_3d_0(self):

        data = np.array([[[1, 3],
                          [1, 3]],

                         [[2, 4],
                          [2, 4]]])
        time = None
        condition = None

        data_expected = pd.DataFrame({'condition': [0, 0, 0, 0, 1, 1, 1, 1],
                                      'unit': [0, 0, 1, 1, 0, 0, 1, 1],
                                      'time': [0, 1, 0, 1, 0, 1, 0, 1],
                                      'value': [1, 1, 2, 2, 3, 3, 4, 4]})

        tsp = TimeSeriesPlotter(data, time=time, condition=condition, legend=False)
        data_actual = tsp.data.sort_values(['condition', 'unit']).reset_index(drop=True)
        pdt.assert_frame_equal(data_expected, data_actual)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=False, legend_title=None) == tsp.legend
        assert dict(xlabel=None, ylabel=None) == tsp.labels

    def test_init_from_array_3d_1(self):

        data = np.array([[[1, 1], [2, 2]],
                          [[3, 3], [4, 4]]])
        time = pd.Series([0, 2], name='tiempo')
        condition = pd.Series(['a', 'b'], name='condicion')

        data_expected = pd.DataFrame({'condition': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                                      'unit': [0, 0, 1, 1, 0, 0, 1, 1],
                                      'time': [0, 2, 0, 2, 0, 2, 0, 2],
                                      'value': [1, 1, 2, 2, 3, 3, 4, 4]})

        tsp = TimeSeriesPlotter(data, time=time, condition=condition, legend=True, value='valor')
        data_actual = tsp.data.sort_values(['condition', 'unit']).reset_index(drop=True)
        pdt.assert_frame_equal(data_expected, data_actual)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=True, legend_title='condicion') == tsp.legend
        assert dict(xlabel='tiempo', ylabel='valor') == tsp.labels


if __name__ == '__main__':
    nose.runmodule(exit=False)