"""Tests for timeseries plotting utilities."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import matplotlib as mpl
import nose
import nose.tools as nt
"""
from . import PlotTestCase
from .. import timeseries_new as tsn
from .. import utils
"""
from seaborn.tests import PlotTestCase
from seaborn.timeseries_new import _TimeSeriesPlotter
from seaborn import utils
from seaborn.palettes import color_palette

"""
class TestTimeSeriesPlotterDataInit(PlotTestCase):

    # TODO: what about `unit is not None`?
    @nt.raises(ValueError)
    def test_init_value_error(self):
        _TimeSeriesPlotter(np.array([1, 2]), condition=None,
                          color={'a': 'green', 'b': 'blue'})

    def test_init_from_df(self):

        data = pd.DataFrame({'condition': ['a', 'a', 'a', 'a', 'b', 'b'],
                             'unit': [0, 0, 1, 1, 0, 0],
                             'time': [0, 1, 0, 1, 0, 1],
                             'value': [1, 1, 2, 2, 3, 3]})

        tsp = _TimeSeriesPlotter(data, value='value', time='time', unit='unit', condition='condition',
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

        tsp = _TimeSeriesPlotter(data, time=time, condition=condition, legend=False)

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

        tsp = _TimeSeriesPlotter(data, time=time, condition=condition, value='valor', legend=True)

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
        tsp = _TimeSeriesPlotter(data, time=time, condition=condition, legend=False)

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

        tsp = _TimeSeriesPlotter(data, time=time, condition=condition, legend=False)
        data_actual = tsp.data.sort_values(['condition', 'unit']).reset_index(drop=True)
        pdt.assert_frame_equal(data_expected, data_actual)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=False, legend_title=None) == tsp.legend
        assert dict(xlabel=None, ylabel=None) == tsp.labels

    def test_init_from_array_3d_1(self):

        data = np.array([[[1, 3],
                          [1, 3]],

                         [[2, 4],
                          [2, 4]]])
        time = pd.Series([0, 2], name='tiempo')
        condition = pd.Series(['a', 'b'], name='condicion')

        data_expected = pd.DataFrame({'condition': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                                      'unit': [0, 0, 1, 1, 0, 0, 1, 1],
                                      'time': [0, 2, 0, 2, 0, 2, 0, 2],
                                      'value': [1, 1, 2, 2, 3, 3, 4, 4]})

        tsp = _TimeSeriesPlotter(data, time=time, condition=condition, legend=True, value='valor')
        data_actual = tsp.data.sort_values(['condition', 'unit']).reset_index(drop=True)
        pdt.assert_frame_equal(data_expected, data_actual)
        assert dict(condition='condition', unit='unit', time='time', value='value') == tsp.names
        assert dict(legend=True, legend_title='condicion') == tsp.legend
        assert dict(xlabel='tiempo', ylabel='valor') == tsp.labels
"""

"""
import unittest
import mock

def check():
    return test()
def test():
    return "test"

class CheckTest(unittest.TestCase):
    @mock.patch('__main__.test')
    def test_test(self, mocked):
        mocked.return_value = "mocked"
        self.assertEqual(check(), "mocked")


if __name__ == '__main__':
    unittest.main()
"""

class TestTimeSeriesPlotterColor(PlotTestCase):

    # color is None and we have more different conditions than colors
    # in the current color cycle, therefore we use 'husl' palette
    def test_color_is_none_and_len_current_palatte_le_n_cond(self):
        color = None
        current_palette = utils.get_color_cycle()
        # create more conditions than len of current palette
        conditions = np.array(['cond_{}'.format(n) for n
                               in range(len(current_palette) + 1)])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = color_palette('husl', len(conditions))
        colors_expected = [mpl.colors.colorConverter.to_rgb(c) for c in colors_expected]

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is None and the length of the current color cycle is sufficient
    # for the number of different conditions, therefore we use the color cycle
    def test_color_is_none_and_len_current_palatte_geq_n_cond(self):
        color = None
        current_palette = utils.get_color_cycle()
        # create more conditions than len of current palette
        conditions = np.array(['cond_{}'.format(n) for n
                               in range(len(current_palette) - 1)])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = current_palette[:len(current_palette) - 1]
        colors_expected = [mpl.colors.colorConverter.to_rgb(c) for c in colors_expected]

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is a dict which maps each condition to a color
    def test_color_is_valid_dict(self):
        color = {'a': 'r', 'b': 'g'}
        conditions = np.array(['a', 'b'])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = [mpl.colors.colorConverter.to_rgb(color[c]) for c in conditions]

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is a dict which maps each condition to a color, but there are
    # invalid values, i.e. they cannot be converted to rgb.
    @nt.raises(ValueError)
    def test_color_is_dict_with_invalid_values(self):
        color = {'a': 'r', 'b': 'this is no color'}
        conditions = np.array(['a', 'b'])
        _TimeSeriesPlotter._set_up_color_palette(color, conditions)

    # color is a dict but fails to provide a color for all conditions
    @nt.raises(ValueError)
    def test_color_is_invalid_dict(self):
        color = {'a': 'r', 'b': 'g'}
        conditions = np.array(['a', 'b', 'c'])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = [mpl.colors.colorConverter.to_rgb(color[c]) for c in conditions]

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is a palette, therefore a color is created for each condition
    def test_color_is_palette(self):
        color = 'husl'
        conditions = np.array(['a', 'b'])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = color_palette(color, len(conditions))
        colors_expected = [mpl.colors.colorConverter.to_rgb(c) for c in colors_expected]

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is not None, not dict not palette, it is supposed to be a color,
    # therefore each condition gets the same color
    def test_color_is_color(self):
        color = 'red'
        conditions = np.array(['a', 'b'])
        colors = _TimeSeriesPlotter._set_up_color_palette(color, conditions)
        colors_expected = [mpl.colors.colorConverter.to_rgb(color)] * len(conditions)

        npt.assert_array_almost_equal(colors, colors_expected)

    # color is useless, raise ValueError
    @nt.raises(ValueError)
    def test_color_is_no_color(self):
        color = 'blaaa'
        conditions = np.array(['a', 'b'])
        _TimeSeriesPlotter._set_up_color_palette(color, conditions)


"""
class TestTimeSeriesPlotterPlotData(PlotTestCase):

    def test_compute_plot_data(self):
        pass


class TestTimeSeriesPlotterPlot(PlotTestCase):

    def test_axis_labels():
        pass
"""

if __name__ == '__main__':
    nose.runmodule(exit=False)