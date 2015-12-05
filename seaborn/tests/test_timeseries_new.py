"""Tests for timeseries plotting utilities."""
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import matplotlib as mpl
import matplotlib.pyplot as plt
import nose
import nose.tools as nt
"""
from . import PlotTestCase
from .. import timeseries_new as tsn
from .. import utils
"""
from seaborn.tests import PlotTestCase
from seaborn.timeseries_new import _TimeSeriesPlotter, tsplot
from seaborn import utils
from seaborn.palettes import color_palette

# TODO: add a test that checks if computed data is also plotted - this closes the circle
# TODO: replace fake data by gammas


class TestDataInit(PlotTestCase):

    df = pd.DataFrame({'condition': ['a', 'a', 'a', 'a', 'b', 'b'],
                       'unit': [0, 0, 1, 1, 0, 0],
                       'time': [0, 1, 0, 1, 0, 1],
                       'value': [1, 1, 2, 2, 3, 3]})
    # TODO: better wording for df_kwargs?
    df_kwargs = dict(value='value', time='time', unit='unit',
                     condition='condition', legend=True)

    # TODO: what about `unit is not None`?
    @nt.raises(ValueError)
    def test_init_value_error(self):
        _TimeSeriesPlotter(np.array([1, 2]), condition=None,
                           color={'a': 'green', 'b': 'blue'})

    def test_init_from_df(self):

        tsp = _TimeSeriesPlotter(self.df, **self.df_kwargs)

        pdt.assert_frame_equal(self.df, tsp.data)
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=True,
                                         legend_title='condition'))
        nt.assert_equal(tsp.labels, dict(xlabel='time', ylabel='value'))

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
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=False, legend_title=None))
        nt.assert_equal(tsp.labels, dict(xlabel=None, ylabel=None))

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
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=True,
                                         legend_title='condicion'))
        nt.assert_equal(tsp.labels, dict(xlabel='tiempo', ylabel='valor'))

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
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=False, legend_title=None))
        nt.assert_equal(tsp.labels, dict(xlabel=None, ylabel=None))

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
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=False, legend_title=None))
        nt.assert_equal(tsp.labels, dict(xlabel=None, ylabel=None))

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
        nt.assert_equal(tsp.names, dict(condition='condition', unit='unit',
                                        time='time', value='value'))
        nt.assert_equal(tsp.legend, dict(legend=True,
                                         legend_title='condicion'))
        nt.assert_equal(tsp.labels, dict(xlabel='tiempo', ylabel='valor'))

    # interpolate is False and no values for marker or linestyle are supplied,
    # use default marker='o' and ls=''
    def test_interpolate_is_False_and_no_kwargs_supplied(self):

        tsp = _TimeSeriesPlotter(self.df,
                                 interpolate=False,
                                 #no marker
                                 #no ls
                                 **self.df_kwargs)

        nt.assert_equal(tsp.kwargs, dict(linestyle='', marker='o'))

    # interpolate is False and values for marker and linestyle are supplied,
    # use these"
    def test_interpolate_is_False_and_kwargs_supplied(self):

        tsp = _TimeSeriesPlotter(self.df,
                                 interpolate=False,
                                 marker='x',
                                 ls='-.',
                                 **self.df_kwargs)

        nt.assert_equal(tsp.kwargs, dict(linestyle='-.', marker='x'))

    # interpolate is True and no values for marker or linestyle are supplied,
    # use default marker='' and ls='-'
    def test_interpolate_is_True_and_no_kwargs_supplied(self):

        tsp = _TimeSeriesPlotter(self.df,
                                 interpolate=True,
                                 #no marker
                                 #no ls
                                 **self.df_kwargs)
        nt.assert_equal(tsp.kwargs, dict(linestyle='-', marker=''))

    # interpolate is True and values for marker and linestyle are supplied,
    # use these
    def test_interpolate_is_True_and_kwargs_supplied(self):

        tsp = _TimeSeriesPlotter(self.df,
                                 interpolate=True,
                                 marker='x',
                                 ls='-.',
                                 **self.df_kwargs)
        nt.assert_equal(tsp.kwargs, dict(linestyle='-.', marker='x'))

    # make sure that err_style is iterable
    def test_err_style_is_string_type(self):
        err_style = 'ci_band'
        err_style_expected = [err_style]
        tsp = _TimeSeriesPlotter(self.df, err_style=err_style,
                                 **self.df_kwargs)
        nt.assert_equal(tsp.err_style, err_style_expected)

    # make sure that err_style is iterable
    def test_err_style_is_None(self):
        err_style = None
        err_style_expected = []
        tsp = _TimeSeriesPlotter(self.df, err_style=err_style,
                                 **self.df_kwargs)
        nt.assert_equal(tsp.err_style, err_style_expected)

    # make sure that ci is iterable
    def test_ci(self):
        ci = 68
        tsp = _TimeSeriesPlotter(self.df, ci=ci,
                                 **self.df_kwargs)
        ci_expected = [ci]
        nt.assert_equal(tsp.ci, ci_expected)


class TestColor(PlotTestCase):

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


class TestPlotData(PlotTestCase):

    gammas = utils.load_dataset("gammas")
    gammas_kwargs = dict(time="timepoint", value="BOLD signal",
                         unit="subject", condition="ROI")

    color = {'AG': 'red', 'IPS': 'green', 'V1': 'blue'}
    color = {k: mpl.colors.colorConverter.to_rgb(c)
             for k, c in color.items()}

    def test_compute_plot_data(self):

        n_boot = 200
        estimator = np.mean
        ci = [68, 99]
        tsp = _TimeSeriesPlotter(self.gammas, estimator=estimator, n_boot=n_boot,
                                 color=self.color, ci=ci, **self.gammas_kwargs)

        for c, plot_data in enumerate(tsp._compute_plot_data()):
            cond, df_c, x, boot_data, cis, central_data = plot_data
            (ci68_low, ci68_high), (ci99_low, ci99_high) = cis
            # condition in the right order?
            cond_expected = self.gammas[self.gammas_kwargs['condition']].unique()[c]
            nt.assert_equal(cond, cond_expected)
            # are colors mapped correctly?
            color_expected = self.color[cond_expected]
            nt.assert_equal(tsp.colors[c], color_expected)
            # condition-dataframe
            gammas_c = self.gammas[self.gammas[self.gammas_kwargs['condition']]
                                   == cond_expected]
            df_c_expected = gammas_c.pivot(index=self.gammas_kwargs['unit'],
                                           columns=self.gammas_kwargs['time'],
                                           values=self.gammas_kwargs['value'])
            pdt.assert_frame_equal(df_c, df_c_expected)
            # time
            x_expected = self.gammas[self.gammas_kwargs['time']].unique()
            npt.assert_allclose(x, x_expected)
            # central data
            central_data_expected = estimator(df_c, axis=0).values
            npt.assert_allclose(central_data, central_data_expected)
            # check cis
            npt.assert_array_less(ci68_low, central_data)
            npt.assert_array_less(central_data, ci68_high)
            npt.assert_array_less(ci99_low, central_data)
            npt.assert_array_less(central_data, ci99_high)
            npt.assert_array_less(ci68_high - ci68_low, ci99_high - ci99_low)
            # check number of bootstrapped samples
            nt.assert_equal(boot_data.shape[0], n_boot)


if __name__ == '__main__':
    nose.runmodule(exit=False)
