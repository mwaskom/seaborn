"""Tests for timeseries plotting utilities."""
import numpy as np
import pandas as pd

import nose.tools as nt
"""
from . import PlotTestCase
from .. import timeseries_new as tsn
from .. import utils
"""
from seaborn.tests import PlotTestCase
import seaborn.timeseries_new as tsn
import seaborn.utils


class TestTimeSeriesPlotter(PlotTestCase):

    gammas_time = 'timepoint'
    gammas_value = 'BOLD signal'
    gammas_unit = 'subject'
    gammas_condition = 'ROI'

    gammas = utils.load_dataset('gammas')

    pass

