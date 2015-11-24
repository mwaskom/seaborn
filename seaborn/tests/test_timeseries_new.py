"""Tests for timeseries plotting utilities."""
import numpy as np
import pandas as pd

import nose.tools as nt

from . import PlotTestCase
from .. import timeseries_new as tsn
from .. import utils

class TestTimeSeriesPlotter(PlotTestCase):

    gamms = utils.load_dataset("gammas")

    pass

