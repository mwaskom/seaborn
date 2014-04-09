import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
from numpy.testing.decorators import skipif

from .. import clustering as cl


class TestMatrixPlotter(object):

    rs = np.random.RandomState(88)
    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           d=rs.randint(-2, 3, 60),
                           y=rs.gamma(4, size=60),
                           s=np.tile(list("abcdefghij"), 6)))
    df["z"] = df.y + rs.randn(60)
    df["y_na"] = df.y.copy()
    df.y_na.ix[[10, 20, 30]] = np.nan

    def test_establish_variables_from_frame(self):
        p = cl._MatrixPLotter()
        p.establish_variables(self.df, pivot_kws={})
