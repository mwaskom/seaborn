import numpy as np
import matplotlib.pyplot as plt
import pytest

# Attempted workaround for change in pandas 0.21
# Can be removed when matplotlib 2.2 is released
from pandas.tseries import converter
converter.register()


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(47)
