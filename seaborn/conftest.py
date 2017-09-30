import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(47)
