"""Small utilities related to plotting."""
import numpy as np
from numpy.testing import assert_array_equal

from .. import utils


def test_ci_to_errsize():

    cis = [(.5, 1.75),
           (1, 1.5)]

    heights = [1, 1.5]

    actual_errsize = np.array([[.5, .5],
                               [.75, 0]])

    test_errsize = utils.ci_to_errsize(cis, heights)

    assert_array_equal(actual_errsize, test_errsize)
