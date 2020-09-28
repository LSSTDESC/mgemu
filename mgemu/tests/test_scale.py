"""
"""
# Use relative imports when importing objects from elsewhere in the library
from ..scale import scale01
import numpy as np

test_input_cosmological_params = np.array([0.14,   0.967,  0.8, -5.0,  1.0])


def test_scale01():
    res = scale01(test_input_cosmological_params)
    assert res.shape == test_input_cosmological_params.shape
