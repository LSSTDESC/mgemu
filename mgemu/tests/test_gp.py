"""
"""
import numpy as np
from ..gp import gp_predict, gp_emu
from ..load import model_load
from ..scale import scale01, DEFAULT_MEAN, DEFAULT_STD


test_gp_model, test_pca_model = model_load(98, 6)
test_input_cosmological_params = np.array([0.126, 0.967, 0.8, 1e-5, 1.0])
test_input_logfr0 = np.array([0.126, 0.967, 0.8, -5, 1.0])
scaled_input = scale01(test_input_logfr0, DEFAULT_MEAN, DEFAULT_STD)


def test_gp_predict():
    res = gp_predict(test_gp_model, np.expand_dims(scaled_input, axis=0))
    assert np.all(np.isfinite(res))


def test_gp_emu():
    res = gp_emu(test_gp_model, test_pca_model, test_input_cosmological_params)
    assert np.all(np.isfinite(res))
