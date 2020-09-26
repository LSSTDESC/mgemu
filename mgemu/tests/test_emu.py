"""
"""
from ..emu import emu
import numpy as np

test_Omh2 = 0.126
test_ns = 0.971
test_s8 = 0.82
test_fR0 = 1e-5
test_n = 1.0
test_z = 0.1


def test_emu():
    res = emu(test_Omh2, test_ns, test_s8, test_fR0, test_n, test_z)
    assert np.all(np.isfinite(res))
