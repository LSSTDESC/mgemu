"""
"""
# Use relative imports when importing objects from elsewhere in the library
from ..load import model_load

test_snap_ID = 98
test_nRankMax = 6


def test_model_load():
    res = model_load(test_snap_ID, test_nRankMax)
    assert res is not None
