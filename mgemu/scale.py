"""
Rescaling the input parameters.
"""

import numpy as np

__all__ = ("scale01",)

DEFAULT_MEAN = np.array(
    [0.1369883,   0.95245613,  0.80035087, -5.93216374,  1.97894743])
DEFAULT_STD = np.array(
    [0.01013207, 0.05845525, 0.0567193, 1.17136822, 1.21287971])


def scale01(f, fmean=DEFAULT_MEAN, fstd=DEFAULT_STD):
    """Returns a standardized rescaling of the input values.
     by mean and variance of the training scheme

    Parameters
    ----------

    f: ndarray of shape (5, )
        Input Cosmological parameters

    fmean: ndarray of shape (5, )
        Mean of the training design, given in 'model/paralims_nCorr_val_2.txt'

    fstd: ndarray of shape (5, )
        Standard deviation of the training design, given in 'model/paralims_nCorr_val_2.txt'

    Returns
    _______

    f_rescaled: ndarray of shape (5, )
        Rescaled Cosmological parameters

    """

    return (f - fmean)/fstd
