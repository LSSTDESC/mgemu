"""
Gaussian Process prediction and emulator output for given training redshifts.
"""

from .load import model_load
from .scale import scale01, DEFAULT_MEAN, DEFAULT_STD
import numpy as np
import pickle
from sklearn.decomposition import PCA
import gpflow

__all__ = ("gp_predict", "gp_emu",)


def gp_predict(gpmodel, para_array_rescaled):
    """Returns a Gaussian Process prediction of PCA weights

    Parameters
    ----------

    gpmodel: GPflow object
        Trained GPflow model corresponding to a particular redshift

    para_array_rescaled: ndarray of shape (5, )
        Rescaled Cosmological parameters


    Returns
    _______

    W_predArray: float or ndarray of shape (nRankMax, )
        Mean GP prediction of the PCA weights

    W_varArray: float or ndarray of shape (nRankMax, )
        Variance in the GP prediction of the PCA weights

    """

    m1p = gpmodel.predict_f(para_array_rescaled)

    # [0] is the mean and [1] is the std of the prediction
    W_predArray = m1p[0]
    W_varArray = m1p[1]

    return W_predArray, W_varArray


def gp_emu(gpmodel, pcamodel, para_array):
    """Returns the emulator prediction of Boost in power spectrum, for a given snapshot.

    Parameters
    ----------

    gpmodel: GPflow object
        Trained GPflow model corresponding to a particular redshift

    pcamodel: sklearn object
        Trained PCA model corresponding to a particular redshift

    para_array: ndarray of shape (5, )
        Cosmological parameters (not scaled)
        para_array[3] is f_R_0, not in a logarithmic scale


    Returns
    _______

    x_decoded: ndarray of shape (213, )
        Boost in power spectrum predicted for given GP and PCA models

    """
    para_array = np.array(para_array)
    para_array[3] = np.log10(para_array[3])
    para_array_rescaled = scale01(para_array, DEFAULT_MEAN, DEFAULT_STD)
    if len(para_array.shape) == 1:
        W_predArray, _ = gp_predict(
            gpmodel, np.expand_dims(para_array_rescaled, axis=0))
        x_decoded = pcamodel.inverse_transform(W_predArray)
        return np.squeeze(x_decoded)  # [0]
    else:
        raise NotImplementedError("Batch outputs not implemented.")
