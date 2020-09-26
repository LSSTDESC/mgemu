"""
"""
from .load import model_load
from .scale import scale01, DEFAULT_MEAN, DEFAULT_STD
import numpy as np
import pickle
from sklearn.decomposition import PCA
import gpflow

__all__ = ("gp_predict", "gp_emu",)

def gp_predict(gpmodel, para_array_rescaled):
    '''
    GP prediction of the latent space variables
    '''
    m1p = gpmodel.predict_f(para_array_rescaled)
    
    # [0] is the mean and [1] is the std of the prediction
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    
    return W_predArray, W_varArray

def gp_emu(gpmodel, pcamodel, para_array):
    '''
    Combining GP prediction with PCA reconstrution to output p(k) ratio for the given snapshot
    '''
    para_array = np.array(para_array)
    para_array[3] = np.log10(para_array[3])
    para_array_rescaled = scale01(para_array, DEFAULT_MEAN, DEFAULT_STD)
    if len(para_array.shape) == 1:
        W_predArray, _ = gp_predict(gpmodel, np.expand_dims(para_array_rescaled, axis=0))
        x_decoded = pcamodel.inverse_transform(W_predArray)
        return np.squeeze(x_decoded)#[0]
    else:
        raise NotImplementedError("Batch outputs not implemented.")
