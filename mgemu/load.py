"""
Loading pretrained GPflow and sklearn models.
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import gpflow


DEFAULT_PCA_RANK = 6

__all__ = ("model_load", "model_load_all", )

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def model_load(snap_ID, nRankMax=DEFAULT_PCA_RANK):
    """Returns pretrained GP and PCA models (saved in ./models/) for given snapshot.

    Parameters
    ----------

    snap_ID: int between 0 to 99
        Corresponds to different time stamps

    nRankMax: int
        Number of truncated PCA bases. Only valid nRankMax for now is 6.

    Returns
    _______

    GPm: GPflow predictor object
    PCAm: sklearn predictor object.

    """

    GPmodelname = "GP_smooth_rank" + str(nRankMax) + "snap" + str(snap_ID)
    GPmodel = os.path.join(_THIS_DRNAME, "models", GPmodelname)

    PCAmodelname = "PCA_smooth_rank" + str(nRankMax) + "snap" + str(snap_ID)
    PCAmodel = os.path.join(_THIS_DRNAME, "models", PCAmodelname)

    ctx_for_loading = gpflow.saver.SaverContext(autocompile=False)
    saver = gpflow.saver.Saver()
    GPm = saver.load(GPmodel, context=ctx_for_loading)
    GPm.clear()
    GPm.compile()
    PCAm = pickle.load(open(PCAmodel, 'rb'))
    return GPm, PCAm



def model_load_all(nRankMax=DEFAULT_PCA_RANK):
    """Returns pretrained all GP and PCA models (saved in ./models/) for given snapshot.

    Parameters
    ----------

    snap_ID: int between 0 to 99
        Corresponds to different time stamps

    nRankMax: int
        Number of truncated PCA bases. Only valid nRankMax for now is 6.

    Returns
    _______

    GPm: GPflow predictor object
    PCAm: sklearn predictor object.

    """

    GPmodel_list = []
    PCAmodel_list = []

    
    for snap_ID in range(100):
        GPm, PCAm = model_load(snap_ID, nRankMax)

        GPmodel_list.append(GPm)
        PCAmodel_list.append(PCAm)

    return GPmodel_list, PCAmodel_list
