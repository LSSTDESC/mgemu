import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import gpflow

"""

Loading pretrained GP and PCA models (saved in ./models/) for given snapshot

nRankMax: Number of basis vectors in truncated PCA, default = 6

"""

DEFAULT_PCA_RANK = 6

__all__ = ("model_load",)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))

def model_load(snap_ID, nRankMax = DEFAULT_PCA_RANK):
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

