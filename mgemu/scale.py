import numpy as np

__all__ = ("scale01",)

DEFAULT_MEAN = np.array([ 0.1369883,   0.95245613,  0.80035087, -5.93216374,  1.97894743])
DEFAULT_STD = np.array([0.01013207, 0.05845525, 0.0567193, 1.17136822, 1.21287971])

def scale01(f, fmean = DEFAULT_MEAN , fstd = DEFAULT_STD):
    '''
    Normalizing the input parameters by mean and variance of the training scheme (log(fr0))
    fmean, ftd: are the mean and std of the experimental design, given in 'model/paralims_nCorr_val_2.txt'
    '''
    return (f - fmean)/fstd
