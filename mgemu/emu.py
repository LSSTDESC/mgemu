"""
Redshift interpolation for Boost in power spectrum.
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA
import gpflow
from .load import model_load, model_load_all, DEFAULT_PCA_RANK
from .scale import scale01
from .gp import gp_emu


__all__ = ("emu", "emu_fast", )

DEFAULT_SCALE_FACTOR = np.linspace(0.0298, 1.00000, 100).round(decimals=7)
DEFAULT_KBINS = np.logspace(np.log(0.03), np.log(
    3.5), 301, base=np.e).round(decimals=7)
DEFAULT_KMASK = np.array([0,   1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  25,  26,  27,
                          28,  29,  30,  31,  32,  33,  34,  35,  36,  38,  39,  40,  41,
                          42,  43,  44,  45,  47,  48,  49,  50,  51,  52,  54,  55,  56,
                          57,  58,  60,  61,  62,  63,  64,  65,  66,  67,  69,  70,  71,
                          73,  74,  76,  77,  79,  80,  82,  84,  85,  87,  88,  89,  91,
                          93,  96,  99, 101, 102, 107, 108, 111, 118])
LAST_SNAP = 99
TOT_BINS = 300

ALL_GP, ALL_PCA = model_load_all(nRankMax=DEFAULT_PCA_RANK)

def emu(Omh2, ns, s8, fR0, n, z):
    """Returns the emulator prediction of Boost in power spectrum, for a redshift between 0 < z < 49

    Parameters
    ----------

    Omh2: float
        Physical matter density parameter (O_m h^2) in range [0.12, 0.15]. Here h = 0.67, a constant in the emulator design

    ns: float
        Scalar spectral index (n_s) in range [0.85, 1.1]

    s8: float
        The present root-mean-square matter fluctuation averaged over a sphere of radius 8Mpc/h (\sigma_8), in the range [0.7, 0.9]

    fR0: float
        Hu-Sawicki model parameter (f_R_0) in range [1e-8, 1e-4]

    n: float
        Hu-Sawicki model parameter (n) in range [0, 4]

    z: float
        Redshift between [0, 49]



    Returns
    _______

    Pk_interp: ndarray of shape (213, )
        Boost in power spectrum predicted by interpolating between training redshift values.

    """
    # redshift of all snapshots
    # altertively z_all = np.loadtxt('TrainedModels/timestepsCOLA.txt', skiprows=1)[:, 1]
    a = DEFAULT_SCALE_FACTOR
    z_all = (1/a) - 1

    # k values of summary statistics
    # alternatively kvals = np.loadtxt('TrainedModels/'ratiobins.txt')[:,0]
    kb = DEFAULT_KBINS
    k1 = 0.5*(kb[0:-1] + kb[1:]).round(decimals=7)
    kmask = DEFAULT_KMASK
    kvals = k1[..., [i for i in np.arange(TOT_BINS) if i not in kmask]]

    if (z == 0):
        # No redshift interpolation for z=0
        GPm, PCAm = model_load(snap_ID=LAST_SNAP, nRankMax=DEFAULT_PCA_RANK)
        Pk_interp = gp_emu(GPm, PCAm, [Omh2, ns, s8, fR0, n])

    else:
        # Linear interpolation between z1 < z < z2
        snap_idx_nearest = (np.abs(z_all - z)).argmin()
        if (z > z_all[snap_idx_nearest]):
            snap_ID_z1 = snap_idx_nearest - 1
        else:
            snap_ID_z1 = snap_idx_nearest
        snap_ID_z2 = snap_ID_z1 + 1

        GPm1, PCAm1 = model_load(snap_ID=snap_ID_z1, nRankMax=DEFAULT_PCA_RANK)
        Pk_z1 = gp_emu(GPm1, PCAm1, [Omh2, ns, s8, fR0, n])
        z1 = z_all[snap_ID_z1]

        GPm2, PCAm2 = model_load(snap_ID=snap_ID_z2, nRankMax=DEFAULT_PCA_RANK)
        Pk_z2 = gp_emu(GPm2, PCAm2, [Omh2, ns, s8, fR0, n])
        z2 = z_all[snap_ID_z2]

        Pk_interp = np.zeros_like(Pk_z1)
        Pk_interp = Pk_z2 + (Pk_z1 - Pk_z2)*(z - z2)/(z1 - z2)
    return Pk_interp, kvals


def emu_fast(Omh2, ns, s8, fR0, n, z):
    """Returns the emulator prediction of Boost in power spectrum, for a redshift between 0 < z < 49

    Parameters
    ----------

    Omh2: float
        Physical matter density parameter (O_m h^2) in range [0.12, 0.15]. Here h = 0.67, a constant in the emulator design

    ns: float
        Scalar spectral index (n_s) in range [0.85, 1.1]

    s8: float
        The present root-mean-square matter fluctuation averaged over a sphere of radius 8Mpc/h (\sigma_8), in the range [0.7, 0.9]

    fR0: float
        Hu-Sawicki model parameter (f_R_0) in range [1e-8, 1e-4]

    n: float
        Hu-Sawicki model parameter (n) in range [0, 4]

    z: float
        Redshift between [0, 49]



    Returns
    _______

    Pk_interp: ndarray of shape (213, )
        Boost in power spectrum predicted by interpolating between training redshift values.

    """
    # redshift of all snapshots
    # altertively z_all = np.loadtxt('TrainedModels/timestepsCOLA.txt', skiprows=1)[:, 1]
    a = DEFAULT_SCALE_FACTOR
    z_all = (1/a) - 1

    # k values of summary statistics
    # alternatively kvals = np.loadtxt('TrainedModels/'ratiobins.txt')[:,0]
    kb = DEFAULT_KBINS
    k1 = 0.5*(kb[0:-1] + kb[1:]).round(decimals=7)
    kmask = DEFAULT_KMASK
    kvals = k1[..., [i for i in np.arange(TOT_BINS) if i not in kmask]]

    if (z == 0):
        # No redshift interpolation for z=0
        GPm, PCAm = ALL_GP[LAST_SNAP]
        PCAm = ALL_PCA[LAST_SNAP]
        Pk_interp = gp_emu(GPm, PCAm, [Omh2, ns, s8, fR0, n])

    else:
        # Linear interpolation between z1 < z < z2
        snap_idx_nearest = (np.abs(z_all - z)).argmin()
        if (z > z_all[snap_idx_nearest]):
            snap_ID_z1 = snap_idx_nearest - 1
        else:
            snap_ID_z1 = snap_idx_nearest
        snap_ID_z2 = snap_ID_z1 + 1

        GPm1 = ALL_GP[snap_ID_z1]
        PCAm1 = ALL_PCA[snap_ID_z1] 
        Pk_z1 = gp_emu(GPm1, PCAm1, [Omh2, ns, s8, fR0, n])
        z1 = z_all[snap_ID_z1]


        GPm2 = ALL_GP[snap_ID_z2]
        PCAm2 = ALL_PCA[snap_ID_z2]  
        Pk_z2 = gp_emu(GPm2, PCAm2, [Omh2, ns, s8, fR0, n])
        z2 = z_all[snap_ID_z2]

        Pk_interp = np.zeros_like(Pk_z1)
        Pk_interp = Pk_z2 + (Pk_z1 - Pk_z2)*(z - z2)/(z1 - z2)
    return Pk_interp, kvals
