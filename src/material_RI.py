import numpy as np
from scipy.constants import c
from scipy.interpolate import UnivariateSpline
import h5py
from functions import read_hdf5


def sellmeier(wl, b1, b2, b3, c1, c2, c3):
    return np.sqrt(
        1
        + b1 * wl**2 / (wl**2 - c1**2)
        + b2 * wl**2 / (wl**2 - c2**2)
        + b3 * wl**2 / (wl**2 - c3**2)
    )


def bk7(wavelength):
    b1 = 1.03961212
    b2 = 0.231792344
    b3 = 1.01046945
    c1 = 6.00069867 * 1e-3
    c2 = 2.00179144 * 1e-2
    c3 = 1.03560653 * 1e2

    n = sellmeier(wavelength, b1, b2, b3, c1, c2, c3)
    return {"n": n}


def fused_silica(wavelength):
    b1 = 0.6961663
    b2 = 0.4079426
    b3 = 0.8974794
    c1 = 0.0684043
    c2 = 0.1162414
    c3 = 9.896161

    n = sellmeier(wavelength, b1, b2, b3, c1, c2, c3)
    return {"n": n}


def tolluene(wavelength):
    b1 = 1.17477
    b2 = 0
    b3 = 0
    c1 = np.sqrt(0.01825)
    c2 = 0
    c3 = 0

    n = sellmeier(wavelength, b1, b2, b3, c1, c2, c3)
    return {"n": n}
