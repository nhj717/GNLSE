from scipy.constants import c
from scipy.interpolate import UnivariateSpline
from numpy import pi


# loads h5 data of n_eff and converts it to beta1, beta2 and beta3 data
def material_dispersion(wl_um, n):
    wl = wl_um * 1e-6
    n_spl = UnivariateSpline(wl, n, k=5)
    d2_n_spl = n_spl.derivative(n=2)
    D = -wl / c * d2_n_spl(wl) * 1e6
    return {"D": D}


# loads h5 data of n_eff and converts it to beta1, beta2 and beta3 data
def mode_dispersion(wl_um, n):
    wl = wl_um * 1e-6
    beta0 = 2 * pi / wl
    beta = n * beta0
    w = c * beta0

    beta_spl = UnivariateSpline(w, beta, k=5)
    beta1 = beta_spl.derivative(n=1)
    beta1_wl = beta1(w)
    beta2 = beta_spl.derivative(n=2)
    beta2_wl = beta2(w)
    beta3 = beta_spl.derivative(n=3)
    beta3_wl = beta3(w)
    D = -2 * pi * c / wl**2 * beta2_wl * 1e6
    arg_dict = {"beta1": beta1_wl, "beta2": beta2_wl, "beta3": beta3_wl, "D": D}
    return arg_dict
