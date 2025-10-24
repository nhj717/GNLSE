from splitstep import fiber_propagation
from functions import read_hdf5 as rdhd
from numpy import flip, imag, real, sqrt, log
from scipy.interpolate import UnivariateSpline
from scipy.constants import c, pi
from datetime import datetime
import os

folder_path = os.getcwd()
location = os.path.join(folder_path, "dispersion_data", "waveguide.h5")
waveguidename = "pcf"
data_label, data = rdhd(location, waveguidename)
print(data_label)

# Initial Pulse information
E_pulse = 8e-9  # pulse energy in J
# P0 = 1e3                                   #peak power in W
lambda0_um = 0.920  # pump wl in um
lambda0 = lambda0_um * 1e-6
w0 = 2 * pi * c / lambda0
T_FWHM = 150e-15
T0 = T_FWHM / (2 * sqrt(log(2)))
P0 = E_pulse / T_FWHM  # peak power in W

# Fiber informaiton
omega = data[data_label.index("omega")]
beta1 = data[data_label.index("beta1")]
beta2 = data[data_label.index("beta2")]
beta3 = data[data_label.index("beta3")]
neff = data[data_label.index("n_eff")]
n = real(neff)
k = -imag(neff)
alpha = 2 * omega * k / c
beta1_spl = UnivariateSpline(omega, beta1, k=5)
beta2_spl = UnivariateSpline(omega, beta2, k=5)
beta3_spl = UnivariateSpline(omega, beta3, k=5)
n_spl = UnivariateSpline(omega, n, k=5)
alpha_spl = UnivariateSpline(omega, alpha, k=5)
s = 1 / w0 * 0  # self steepening in seconds
tr = 3e-15 * 0  # Raman scattering in seconds
n2 = 2.6e-20  # nonlinear index in m^2/W
A_eff = 9.2e-12  # Effective mode area in m2
gamma = 2 * pi * n2 / lambda0 / A_eff
# gamma = 0
C = 0
z_tot = 0.105  # Fiber length in m


A = datetime.now()

# sim = fiber_propagation(lambda0, P0, T0, z_tot)

sim = fiber_propagation(lambda0, P0, T0, z_tot, C, 0, 20e-27, 0.1e-39, s, tr, gamma)
sim.run("gaussian")

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
