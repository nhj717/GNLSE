from RK4IP import fiber_propagation
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

# P0 = 1e3                                   #peak power in W
lambda0_um = 0.850 # pump wl in um
lambda0 = lambda0_um * 1e-6
w0 = 2 * pi * c / lambda0
T_FWHM = 50e-15
T0 = T_FWHM / (2 * sqrt(log(2)))
P0 = 1e4

# Fiber informaiton
beta2 = -1.276e-2
beta3 = 8.119e-5
beta4 = -1.321e-7
alpha = 0

s = 1 / w0  # self steepening in seconds
tr = 3e-17  # Raman scattering in seconds
n2 = 2.6e-20  # nonlinear index in m^2/W
A_eff = 9.2e-12  # Effective mode area in m2
gamma = 0.045
# gamma = 0
C = 0
z_tot = 0.1  # Fiber length in m


A = datetime.now()
# sim = fiber_propagation(lambda0, P0, T0, z_tot)
# sim = fiber_propagation(lambda0, P0, T0, z_tot, C, 0, 0, 0, 0, gamma)
sim = fiber_propagation(
    lambda0, P0, T0, z_tot, C, 0, beta2, beta3, beta4, gamma
)
sim.source("gaussian")
sim.propagate()

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
