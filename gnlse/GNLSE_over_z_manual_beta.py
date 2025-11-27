from RK4IP_z_manual import fiber_propagation
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
lambda0_um = 0.850  # pump wl in um
lambda0 = lambda0_um * 1e-6
w0 = 2 * pi * c / lambda0
T_FWHM = 50e-15
T0 = T_FWHM / (2 * log(1 + sqrt(2)))
P0 = 10e3  # peak power in W

# Fiber informaiton
beta2 = -1.276e-26
beta3 = 8.119e-41
beta4 = -1.321e-55
beta5 = 3.032e-70
beta6 = -4.196e-85
beta7 = 2.57e-100
alpha = 0

gamma = 0.045
C = 0
z_tot = 0.1  # Fiber length in m


A = datetime.now()
sim = fiber_propagation(
    lambda0, P0, T0, z_tot, C, alpha, beta2, beta3, beta4, beta5, beta6, beta7, gamma
)
sim.source("sech")
sim.propagate()

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
