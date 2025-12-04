from RK4IP_z import fiber_propagation
from functions import read_hdf5 as rdhd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.constants import c, pi
from scipy.fft import fftfreq
from datetime import datetime
import os

# Initial Pulse information
shape = "gaussian"
lambda0_um = 0.920  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 140e-15
if shape == "gaussian":
    T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
elif shape == "lorentzian":
    T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # pulse duration in seconds

###From Average Power###
# Pavg = 0.4  # Watts
# R_R = 80e6
# P0 = Pavg / (T0 * R_R)
Pavg = 1.2  # Watts
R_R = 80e6
P0 = Pavg / (T0 * R_R)
# P0 = 15e3

C = 0  # chirp

# Grid information
z_tot = 0.1  # Fiber length in m
z_steps = 2**10
dz = z_tot / z_steps
z = np.arange(0, z_steps) * dz  # z grid for simulation

T_span = 100 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, T_span / t_steps)  # freq grid for simulation
omega = 2 * np.pi * f
omega_diff = omega - omega0

# Fiber informaiton
folder_path = os.getcwd()
location = os.path.join(folder_path, "dispersion_data", "waveguide.h5")
waveguidename = "20230330_4_ideal"
data_label, data = rdhd(location, waveguidename)
print(data_label)


# Fiber informaiton
omega_data = data[data_label.index("omega")]
beta1 = data[data_label.index("beta1")]
beta2 = data[data_label.index("beta2")]
neff = data[data_label.index("n_eff")]
Aeff = data[data_label.index("A_eff")]
n = np.real(neff)
k = -np.imag(neff)
alpha = 2 * omega_data * k / c
beta0 = n * omega_data / c
beta0_spl = UnivariateSpline(omega_data, beta0, k=5)
beta1_spl = UnivariateSpline(omega_data, beta1, k=5)
alpha_spl = UnivariateSpline(omega_data, alpha, k=5)

beta_w = 1j * (beta0_spl(omega) - beta0_spl(omega0) - beta1_spl(omega0) * omega_diff)
n2 = 2.6e-20  # nonlinear index of glass at 920nm in m^2/W
A_eff = Aeff[np.argmin(abs(omega_data - omega0))]  # Effective mode area in m2
gamma = 2 * pi * n2 / lambda0 / A_eff
# gamma = 0
C = 0
z_tot = 0.1  # Fiber length in m


A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega_diff)
sim.source("sech", P0, T0, C)
sim.propagate("RK4IP", alpha, beta_w, gamma)

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
