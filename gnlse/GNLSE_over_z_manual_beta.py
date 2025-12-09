from RK4IP_z import fiber_propagation
import numpy as np
from scipy.fft import fftfreq
from scipy.constants import c, pi
from math import factorial
from datetime import datetime

# Initial Pulse information
shape = "sech"
lambda0_um = 0.850  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 50e-15
if shape == "gaussian":
    T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
elif shape == "sech":
    T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # pulse duration in seconds
P0 = 10e3  # peak power in W
C = 0  # chirp

# Grid information
z_tot = 0.1  # Fiber length in m
z_steps = 2**14
dz = z_tot / z_steps
z = np.arange(0, z_steps) * dz  # z grid for simulation

T_span = 200 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, dt)  # freq grid for simulation
omega = 2 * np.pi * f

# Fiber informaiton
beta2 = -1.276e-26
beta3 = 8.119e-41
beta4 = -1.321e-55
beta5 = 3.032e-70
beta6 = -4.196e-85
beta7 = 2.57e-100
# beta2 = 1.276e-26
# beta3 = 0
# beta4 = 0
# beta5 = 0
# beta6 = 0
# beta7 = 0
alpha = 0
beta_w = 1j * (
    beta2 / factorial(2) * omega**2
    + beta3 / factorial(3) * omega**3
    + beta4 / factorial(4) * omega**4
    + beta5 / factorial(5) * omega**5
    + beta6 / factorial(6) * omega**6
    + beta7 / factorial(7) * omega**7
)  # relevent propagation constant order from is from 2
gamma = 0.045  # nonlinear coeff from the fiber in W^-1/m
fr = 0.18

A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega)
sim.source(shape, P0, T0, C)
sim.propagate("SSFM_symmetric", alpha, beta_w, gamma, fr)

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
