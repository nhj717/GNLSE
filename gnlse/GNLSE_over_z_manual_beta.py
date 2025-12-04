from RK4IP_z import fiber_propagation
import numpy as np
from scipy.fft import fftfreq
from scipy.constants import c, pi
from math import factorial
from datetime import datetime

# Initial Pulse information
lambda0_um = 0.850  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 50e-15
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # pulse duration in seconds
P0 = 10e3  # peak power in W
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
beta2 = -1.276e-26
beta3 = 8.119e-41
beta4 = -1.321e-55
beta5 = 3.032e-70
beta6 = -4.196e-85
beta7 = 2.57e-100
alpha = 0
beta_w = 1j * (
    beta2 / factorial(2) * omega_diff**2
    + beta3 / factorial(3) * omega_diff**3
    + beta4 / factorial(4) * omega_diff**4
    + beta5 / factorial(5) * omega_diff**5
    + beta6 / factorial(6) * omega_diff**6
    + beta7 / factorial(7) * omega_diff**7
)  # relevent propagation constant order from is from 2
gamma = 0.045  # nonlinear coeff from the fiber in W^-1/m

A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega_diff)
sim.source("sech", P0, T0, C)
sim.propagate("RK4IP", alpha, beta_w, gamma)

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
