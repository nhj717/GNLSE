from propagate import fiber_propagation
import numpy as np
from scipy.fft import fftfreq
from scipy.constants import c, pi
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
z_steps = 2**10
dz = z_tot / z_steps
z = np.arange(0, z_steps) * dz  # z grid for simulation

T_span = 200 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, dt)  # freq grid for simulation
omega = 2 * np.pi * f

# Fiber information
alpha = 0    # loss of the fiber
beta = [
    -1.276e-26,
    8.119e-41,
    -1.321e-55,
    3.032e-70,
    -4.196e-85,
    2.57e-100,
]  # propagation constants in the unit of s/m, s^2/m, ...
gamma = 0.045  # nonlinear coeff from the fiber in W^-1/m
fr = 0.18
self_steepening = 1

# simulation_type = "RK4IP"
simulation_type = "RK4IP"
###        RUN SIMULATION    ###
A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega)
sim.source(shape, P0, T0, C)
sim.propagate(simulation_type, alpha, beta, gamma, fr, self_steepening)
B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw_z()
