from propagate import fiber_propagation
import numpy as np
from scipy.fft import fftfreq
from scipy.constants import c, pi
from datetime import datetime

# Initial Pulse information
pulse_shape = "gaussian"
lambda0_um = 0.920  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 150e-15
if pulse_shape == "gaussian":
    T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
elif pulse_shape == "sech":
    T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # pulse duration in seconds
P0 = 1e5  # peak power in W
C = 0  # chirp

# Grid information
z_tot = 0.01  # Fiber length in m
z_steps = 2**10
dz = z_tot / z_steps
z = np.arange(0, z_steps) * dz  # z grid for simulation

T_span = 200 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, dt)  # freq grid for simulation
omega = 2 * np.pi * f

# Fiber informaiton
alpha = None  # loss of the fiber
waveguide_name = "20240422_3B_ideal"  # name of the waveguide
n2 = 2.6e-20  # nonlinear index in m^2/W
gamma = 2 * pi * n2 / lambda0
fr = 0.0
self_steepening = False

# simulation_type = "RK4IP"
simulation_type = "SSFM_Vishal"
###        RUN SIMULATION    ###
A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega, pulse_shape, P0, T0, C)
sim.source()
sim.propagate(simulation_type, alpha, waveguide_name, gamma, fr, self_steepening)
B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw_z(wl_range=[735, 1095])
