from propagate import fiber_propagation
import numpy as np
from scipy.fft import fftfreq
from scipy.constants import c, pi
from datetime import datetime

# Initial Pulse information
pulse_shape = "gaussian"
lambda0_um = 0.850  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 50e-15
if pulse_shape == "gaussian":
    T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
elif pulse_shape == "sech":
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

Np = 2**7
# P_max = 1e3
Pavg = 1.2  # Watts
R_R = 80e6
P_max = Pavg / (T0 * R_R)
delP = P0 / Np
P = np.arange(1, Np + 1) * delP


# Fiber informaiton
alpha = None  # loss of the fiber; type None to turn off alpha
waveguide_name = "20230330_4_ideal"  # name of the waveguide
n2 = 2.7e-20  # nonlinear index in m^2/W
gamma = 2 * pi * n2 / lambda0
fr = 0.18
self_steepening = True  # True or False to include self steepening

simulation_type = "SSFM_Vishal"
###        RUN SIMULATION    ###
A = datetime.now()
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega, pulse_shape, P0, T0, C)
sim.propagate_P(simulation_type, alpha, waveguide_name, gamma, fr, self_steepening, P)
B = datetime.now()
print("time : for loop", (B - A).total_seconds())
sim.draw_P()
