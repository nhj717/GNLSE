from propagate_z import fiber_propagation
import numpy as np
from scipy.fft import fftfreq, ifftshift
from scipy.constants import c, pi
import matplotlib.pyplot as plt

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
elif shape == "lorentzian":
    T0 = T_FWHM

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


###        Set Parameters in class     ###
sim = fiber_propagation(omega0, dz, z, dt, t, f, omega)
sim.source(shape, P0, T0, C)


###   Plot   ###
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"Initial Pulse Plot")
ax1.plot(t * 1e12, 1e-9 * abs(sim.E[:, 0]) ** 2, label="before")
ax1.set_xlim(-20 * T0 * 1e12, 20 * T0 * 1e12)
ax1.set_xlabel("Time [ps]")
ax1.set_ylabel("Intensity")
ax1.legend()

f = (omega + omega0) / 2 / pi
f = ifftshift(f)
mask_pos = f > 0
f = f[mask_pos]
wavelength = c / f * 1e9
S = abs(sim.spectrum[:, 0]) ** 2
spectrum = 10 * np.log10((S + np.finfo(float).eps) / np.max(S))
spectrum = ifftshift(spectrum)
spectrum = spectrum[mask_pos]
ax2.plot(
    wavelength,
    spectrum,
    label="before",
)
ax2.set_xlim(500, 1400)
ax2.set_xlabel("Wavelength [nm]")
ax2.set_ylim(-200, 10)
ax2.legend()

plt.show(block=True)
