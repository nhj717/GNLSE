from RK4IP_z import fiber_propagation
from scipy import fft
from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

E_pulse = 1e-3  # pulse energy in mJ
P0 = 10e3  # peak power in W
lamb0 = 850e-9
T_FWHM = 50e-15
# T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # for Sech Pulse
# P0 = E_pulse / T_FWHM

A = datetime.now()

sim = fiber_propagation(lamb0, P0, T0)
sim.source("sech")

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

# ---- Retrieve data from the sim ----
freq = sim.f
I = abs(sim.E[:, 0]) ** 2
I /= I.max()
spec = abs(sim.spectrum[:, 0]) ** 2
spec /= spec.max()

# ---- Convert to wavelength axis ----
mask_pos = freq > 0
freq_pos = freq[mask_pos]
lambda_axis = c / freq_pos
jacobian = c / (lambda_axis**2)
S_num_lambda = spec[mask_pos] * jacobian

# ---- Analytic spectrum ----
omega_axis = 2 * np.pi * freq_pos
spec_ana_env = 1 / np.cosh((np.pi * T0 * (omega_axis - sim.omega0)) / 2) ** 2
spec_ana_env /= spec_ana_env.max()
S_ana_lambda = spec_ana_env * jacobian

# ---- Small floor to avoid log(0)
eps = np.finfo(float).eps

# ---- Convert to dB safely
S_num_dB = 10 * np.log10((S_num_lambda + eps) / np.max(S_num_lambda + eps))
S_ana_dB = 10 * np.log10((S_ana_lambda + eps) / np.max(S_ana_lambda + eps))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("simulation")

ax1.plot(sim.t * 1e15, I * 1e-9)
ax1.set_title("Time domain")
ax1.set_xlim(-50 * T0 * 1e15, 50 * T0 * 1e15)
ax1.set_xlabel("t [fs]")
ax1.set_ylabel("Power [GW]")

ax2.plot(lambda_axis * 1e9, S_num_dB, label="Numeric FFT")
ax2.plot(lambda_axis * 1e9, S_ana_dB, "--", label="Analytic Sech")
ax2.set_title("Freq. domain")
ax2.set_xlim(500, 1500)
# ax2.set_ylim(-80, 20)
ax2.set_xlabel("Wavelength [nm]")
ax2.legend()
plt.show(block=True)
