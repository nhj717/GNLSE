from RK4IP import fiber_propagation
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

E_pulse = 1e-3  # pulse energy in mJ
P0 = 1e3  # peak power in W
lamb0 = 0.920
T_FWHM = 10e-15
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
P0 = E_pulse / T_FWHM

A = datetime.now()

sim = fiber_propagation(lamb0, P0, T0)
sim.source("gaussian")

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

f = fft.fftshift(sim.f_D)
t = sim.t
I = P0 * abs(sim.E[:, 0]) ** 2
spec = abs(fft.fftshift(sim.spectrum[:, 0]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("simulation")

ax1.plot(t * 1e15, I * 1e-9)
ax1.set_title("Time domain")
ax1.set_xlim(-5 * T0 * 1e15, 5 * T0 * 1e15)
ax1.set_xlabel("t [fs]")
ax1.set_ylabel("Power [GW]")

ax2.plot(f * 1e-15, spec)
ax2.set_title("Freq. domain")
ax2.set_xlabel("f [THz]")
plt.show(block=True)
