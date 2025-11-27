import numpy as np
import scipy.fft as fft
from raman_response import hollenbeck_cantrell_hr as R_t

# from raman_response import single_damped_HO as R_t0
import matplotlib.pyplot as plt

T_FWHM = 50e-15
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))

Tspan = 200 * T0  # total simulation grid for tau
t_steps = 2**13  # No. of tau steps
delt = Tspan / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * delt  # tau grid for simulations
hR_t = R_t(t)  # Raman response

plt.plot(t, hR_t)
plt.show(block=True)

f = fft.fftshift(fft.fftfreq(t_steps, delt))
HR_f = fft.fftshift(fft.fft(hR_t))
plt.plot(f, np.imag(HR_f))
plt.show(block=True)
