import numpy as np
from raman_response import hollenbeck_cantrell_hr as R_t
from raman_response import single_damped_HO as R_t0
import matplotlib.pyplot as plt

T_FWHM = 50e-15
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))

Tspan = 200 * T0  # total simulation grid for tau
t_steps = 2 ** 13  # No. of tau steps
delt = Tspan / t_steps
t = (
        np.arange(-t_steps / 2, t_steps / 2) * delt
)  # tau grid for simulations
hR_t = R_t0(t)  # Raman response

plt.plot(t, hR_t)
plt.show(block=True)