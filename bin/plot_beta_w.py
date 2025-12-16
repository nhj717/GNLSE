from operators import linear_operator as L_op
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

# Grid information
T_span = 200 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, dt)  # freq grid for simulation
omega = 2 * np.pi * f

# Fiber informaiton
alpha = 0  # loss of the fiber
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

L, Aeff = L_op(alpha, beta, omega0, omega)
omega = ifftshift(omega)
beta_w = ifftshift(np.imag(L))
# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["Angular Freq.[rad Hz]", "Beta2-Beta0-Beta1*omega"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
ax.plot(omega, beta_w, color="black")
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
# ax.set_xlim(0.8,1.2)
# ax.set_ylim(-500, 0)
plt.show(block=True)
