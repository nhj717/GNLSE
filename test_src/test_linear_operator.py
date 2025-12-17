from scipy.constants import pi,c
import numpy as np
from scipy.fft import fftfreq,ifftshift
import matplotlib.pyplot as plt
import os
from functions import read_hdf5 as rdhd
from scipy.interpolate import UnivariateSpline

lambda0_um = 0.92  # pump wavelength in um
lambda0 = lambda0_um * 1e-6
omega0 = 2 * pi * c / lambda0
T_FWHM = 140e-15
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
T_span = 150 * T0
t_steps = 2**12
dt = T_span / t_steps
t = np.arange(-t_steps / 2, t_steps / 2) * dt  # tau grid for simulations
f = fftfreq(t_steps, dt)  # freq grid for simulation
omega = 2 * np.pi * f

folder_path = os.getcwd()
location = os.path.join(folder_path, "dispersion_data", "waveguide.h5")
data_label, data = rdhd(location, "20240422_3B_ideal" , read=False)

omega_data = data[data_label.index("omega")]
D = data[data_label.index("D")]
D_spl = UnivariateSpline(omega_data, D, k=5)
omega_true = omega+omega0
D_w = ifftshift(D_spl(omega_true))

wl_w = ifftshift(2*pi*c/omega_true)




# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "D [ps/(nm km)]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_w, D_w, color="black", label="Total Dispersion")
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
# ax.set_xlim(0.8,1.2)
# ax.set_ylim(-500, 0)
ax.legend()
plt.show(block=True)