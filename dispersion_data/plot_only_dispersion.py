from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
waveguidename = "5ring_twisted_0.8"
data_label, data = rdhd(
    os.path.join(folder_path, "twisted_waveguide.h5"), waveguidename
)
print(data_label)

j = -1
twist_period = 0.01
alpha = 2 * pi / twist_period
w = data[data_label.index("omega")]
wl_um = data[data_label.index("wl_um")]
wl = wl_um * 1e-6
D = data[data_label.index("D")]
neff = data[data_label.index("n_eff")]
neff_spl = UnivariateSpline(w, neff, k=5, s=1).derivative(n=2)(w)
beta0 = np.real(neff) * 2 * pi / wl + j * alpha
beta_spl = UnivariateSpline(w, beta0, k=5, s=1e7)
beta1 = beta_spl.derivative(n=1)
beta1_w = beta1(w)
beta2 = beta_spl.derivative(n=2)
beta2_w = beta2(w)
# beta2 = beta_spl.derivative(nu=2)
# beta2_w = beta2(w)
# D = -2 * pi * c / wl**2 * beta2_w * 1e6

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "D [ps/(nm km)]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
# pcm = ax.plot(wl_um, D, color="black", marker="o", markersize=5)
# pcm = ax.plot(w, neff_spl, color="red")
pcm = ax.plot(wl_um, D, label="Total Dispersion")
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
# ax.set_xlim(0.5, 2)
# ax.set_ylim(-5, 5)
# ax.legend()
plt.show(block=True)
