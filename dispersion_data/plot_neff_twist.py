from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
p_tag = "cap_t"
p1_value, p2_value = (
    "640E-9",
    "640E-9_lcp",
)
waveguidenames = [
    f"{p_tag}_{p1_value}",
    f"{p_tag}_{p2_value}",
]
wl_um, beta, neff = [], [], []
for name in waveguidenames:
    data_label, data = rdhd(os.path.join(folder_path, "twisted_waveguide.h5"), name)
    wl_um.append(data[data_label.index("wl_um")])
    beta.append(data[data_label.index("beta0")])
    neff.append(data[data_label.index("n_eff")])

CB = np.real(beta[1] - beta[0])

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "Effective index"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_um[0], neff[0], label="RCP")
pcm = ax.plot(wl_um[1], neff[1], label="LCP")

ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([0.5, 1.0, 1.5, 2.0])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
ax.set_xlim(0.73, 1.27)
ax.set_ylim(0.9995, 1.0001)
ax.legend()
plt.show(block=True)
