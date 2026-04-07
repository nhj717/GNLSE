from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
waveguidename = "20230907"
data_label, data = rdhd(
    os.path.join(folder_path, "twisted_waveguide.h5"), waveguidename
)
print(data_label)


w = data[data_label.index("omega")]
print(np.size(w))
wl_um = data[data_label.index("wl_um")]
wl = wl_um * 1e-6
neff = data[data_label.index("n_eff")]
k = -np.imag(neff)
alpha = 20 * w * k / c / np.log(10)

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "Loss [dB/m]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_um, alpha, color="black")
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
ax.set_xlim(0.5, 2)
ax.set_ylim(0, 50)
# ax.legend()
plt.show(block=True)
