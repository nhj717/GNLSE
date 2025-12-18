from functions import read_hdf5 as rdhd
from numpy import flip
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
waveguidename = "20240422_3B_ideal"
data_label_w, data_w = rdhd(os.path.join(folder_path, "waveguide.h5"), waveguidename)
print(data_label_w)

wl_w = data_w[data_label_w.index("wl_um")]
Aeff = data_w[data_label_w.index("A_eff")]


# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "Effective Area [m^2]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_w, Aeff, color="black", label="A_eff")
ax.set_xlabel(label_list[0], fontsize=fsize)
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
