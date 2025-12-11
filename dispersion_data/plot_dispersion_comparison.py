from functions import read_hdf5 as rdhd
from numpy import flip
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
materialname = "fused_silica"
waveguidename1 = "20240422_3B_ideal"
waveguidename2 = "20240422_3B_SEM"

data_label_w1, data_w1 = rdhd(os.path.join(folder_path, "waveguide.h5"), waveguidename1)
data_label_w2, data_w2 = rdhd(os.path.join(folder_path, "waveguide.h5"), waveguidename2)
print(data_label_w1)
print(data_label_w2)

wl_w1 = data_w1[data_label_w1.index("wl_um")]
D_w1 = data_w1[data_label_w1.index("D")]
wl_w2 = data_w2[data_label_w2.index("wl_um")]
D_w2 = data_w2[data_label_w2.index("D")]

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "D [ps/(nm km)]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_w1, D_w1, color="black", label="Ideal Dispersion")
pcm2 = ax.plot(wl_w2, D_w2, color="blue", label="SEM Dispersion fit")

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
