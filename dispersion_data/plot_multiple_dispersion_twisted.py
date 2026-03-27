from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))

# p_tag = "cap_t"
# p1_value, p2_value, p3_value, p4_value, p5_value = (
#     "600E-9_rcp",
#     "620E-9_rcp",
#     "640E-9_rcp",
#     "660E-9_rcp",
#     "680E-9_rcp",
# )
# waveguidenames = [
#     f"{p_tag}_{p1_value}",
#     f"{p_tag}_{p2_value}",
#     f"{p_tag}_{p3_value}",
#     f"{p_tag}_{p4_value}",
#     f"{p_tag}_{p5_value}",
# ]

waveguidenames = [
    "20230907_rcp",
    "20230907_SEM_rcp",
    "20230907_SEM_straight_rcp",
]

wl_um, D = [], []
for name in waveguidenames:
    data_label, data = rdhd(os.path.join(folder_path, "twisted_waveguide.h5"), name)
    wl_um.append(data[data_label.index("wl_um")])
    D.append(data[data_label.index("D")])

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "D [ps/(nm km)]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_um[0], D[0], label=f"{waveguidenames[0]}")
pcm = ax.plot(wl_um[1], D[1], label=f"{waveguidenames[1]}")
pcm = ax.plot(wl_um[2], D[2], label=f"{waveguidenames[1]}")
# pcm = ax.plot(wl_um[3], D[3], label=f"{p_tag} = {p4_value}")
# pcm = ax.plot(wl_um[4], D[4], label=f"{p_tag} = {p5_value}")
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
# ax.set_xlim(0.5, 2)
ax.set_ylim(-200, 200)
ax.legend()
plt.show(block=True)
