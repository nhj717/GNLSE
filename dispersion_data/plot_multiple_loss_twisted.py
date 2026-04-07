from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
# p_tag = "d_ratio"
# p1_value, p2_value, p3_value, p4_value = (
#     "20230907_rcp",
#     "0.5_rcp",
#     "0.6_rcp",
#     "0.7_rcp",
# )
# waveguidenames = [
#     f"{p_tag}_{p1_value}",
#     f"{p_tag}_{p2_value}",
#     f"{p_tag}_{p3_value}",
#     f"{p_tag}_{p4_value}",
# ]
waveguidenames = [
    "20230907_lcp",
    "20230907_rcp",
    "20230907_SEM_lcp",
    "20230907_SEM_rcp",
    "20230907_SEM_straight_lcp",
    "20230907_SEM_straight_rcp",
]

wl_um, loss = [], []
for name in waveguidenames:
    data_label, data = rdhd(os.path.join(folder_path, "twisted_waveguide.h5"), name)
    wl_um.append(data[data_label.index("wl_um")])
    w = data[data_label.index("omega")]
    neff = data[data_label.index("n_eff")]
    k = -np.imag(neff)
    loss.append(data[data_label.index("loss")])


# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "Loss [dB/m]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_um[0], loss[0], label=f"{waveguidenames[0]}")
pcm = ax.plot(wl_um[1], loss[1], label=f"{waveguidenames[1]}")
pcm = ax.plot(wl_um[2], loss[2], label=f"{waveguidenames[2]}")
pcm = ax.plot(wl_um[3], loss[3], label=f"{waveguidenames[3]}")
ax.grid()
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([0.5, 1.0, 1.5, 2.0])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
ax.set_xlim(0.5, 2.0)
# ax.set_ylim(-1, 150)
ax.legend()
plt.show(block=True)
