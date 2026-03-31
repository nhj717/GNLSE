from functions import read_hdf5 as rdhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
p_tag = "d_ratio"
(p1_value, p2_value, p3_value, p4_value, p5_value, p6_value, p7_value, p8_value) = (
    "0.4_rcp",
    "0.5_rcp",
    "0.6_rcp",
    "0.7_rcp",
    "0.4_lcp",
    "0.5_lcp",
    "0.6_lcp",
    "0.7_lcp",
)
waveguidenames = [
    f"{p_tag}_{p1_value}",
    f"{p_tag}_{p2_value}",
    f"{p_tag}_{p3_value}",
    f"{p_tag}_{p4_value}",
    f"{p_tag}_{p5_value}",
    f"{p_tag}_{p6_value}",
    f"{p_tag}_{p7_value}",
    f"{p_tag}_{p8_value}",
]

# waveguidenames = [
#     "20230907_SEM_rcp",
#     "20230907_SEM_lcp",
#     "20230907_SEM_straight_rcp",
#     "20230907_SEM_straight_lcp",
# ]
wl_um, beta = [], []
for name in waveguidenames:
    data_label, data = rdhd(os.path.join(folder_path, "twisted_waveguide.h5"), name)
    wl_um.append(data[data_label.index("wl_um")])
    beta.append(data[data_label.index("beta0")])
CB = np.real(beta[4] - beta[0]) * wl_um[0] / (2 * np.pi)
CB2 = np.real(beta[5] - beta[1]) * wl_um[0] / (2 * np.pi)
CB3 = np.real(beta[6] - beta[2]) * wl_um[0] / (2 * np.pi)
CB4 = np.real(beta[7] - beta[3]) * wl_um[0] / (2 * np.pi)
# CB5 = np.real(beta[9] - beta[4]) * wl_um[0] / (2 * np.pi)

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "CB [uRIU]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_um[0], CB, c="black", label="0.4")
pcm = ax.plot(wl_um[0], CB2, c="red", label="0.5")
pcm = ax.plot(wl_um[0], CB3, label="0.6")
pcm = ax.plot(wl_um[0], CB4, label="0.7")
# pcm = ax.plot(wl_um[0], CB5, label="680nm")
ax.grid()
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis="both", which="major", size=4, width=2, labelsize=10)
# ax.set_xticks([0.5, 1.0, 1.5, 2.0])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)
# ax.set_xlim(0.85, 1.15)
# ax.set_ylim(-0.1, 0.5)
ax.legend()
plt.show(block=True)
