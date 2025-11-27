from functions import read_hdf5 as rdhd
from numpy import flip
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
materialname = "fused_silica"
waveguidename = "20240422_3B_SEM_reduced"
data_label_m, data_m = rdhd(os.path.join(folder_path, "material.h5"), materialname)
data_label_w, data_w = rdhd(os.path.join(folder_path, "waveguide.h5"), waveguidename)
print(data_label_m)
print(data_label_w)

wl_m = data_m[data_label_m.index("wl")]
D_m = data_m[data_label_m.index("D")]
n = data_m[data_label_m.index("n")]

wl_w = data_w[data_label_w.index("wl_um")]
D_w = data_w[data_label_w.index("D")]
neff = data_w[data_label_w.index("n_eff")]
D_w_spl = UnivariateSpline(flip(wl_w), flip(D_w), k=5)
D_w_fit = D_w_spl(wl_m)

D_waveguide = D_w_fit - D_m

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ["wavelength [um]", "D [ps/(nm km)]"]  # xlabel and ylabel
(width, height, fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect=layout)
pcm = ax.plot(wl_w, D_w, color="black", label="Total Dispersion", linestyle=":")
pcm2 = ax.plot(wl_m, D_w_fit, color="blue", label="Total Dispersion fit")
pcm3 = ax.plot(wl_m, D_m, color="red", label="material")
pcm4 = ax.plot(wl_m, D_waveguide, color="green", label="waveguide")
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
