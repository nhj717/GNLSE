from functions import read_hdf5 as rdhd
import matplotlib.pyplot as plt
import os
import numpy as np

folder_path = os.path.dirname(os.path.abspath(__file__))
file = "5ring_ratio_cmap_half_period"
p_tag = "d_ratio"
p_value = "0.5"
waveguidename = f"{p_tag}_{p_value}.h5"

wl_um, D = [], []
data_label, data = rdhd(os.path.join(folder_path, file, waveguidename))

x = data[data_label.index("freq")][0]
y = data[data_label.index("mode_num")][1]

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
label_list = ["Freq", "neff_re"]  # xlabel and ylabel
(width, height, fsize) = size_parameter
fig, ax = plt.subplots(figsize=(width * mm, height * mm))
pcm = ax.plot(x, y, c="black")

# ax.set_xlabel(label_list[0], fontsize=fsize)
# ax.set_ylabel(label_list[1], fontsize=fsize)
# ax.set_xlim(0.5, 2)
# ax.set_ylim(-200, 200)
# fig.tight_layout()
plt.show(block=True)
