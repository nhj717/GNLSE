from functions import read_hdf5 as rdhd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import os
import glob

folder_path = os.path.dirname(os.path.abspath(__file__))
file = "cmap_test"
data_key = "loss"  # define the result of interest
mode_number = 0
h5_files = glob.glob("{}/{}/*.h5".format(folder_path, file))
f_list = []
p1_list = []
data_list = []

for h5_file in h5_files:
    data_label, data = rdhd(os.path.join(folder_path, file, h5_file), None, False)
    f_list.append(data[data_label.index("freq")][mode_number])
    p1_list.append(data[data_label.index("p1")][mode_number])
    data_list.append(data[data_label.index(data_key)][mode_number])

# Convert combined list of data into a numpy array
f_array = np.array(f_list)
p1 = np.array(p1_list)
data_array = np.array(data_list)

# sort by p1 (row axis)
idx = np.argsort(p1)
p1 = p1[idx]
f_array = f_array[idx, :]
data_array = data_array[idx, :]
p1_array = np.repeat(p1[:, None], f_array.shape[1], axis=1)

# smoothen data if desired
data_smooth = gaussian_filter(data_array, sigma=(1.0, 0.5))  # (p1, freq) sigmas

# plot pcolor
fig, ax = plt.subplots(figsize=(6, 5))
# pc = ax.pcolormesh(f_array, p1_array, data_smooth, cmap="hot", shading="auto")
pc = ax.imshow(
    data_array, aspect="auto", origin="lower", interpolation="bicubic", cmap="hot"
)
ax.set_box_aspect(1)
plt.colorbar(pc, label="Loss [dB/m]")
ax.set_xlabel("freq [Hz]")
ax.set_ylabel("d_ratio")
plt.show(block=True)
