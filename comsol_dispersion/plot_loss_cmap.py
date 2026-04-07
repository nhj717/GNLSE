from functions import read_hdf5 as rdhd
from functions import mode_overlap
import matplotlib.pyplot as plt
from scipy.constants import c
import numpy as np
import os
import glob

folder_path = os.path.dirname(os.path.abspath(__file__))
file = "5ring_ratio_cmap"
data_key = "loss"  # define the result of interest
mode_number = 0  # take the mode number from the smallest parameter from data_key
h5_files = glob.glob("{}/{}/*.h5".format(folder_path, file))
f_list = []
p1_list = []
data_list = []
w_list = []
E_re_list = []
E_im_list = []

# take the first freq beam profile of each parameter and perform mode overlap, make a list of mode numbers and only append the particular mode
for h5_file in h5_files:
    data_label, data = rdhd(os.path.join(folder_path, file, h5_file), None, False)
    p1_list.append(data[data_label.index("p1")][0])
    w_list.append(data[data_label.index("w")][0, :])
    E_re_list.append(data[data_label.index("E_re")][:, 0, :])
    E_im_list.append(data[data_label.index("E_im")][:, 0, :])
    f_list.append(data[data_label.index("freq")][0])
    data_list.append(data[data_label.index(data_key)])

p1 = np.array(p1_list)
mode_number_array = np.zeros_like(p1, dtype="int")
overlap_array = np.zeros_like(p1)
f_array = np.array(f_list)
data_array_full = np.array(data_list)
row_indices = np.arange(data_array_full.shape[0])
E_re_array = np.array(E_re_list)
E_im_array = np.array(E_im_list)
w_array = np.array(w_list)

# sort by p1 (row axis)
idx = np.argsort(p1)
p1 = p1[idx]
p1_array = np.repeat(p1[:, None], f_array.shape[1], axis=1)
f_array = f_array[idx, :]
wl_array = c / f_array * 1e6
data_array_full = data_array_full[idx, :]
E_re_array = E_re_array[idx, :]
E_im_array = E_im_array[idx, :]
w_array = w_array[idx, :]


E_ref = np.array(E_re_array[0, mode_number, :]) + 1j * np.array(
    E_im_array[0, mode_number, :]
)

for h5_i in range(np.size(E_re_list, axis=0)):
    if h5_i == 0:
        mode_number_array[h5_i] = mode_number
        overlap_array[h5_i] = 1
    else:
        ov = 0
        for mode_number_i in range(np.size(E_re_list, axis=1)):
            E_i = (
                E_re_array[h5_i, mode_number_i, :]
                + 1j * E_im_array[h5_i, mode_number_i, :]
            )
            ov_i = mode_overlap(E_i, E_ref, w_array[h5_i, :])
            print(ov_i)
            if ov < ov_i:
                mode_number_array[h5_i] = mode_number_i
                overlap_array[h5_i] = ov_i
                ov = ov_i

# Convert combined list of data into a numpy array
row_indices = np.arange(data_array_full.shape[0])
data_array = data_array_full[row_indices, mode_number_array, :]


# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
label_list = ["p1", "mode_number"]  # xlabel and ylabel
(width, height, fsize) = size_parameter
fig, ax = plt.subplots(figsize=(width * mm, height * mm))
pcm = ax.plot(p1, mode_number_array, c="black")
plt.show(block=True)

# plot
mm = 1 / 25.4
size_parameter = (150, 120, 12)  # width, height, font
label_list = ["p1", "Overlap"]  # xlabel and ylabel
(width, height, fsize) = size_parameter
fig, ax = plt.subplots(figsize=(width * mm, height * mm))
pcm = ax.plot(p1, 100 * overlap_array, c="black")
plt.show(block=True)

# smoothen data if desired
# data_smooth = gaussian_filter(data_array, sigma=(1.5, 1.5))  # (p1, freq) sigmas

# plot pcolor
fig, ax = plt.subplots(figsize=(6, 5))
pc = ax.pcolormesh(wl_array, p1_array, data_array, cmap="hot")
# pc.set_clim(0, 10)
ax.set_box_aspect(1)
ax.set_xlabel("Wavelength [um]")
ax.set_ylabel("d_ratio")
fig.colorbar(pc, label="loss [dB/m]")
plt.show(block=True)
