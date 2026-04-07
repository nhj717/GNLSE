from functions import read_hdf5 as rdhd
from functions import mode_overlap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.constants import c
import numpy as np
import os
import glob

folder_path = os.path.dirname(os.path.abspath(__file__))
file = "5ring_ratio_cmap_onehalf_period"
data_key = "neff_re"  # define the result of interest
mode_number = [
    0,
    1,
]  # take the mode number from the smallest parameter from data_key lcp and rcp
twist_period = 1.5e-2
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

p1 = np.array(p1_list)
mode_number_array = np.zeros_like([p1, p1], dtype="int")
idx0 = np.argmin(np.argsort(p1))

for i in mode_number:
    E_ref = np.array(E_re_list[idx0][i, :]) + 1j * np.array(E_im_list[idx0][i, :])
    for h5_i in range(np.size(E_re_list, axis=0)):
        if h5_i == idx0:
            mode_number_array[i, h5_i] = i
        else:
            ov = 0
            for mode_number_i in range(np.size(E_re_list, axis=1)):
                E_i = np.array(E_re_list[h5_i][mode_number_i, :]) + 1j * np.array(
                    E_im_list[h5_i][mode_number_i, :]
                )
                ov_i = mode_overlap(E_i, E_ref, w_list[h5_i])
                if ov < ov_i:
                    mode_number_array[i, h5_i] = mode_number_i
                    ov = ov_i


# Convert combined list of data into a numpy array
p1 = np.array(p1_list)
f_array = np.array(f_list)
wl_array = c / f_array

data_array_full = np.array(data_list)
row_indices = np.arange(data_array_full.shape[0])
rcp_array = data_array_full[row_indices, mode_number_array[0], :]
lcp_array = data_array_full[row_indices, mode_number_array[1], :]
cb_array = (lcp_array - rcp_array) + 2 * wl_array / twist_period
# sort by p1 (row axis)
idx = np.argsort(p1)
p1 = p1[idx]
wl_array = wl_array[idx, :] * 1e6
cb_array = cb_array[idx, :] * 1e6
p1_array = np.repeat(p1[:, None], wl_array.shape[1], axis=1)

# smoothen data if desired
# data_smooth = gaussian_filter(data_array, sigma=(1.5, 1.5))  # (p1, freq) sigmas

# plot pcolor
fig, ax = plt.subplots(figsize=(6, 5))
pc = ax.pcolormesh(wl_array, p1_array, cb_array, cmap="seismic")
pc.set_clim(-2, 2)
ax.set_box_aspect(1)
ax.set_xlabel("Wavelength [um]")
ax.set_ylabel("d_ratio")
fig.colorbar(pc, label="CB [uRIU]")
plt.show(block=True)
