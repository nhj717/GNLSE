from functions import read_hdf5 as rdhd
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
file = "cmap_test"
p_tag = "d_ratio"
p_value = "1.2"
waveguidename = f"{p_tag}_{p_value}.h5"

data_label, data = rdhd(os.path.join(folder_path, file, waveguidename))

mode_number = 2
freq_number = 0
x = data[data_label.index("x")][0]
y = data[data_label.index("y")][0]
E_im = data[data_label.index("E_im")][mode_number, freq_number]
E_re = data[data_label.index("E_re")][mode_number, freq_number]
I = np.zeros_like(x)
for i in range(3):
    I += E_re[i] ** 2 + E_im[i] ** 2
I = I / np.max(I)

###Plot setting should be the same from ModeTrack.m in cluster
D = 10  # in microns
Nr = 80
Ntheta = 120

# reshape to (Ntheta, Nr) consistent with MATLAB polarGrid
V = I.reshape(Ntheta, Nr)
X = x.reshape(Ntheta, Nr)
Y = y.reshape(Ntheta, Nr)

Vpt = V.reshape(-1)
tri = mtri.Triangulation(x * 1e6, y * 1e6)

fig, ax = plt.subplots(figsize=(6, 5))
circ = Circle((0.0, 0.0), D / 2, fill=False, edgecolor="w", linewidth=4, linestyle=":")
ax.add_patch(circ)
pc = ax.tripcolor(tri, Vpt, shading="gouraud", cmap="hot")
ax.set_aspect("equal", "box")
ax.set_facecolor("k")  # empty regions inside axes -> black
plt.colorbar(pc, label="I/I_max")
ax.set_xlim(-3 / 4 * D, 3 / 4 * D)
ax.set_ylim(-3 / 4 * D, 3 / 4 * D)
ax.set_xlabel("x [um]")
ax.set_ylabel("y [um]")
plt.show(block=True)
