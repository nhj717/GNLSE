import pandas as pd
import numpy as np
from scipy.constants import c, pi
from dispersion_generator import twisted_mode_dispersion
from functions import save_dict_to_hdf5 as svhd
import os

# Path to your COMSOL text file
project_path = os.getcwd()
comsol_folder = "comsol_dispersion/5ring_ratio"
waveguidename = "d_ratio_0.4_lcp"
filename = os.path.join(project_path, comsol_folder, waveguidename) + ".txt"

# Read the file, no lines skipped
header_row = pd.read_csv(filename, sep=r"\s+", nrows=1, header=None, engine="python")
# Convert row to a flat list
header = header_row.values.flatten().tolist()

df = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=header, engine="python")
columns_as_lists = {col: df[col].tolist() for col in df.columns}

print(columns_as_lists.keys())  # show the column names

twist_period = 0.01  # in meters
alpha = 2 * pi / twist_period
j = 1  # rcp is -1 in this case

overlap = np.array(df["overlap"])
freq = np.array(df["freq"])
omega = 2 * pi * freq
neff = np.array(df["neff"])
loss = np.array(df["loss"])
Aeff = np.array(df["A_eff"])
neff = [complex(s.replace("i", "j")) for s in neff]

# ********frequency must be in the ascending order********
arg_dict = {
    "freq": freq,
    "omega": omega,
    "wl_um": c / freq * 1e6,
    "n_eff": neff,
    "loss": loss,
    # "A_eff": Aeff,
}
arg_dict.update(
    twisted_mode_dispersion(
        arg_dict["omega"], arg_dict["wl_um"], np.real(arg_dict["n_eff"]), alpha, j, 0
    )
)

folder_path = os.path.dirname(os.path.abspath(__file__))
hdf5_name = "twisted_waveguide.h5"
svhd(os.path.join(folder_path, hdf5_name), waveguidename, arg_dict)
