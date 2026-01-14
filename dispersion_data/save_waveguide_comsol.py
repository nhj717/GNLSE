import pandas as pd
import numpy as np
from scipy.constants import c, pi
from dispersion_generator import mode_dispersion
from functions import save_dict_to_hdf5 as svhd
import os

# Path to your COMSOL text file
project_path = os.getcwd()
comsol_folder = "comsol_dispersion"
waveguidename = "20240422_3B_ideal"
filename = os.path.join(project_path, comsol_folder, waveguidename) + ".txt"

# Read the file, skipping the first 4 lines (metadata)
header_row = pd.read_csv(
    filename, sep=r"\s+", skiprows=4, nrows=1, header=None, engine="python"
)
# Convert row to a flat list
header = header_row.values.flatten().tolist()

# Remove the leading '%'
if header[0] == "%":
    header = header[1:]

df = pd.read_csv(filename, sep=r"\s+", skiprows=5, names=header, engine="python")
columns_as_lists = {col: df[col].tolist() for col in df.columns}

print(columns_as_lists.keys())  # show the column names

mode_overlap = np.array(df["mode_overlap"])
freq = np.array(df["freq"])[mode_overlap == 1]
omega = 2 * pi * freq
neff = np.array(df["neff"])[mode_overlap == 1]
Aeff = np.array(df["A_eff"])[mode_overlap == 1]
neff = [complex(s.replace("i", "j")) for s in neff]

# ********frequency must be in the ascending order********
arg_dict = {
    "freq": freq,
    "omega": omega,
    "wl_um": c / freq * 1e6,
    "n_eff": neff,
    "A_eff": Aeff,
}
arg_dict.update(
    mode_dispersion(arg_dict["omega"], arg_dict["wl_um"], np.real(arg_dict["n_eff"]))
)

folder_path = os.path.dirname(os.path.abspath(__file__))
hdf5_name = "waveguide.h5"
svhd(os.path.join(folder_path, hdf5_name), waveguidename, arg_dict)
