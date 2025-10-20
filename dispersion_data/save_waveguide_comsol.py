import pandas as pd
import numpy as np
from scipy.constants import c
from disp_gen import mode_dispersion
from functions import save_dict_to_hdf5 as svhd
import os

# Path to your COMSOL text file
comsol_folder = r"C:\Users\labadmin\PycharmProjects\GNLSE\comsol_dispersion"
waveguidename = "pcf"
filename = os.path.join(comsol_folder, waveguidename) + ".txt"

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
neff = np.array(df["neff"])[mode_overlap == 1]
neff = [complex(s.replace("i", "j")) for s in neff]

arg_dict = {"freq": freq, "wl": c / freq * 1e6, "n_eff": neff}
arg_dict.update(mode_dispersion(arg_dict["wl"], np.real(arg_dict["n_eff"])))

hdf5_name = "waveguide.h5"
svhd(hdf5_name, waveguidename, arg_dict)
