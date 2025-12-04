from numpy import arange
import material_RI
from dispersion_generator import material_dispersion
from functions import save_dict_to_hdf5 as svhd
import os

folder_path = os.path.dirname(os.path.abspath(__file__))
hdf5_name = "material.h5"
file_name = os.path.join(folder_path, hdf5_name)
material = "fused_silica"
wl_um = arange(0.5, 2.0, 0.01)
arg_dict = {"wl": wl_um}
arg_dict.update(material_RI.fused_silica(wl_um))
arg_dict.update(material_dispersion(wl_um, arg_dict["n"]))
svhd(file_name, material, arg_dict)
