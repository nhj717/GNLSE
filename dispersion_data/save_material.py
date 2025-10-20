from numpy import arange
import material_RI
from disp_gen import material_dispersion
from functions import save_dict_to_hdf5 as svhd

file_name = "material.h5"
material = "fused_silica"
wl_um = arange(0.5, 2.0, 0.01)
arg_dict = {"wl": wl_um}
arg_dict.update(material_RI.fused_silica(wl_um))
arg_dict.update(material_dispersion(wl_um, arg_dict["n"]))
svhd(file_name, material, arg_dict)
