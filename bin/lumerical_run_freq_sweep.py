import sys, os

sys.path.append("C:\\Program Files\\Lumerical\\v252\\api\\python")
import lumapi
import numpy as np
from functions import save_dict_to_hdf5, Saitoh
from datetime import datetime
from scipy.constants import c

description = "freq_sweep"

output_basedir = os.path.dirname(os.path.abspath(__file__))
now = datetime.now()
date_time_str = now.strftime("%Y-%b-%d_%H-%M-%S")
date_str = now.strftime("%Y-%b-%d")
output_dir = os.path.join(output_basedir, date_str, description + "_" + date_time_str)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# settings
d_um = 0.40
pitch_um = 1.56

lambda_um = 0.5
initial_frq_thz = c / (lambda_um * 1e-6) * 1e-12
num_trial_modes = 5

stop_wavelength_um = 2.0
final_frq_thz = c / (stop_wavelength_um * 1e-6) * 1e-12
num_frequency_points = 300
num_sample_modes = num_trial_modes


mode = lumapi.MODE(
    filename="PCF_20240422.lms", hide=True
)  # constructor for the Lumerical product


def set_initial_analysis_props(initial_frq_thz, d, pitch, num_trial_modes):
    mode.switchtolayout()
    mode.select("FDE")
    mode.set("pml sigma", 15)
    mode.set("pml layers", 100)
    mode.setanalysis("frequency", initial_frq_thz * 1e12)
    mode.setanalysis("number of trial modes", num_trial_modes)
    mode.setanalysis("search", "near n")
    wl = c / initial_frq_thz * 1e-6
    s = Saitoh(wl, d, pitch)
    mode.setanalysis("n", s.neff())


def calc_and_save_initial_mode_profiles(output_dir):
    output_filename = os.path.join(output_dir, "data.h5")
    mode.findmodes()
    mode_selection = [1]

    var_names = [
        "surface_normal",
        "dimension",
        "f",
        "neff",
        "ng",
        "loss",
        "TE polarization fraction",
        "waveguide TE/TM fraction",
        "mode effective area",
        "x",
        "y",
        "z",
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
        "Z0",
    ]
    mode_names = ["mode{:d}".format(i) for i in mode_selection]
    group_name = "initial_mode_profiles"
    arg_dict = {}
    for mode_name in mode_names:
        for var_name in var_names:
            arg_dict["{}_{}".format(mode_name, var_name)] = mode.getdata(
                mode_name, var_name
            )
    save_dict_to_hdf5(output_filename, group_name, arg_dict)
    print("initially selected mode data saved at {}".format(output_filename))
    return mode_selection, mode_names, var_names, output_filename


def set_frequency_sweep_props(
    stop_wavelength_um, num_frequency_points, num_sample_modes, d, pitch
):
    mode.setanalysis("stop wavelength", stop_wavelength_um * 1e-6)
    mode.setanalysis("track selected mode", 1)
    mode.setanalysis("number of points", num_frequency_points)
    mode.setanalysis("number of test modes", num_sample_modes)
    mode.setanalysis("detailed dispersion calculation", 0)
    mode.setanalysis("store mode profiles while tracking", 0)


def perform_frequency_sweep(mode_selection):
    arg_dict = {}
    for mode_index in mode_selection:
        mode_name = "mode{:d}".format(mode_index)
        mode.selectmode(mode_index)
        mode.setanalysis("track selected mode", 1)
        mode.frequencysweep()
        var_names = [
            "neff",
            "loss",
            "vg",
            "D",
            "beta",
            "f",
            "f_vg",
            "f_D",
            "mode_number",
            "overlap",
        ]
        for var_name in var_names:
            arg_dict["{}_{}".format(mode_name, var_name)] = mode.getdata(
                "FDE::data::frequencysweep", var_name
            )
    group_name = "frequency_sweep"
    output_filename = os.path.join(output_dir, "data.h5")
    save_dict_to_hdf5(output_filename, group_name, arg_dict)
    print("frequency data saved at {}".format(output_filename))
    return output_filename


set_initial_analysis_props(
    initial_frq_thz=initial_frq_thz,
    d=d_um,
    pitch=pitch_um,
    num_trial_modes=num_trial_modes,
)
mode_selection, mode_names, var_names, output_filename = (
    calc_and_save_initial_mode_profiles(output_dir=output_dir)
)
set_frequency_sweep_props(
    stop_wavelength_um=stop_wavelength_um,
    num_frequency_points=num_frequency_points,
    num_sample_modes=num_sample_modes,
    d=d_um,
    pitch=pitch_um,
)
perform_frequency_sweep(mode_selection=mode_selection)
mode.save("result.lms")
