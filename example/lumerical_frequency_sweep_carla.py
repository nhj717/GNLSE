import sys
import os
import numpy as np
from datetime import datetime
sys.path.append(r'C:\\Program Files\\Lumerical\\v252\\api\\python\\')
#sys.path.append(os.path.dirname(__file__)) #Current directory
import lumapi


description = 'test_sweep'

output_basedir = os.path.dirname(os.path.abspath(__file__))
now = datetime.now()
date_time_str = now.strftime('%Y-%b-%d_%H-%M-%S')
date_str = now.strftime('%Y-%b-%d')
output_dir = os.path.join(output_basedir, date_str, description+'_'+date_time_str)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


from scipy.constants import c # vacuum light velocity

# settings
core_radius_um = 4.1
cladding_radius_um = 62.5
core_refr_index = 1.4491984
cladding_refr_index = 1.4440

lambda_um = 1.550

fde_region_size_um = 2*5*cladding_radius_um
fde_mesh_cell_size_um = 50.0
refinement_region_sizes_um, refinement_region_cell_sizes_um, refinement_region_names = [], [], []
refinement_region_names.append('cladding_mesh_refinement')
refinement_region_sizes_um.append(2*cladding_radius_um*1.1)
refinement_region_cell_sizes_um.append(1.0)
refinement_region_names.append('core_mesh_refinement')
refinement_region_sizes_um.append(2*core_radius_um*1.1)
refinement_region_cell_sizes_um.append(.1)

initial_frq_thz=c/(lambda_um*1e-6)*1e-12
num_trial_modes=7

stop_wavelength_um=lambda_um*5
num_frequency_points=30
num_sample_modes=num_trial_modes



mode = lumapi.MODE(filename = None, hide=False) # constructor for the Lumerical product

def reset():
    mode.switchtolayout()
    mode.deleteall()

def define_material_geometry(core_radius_um, cladding_radius_um,
                             core_refr_index, cladding_refr_index):
    # core
    mode.addcircle()
    mode.set('name', 'core')
    mode.set('radius', core_radius_um*1e-6)
    mode.set('index', core_refr_index)
    # cladding
    mode.addring()
    mode.set('name', 'cladding')
    mode.set('inner radius', core_radius_um*1e-6)
    mode.set('outer radius', cladding_radius_um*1e-6)
    mode.set('index', cladding_refr_index)

def define_mesh_structure(fde_region_size_um, fde_mesh_cell_size_um,
                          refinement_region_sizes_um = [], refinement_region_cell_sizes_um = [], refinement_region_names = []):
    
    for i in range(len(refinement_region_sizes_um)):

        mode.addmesh()
        mode.set('name', refinement_region_names[i])
        mode.set('x', 0)
        mode.set('y', 0)
        mode.set('x span', refinement_region_sizes_um[i]*1e-6)
        mode.set('y span', refinement_region_sizes_um[i]*1e-6)
        mode.set('override x mesh', 1)
        mode.set('override y mesh', 1)
        mode.set('set maximum mesh step', 1)
        mode.set('dx', refinement_region_cell_sizes_um[i]*1e-6)
        mode.set('dy', refinement_region_cell_sizes_um[i]*1e-6)


    mode.addfde()
    mode.set('x', 0)
    mode.set('y', 0)
    mode.set('x span', fde_region_size_um*1e-6)
    mode.set('y span', fde_region_size_um*1e-6)
    mode.set('define x mesh by', 'maximum mesh step')
    mode.set('define y mesh by', 'maximum mesh step')
    mode.set('dx', fde_mesh_cell_size_um*1e-6)
    mode.set('dy', fde_mesh_cell_size_um*1e-6)
    mode.set('x min bc', 'PML')
    mode.set('x max bc', 'PML')
    mode.set('y min bc', 'PML')
    mode.set('y max bc', 'PML')
    
def set_initial_analysis_props(initial_frq_thz, num_trial_modes):
    mode.setanalysis('frequency', initial_frq_thz*1e12)
    mode.setanalysis('number of trial modes', num_trial_modes)
    mode.setanalysis('search', 'near n')
    mode.setanalysis('use max index', 1)

def calc_and_save_initial_mode_profiles(output_dir):
    output_filename = os.path.join(output_dir, 'initial_mode_profile_data.npz')
    mode.findmodes()
    mode_selection = [1,2,3,4,5,6]

    var_names = ['surface_normal', 'dimension', 'f', 'neff', 'ng', 'loss', 'TE polarization fraction', 'waveguide TE/TM fraction', 'mode effective area', 'x', 'y', 'z', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Z0']
    mode_names = ['mode{:d}'.format(i) for i in mode_selection]
    savez_arg_dict = {}
    for mode_name in mode_names:
        for var_name in var_names:
            savez_arg_dict['{}_{}'.format(mode_name, var_name)] = mode.getdata(mode_name, var_name)
    np.savez(output_filename, **savez_arg_dict)
    print('initially selected mode data saved at {}'.format(output_filename))
    return mode_selection, mode_names, var_names, output_filename

def set_frequency_sweep_props(stop_wavelength_um, num_frequency_points, num_sample_modes):
    mode.setanalysis('stop wavelength', stop_wavelength_um*1e-6)
    mode.setanalysis('number of points', num_frequency_points)
    mode.setanalysis('number of test modes', num_sample_modes)
    mode.setanalysis('store mode profiles while tracking', 1)
    mode.setanalysis('track selected mode', 1)

def perform_frequency_sweep(mode_selection):
    savez_arg_dict = {}
    for mode_index in mode_selection:
        mode_name = 'mode{:d}'.format(mode_index)
        mode.selectmode(mode_index)
        mode.setanalysis('track selected mode', 1)
        mode.frequencysweep()
        var_names = ['neff', 'loss', 'vg', 'D', 'beta', 'f', 'f_vg', 'f_D', 'mode_number', 'overlap', 'x', 'y', 'z', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        for var_name in var_names:
            savez_arg_dict['{}_{}'.format(mode_name, var_name)] = mode.getdata('FDE::data::frequencysweep', var_name)
    output_filename = os.path.join(output_dir, 'frequency_sweep_data.npz')
    np.savez(output_filename, **savez_arg_dict)
    print('frequency data saved at {}'.format(output_filename))
    return output_filename

reset()
define_material_geometry(core_radius_um=core_radius_um, cladding_radius_um=cladding_radius_um,
                         core_refr_index=core_refr_index, cladding_refr_index=cladding_refr_index)
define_mesh_structure(fde_region_size_um=fde_region_size_um, fde_mesh_cell_size_um=fde_mesh_cell_size_um,
                      refinement_region_sizes_um=refinement_region_sizes_um, refinement_region_cell_sizes_um=refinement_region_cell_sizes_um, refinement_region_names=refinement_region_names)
set_initial_analysis_props(initial_frq_thz=initial_frq_thz, num_trial_modes=num_trial_modes)
mode_selection, mode_names, var_names, output_filename = calc_and_save_initial_mode_profiles(output_dir=output_dir)
set_frequency_sweep_props(stop_wavelength_um=stop_wavelength_um, num_frequency_points=num_frequency_points, num_sample_modes=num_sample_modes)
perform_frequency_sweep(mode_selection=mode_selection)