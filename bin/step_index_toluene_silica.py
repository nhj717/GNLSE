import sys
sys.path.append('C:\\Program Files\\Lumerical\\v252\\api\\python')
import lumapi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import scipy.constants as c

# settings
core_radius_um = 5
cladding_radius_um = 62.5

fde_region_size_um = 1.5*core_radius_um
fde_mesh_cell_size_um = 50.0
refinement_region_sizes_um, refinement_region_cell_sizes_um, refinement_region_names = [], [], []
refinement_region_names.append('cladding_mesh_refinement')
refinement_region_sizes_um.append(2*cladding_radius_um*1.1)
refinement_region_cell_sizes_um.append(1.0)
refinement_region_names.append('core_mesh_refinement')
refinement_region_sizes_um.append(2*core_radius_um*1.1)
refinement_region_cell_sizes_um.append(.1)

# initial_frq_thz=c/(lambda_um*1e-6)*1e-12
# num_trial_modes=7
#
# stop_wavelength_um=lambda_um*5
# num_frequency_points=10
# num_sample_modes=num_trial_modes



mode = lumapi.MODE(filename = None, hide=False) # constructor for the Lumerical product
mode.save('test.lms')

def reset():
    mode.switchtolayout()
    mode.deleteall()


def define_material_geometry(core_radius_um, cladding_radius_um):
    # set material
    mode.addmaterial()
    mode.setmaterial("sellmeier","name","fused_silica")
    mode.setmaterial("fused_silica","A0",1)
    mode.setmaterial("fused_silica", "B1", 0.6961663)
    mode.setmaterial("fused_silica", "B2", 0.4079426)
    mode.setmaterial("fused_silica", "B3", 0.8974794)
    mode.setmaterial("fused_silica", "C1", 0.0684043**2)
    mode.setmaterial("fused_silica", "C2", 0.1162414**2)
    mode.setmaterial("fused_silica", "C3", 9.896161**2)

    mode.addmaterial("Sellmeier")
    mode.setmaterial("Sellmeier", "name", "toluene")
    mode.setmaterial("fused_silica", "A0", 1)
    mode.setmaterial("fused_silica", "B1", 1.17477)
    mode.setmaterial("fused_silica", "C1", 0.01825)


    # cladding
    mode.addring()
    mode.set('name', 'cladding')
    mode.set('inner radius', core_radius_um * 1e-6)
    mode.set('outer radius', cladding_radius_um * 1e-6)
    mode.set('material', 'fused_silica')

    # core
    mode.addcircle()
    mode.set('name', 'core')
    mode.set('radius', core_radius_um * 1e-6)
    mode.set('material', 'toluene')



def define_mesh_structure(fde_region_size_um, fde_mesh_cell_size_um,
                          refinement_region_sizes_um=[], refinement_region_cell_sizes_um=[],
                          refinement_region_names=[]):
    for i in range(len(refinement_region_sizes_um)):
        mode.addmesh()
        mode.set('name', refinement_region_names[i])
        mode.set('x', 0)
        mode.set('y', 0)
        mode.set('x span', refinement_region_sizes_um[i] * 1e-6)
        mode.set('y span', refinement_region_sizes_um[i] * 1e-6)
        mode.set('override x mesh', 1)
        mode.set('override y mesh', 1)
        mode.set('set maximum mesh step', 1)
        mode.set('dx', refinement_region_cell_sizes_um[i] * 1e-6)
        mode.set('dy', refinement_region_cell_sizes_um[i] * 1e-6)

    mode.addfde()
    mode.set('x', 0)
    mode.set('y', 0)
    mode.set('x span', fde_region_size_um * 1e-6)
    mode.set('y span', fde_region_size_um * 1e-6)
    mode.set('define x mesh by', 'maximum mesh step')
    mode.set('define y mesh by', 'maximum mesh step')
    mode.set('dx', fde_mesh_cell_size_um * 1e-6)
    mode.set('dy', fde_mesh_cell_size_um * 1e-6)
    mode.set('x min bc', 'PML')
    mode.set('x max bc', 'PML')
    mode.set('y min bc', 'PML')
    mode.set('y max bc', 'PML')

reset()
define_material_geometry(core_radius_um=core_radius_um, cladding_radius_um=cladding_radius_um)