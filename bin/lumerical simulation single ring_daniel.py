import sys,os
sys.path.append('C:\\Program Files\\Lumerical\\v252\\api\\python')
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\Lumerical\v252\bin"
# sys.path.append("/opt/lumerical/v221/api/python/")
import lumapi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import scipy.constants as c


def A2d(A):
    return np.sqrt(4 / np.pi * A)


def getMFD(x, y, I):
    # Calculation of effective mode area via overlap integral according to
    # G. P. Agrawal, Nonlinear Fiber Optics, 4th ed. (Academic, 2007)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    numerator = (np.sum(I)*dx*dy)**2
    denominator = np.sum(I**2)*dx*dy
    A = numerator / denominator
    return A2d(A)


if __name__=='__main__':
    ###########################################################################
    # Definitions
    ###########################################################################
    n = 6
    # parameters of 2022-01-20#1 SRF
    core_diameter = 17.3e-6
    capillary_outer_diameter = 9e-6
    capillary_wall_thickness = .25e-6
    jacket_thickness = 20e-6
    wavelength = 0.488
    fde_resolution = 0.2e-6

    ###########################################################################
    # Simulation with Lumerical
    ###########################################################################
    file = 'srf_thesis_data_488'

    if not Path(file+'.npz').exists():
        with lumapi.MODE(hide=True) as mode:
            mode.save('main.lms')
            capillary_inner_diameter = capillary_outer_diameter - 2*capillary_wall_thickness
            jacket_inner_diameter = core_diameter + 2*capillary_outer_diameter
            jacket_outer_diameter = jacket_inner_diameter + 2*jacket_thickness
            fde_region = jacket_inner_diameter - 1.1e-6

            mode.switchtolayout()
            mode.deleteall()

            mode.addring()
            mode.set('name','jacket')
            mode.set('outer radius', jacket_outer_diameter/2)
            mode.set('inner radius', jacket_inner_diameter/2)
            mode.set('material', 'SiO2 (Glass) - Palik')

            for i in range(n):
                mode.addring()
                mode.set('name', f'ring{i}')
                mode.set('outer radius', capillary_outer_diameter/2)
                mode.set('inner radius', capillary_inner_diameter/2)
                mode.set('material', 'SiO2 (Glass) - Palik')
                r = (core_diameter+capillary_outer_diameter)/2
                phi = i*2*np.pi/n
                mode.set('x', r * np.cos(phi))
                mode.set('y', r * np.sin(phi))

            mode.addfde()
            mode.set('x span', fde_region)
            mode.set('y span', fde_region)
            mode.set('define x mesh by', 'maximum mesh step')
            mode.set('define y mesh by', 'maximum mesh step')
            mode.set('dx', fde_resolution)
            mode.set('dy', fde_resolution)
            mode.set('x min bc', 'PML')
            mode.set('x max bc', 'PML')
            mode.set('y min bc', 'PML')
            mode.set('y max bc', 'PML')

            mode.set('frequency', c.c / wavelength * 1e6)
            mode.set('search', 'in range')
            mode.set('n1', 1)
            mode.set('n2', 0.99)

            mode.save()

            print('Find modes...')
            n = mode.findmodes()
            mode.selectmode('mode1')

            # mode.selectmode('mode1')
            # mode.setanalysis('track selected mode', 1)
            # mode.setanalysis('stop wavelength', wavelength * scaling_factor / 1e6)
            # mode.setanalysis('number of points', n_points)
            # mode.setanalysis('number of test modes', 3)
            # # mode.setanalysis('store mode profiles while tracking', True)
            # print('Starting frequency sweep...')
            # mode.frequencysweep()
            x = np.squeeze(mode.getdata(f'FDE::data::mode1', 'x'))
            y = np.squeeze(mode.getdata(f'FDE::data::mode1', 'y'))

            Ex = mode.getdata(f'FDE::data::mode1', 'Ex').squeeze()
            Ey = mode.getdata(f'FDE::data::mode1', 'Ey').squeeze()
            Ez = mode.getdata(f'FDE::data::mode1', 'Ez').squeeze()
            # neff = np.real(mode.getdata(f'FDE::data::frequencysweep', 'neff').squeeze())
            # frequencies = mode.getdata('FDE::data::frequencysweep', 'f').squeeze()
            I = np.abs(Ex**2 + Ey**2 + Ez**2)
            MFD = getMFD(x, y, I)
            print(MFD)

            np.savez(file+'.npz', MFD=MFD, x=x, y=y, Ex=Ex, Ey=Ey, Ez=Ez)

    #     np.savez(file+'.npz',
    #              MFD=MFD,
    #              taper_diameters=taper_diameters,
    #              neff=neff,
    #              d_core=core_diameter,
    #              d_clad=clad_diameter)
    # else:
    #     with np.load(file+'.npz') as f:
    #         MFD = f['MFD']
    #         taper_diameters = f['taper_diameters']
    #         neff = f['neff']

    ###########################################################################
    # Plots
    ###########################################################################
    # plt.rcParams['font.family'] = 'serif'
    # plt.figure(tight_layout=True)
    # plt.plot(taper_diameters * 1e6, MFD * 1e6, linestyle='-', color='black', linewidth=1, zorder=1)
    # plt.plot(taper_diameters * 1e6, MFD * 1e6, '.', color='black', ms=5, zorder=3)
    # plt.xlabel('Taper diameter [µm]')
    # plt.ylabel('MFD [µm]')
    # sns.despine(offset=10, trim=True)
    # plt.savefig(file+'.png', dpi=300)
    #
    # plt.figure(tight_layout=True)
    # plt.plot(taper_diameters * 1e6, neff, c='k')
    # plt.xlabel('Taper diameter [µm]')
    # plt.ylabel('n_eff')
    # plt.savefig(file+'_neff.png', dpi=300)
    #
    # plt.show(blocked=True)
