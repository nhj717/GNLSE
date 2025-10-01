import functions
from scipy.constants import c
from numpy import flip
import matplotlib.pyplot as plt

output_dir = r'C:\Users\hnam\PycharmProjects\GNLSE\bin\2025-Oct-01\freq_sweep_2025-Oct-01_11-41-46\frequency_sweep_data.h5'
group_name1 = 'mode1_f_D'
group_name2 = 'mode1_D'

data_label,freq = functions.read_hdf5(output_dir,group_name1)
data_label,D = functions.read_hdf5(output_dir,group_name2)

wl = flip(c/freq*1E6)
D = flip(D)
plt.plot(wl,D)
plt.xlabel('Wavelength [um]')
plt.ylabel('D [ps/(nm km)]')
plt.show(block=True)

