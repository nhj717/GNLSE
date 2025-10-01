import functions
from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt

output_dir = r'C:\Users\hnam\PycharmProjects\GNLSE\bin\2025-Oct-01\freq_sweep_2025-Oct-01_17-55-53\data.h5'
group_name = 'initial_mode_profiles'
mode = 1

data_label,data = functions.read_hdf5(output_dir,group_name)
print(data_label)
x = data[data_label.index(f'mode{mode}_x')]
y = data[data_label.index(f'mode{mode}_y')]
Ex = data[data_label.index(f'mode{mode}_Ex')]
Ey= data[data_label.index(f'mode{mode}_Ey')]

xx,yy = np.meshgrid(x,y,indexing = 'ij')
I = np.reshape(abs(Ex),(np.size(x),np.size(y)))**2+np.reshape(abs(Ey),(np.size(x),np.size(y)))**2
I = I/np.max(I)


# plot
mm = 1 / 25.4
size_parameter = (150, 110, 12)  # width, height, font
layout = (0.00, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ['x [um]', 'y [um)]']  # xlabel and ylabel
(width,height,fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect = layout)
pcm = ax.pcolor(xx*1E6,yy*1E6,I, cmap='turbo', rasterized=True)
cb1 = plt.colorbar(pcm, shrink=1)
ax.set(aspect='equal')
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis='both', which='major', size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
plt.show(block=True)

