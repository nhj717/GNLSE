import functions
from scipy.constants import c
from numpy import flip
import matplotlib.pyplot as plt

output_dir = r'C:\Users\hnam\PycharmProjects\GNLSE\bin\2025-Oct-01\freq_sweep_2025-Oct-01_17-55-53\data.h5'
group_name = 'frequency_sweep'
mode = 1

data_label, data = functions.read_hdf5(output_dir,group_name)
print(data_label)
freq = data[data_label.index(f'mode{mode}_f')]
loss = data[data_label.index(f'mode{mode}_loss')]

wl = flip(c/freq*1E6)
loss = flip(loss)


# plot
mm = 1 / 25.4
size_parameter = (150, 100, 12)  # width, height, font
layout = (0.1, 0.05, 1.0, 1.0)  # left,bottom,right,top
label_list = ['wavelength [um]', 'Loss [dB/km]']  # xlabel and ylabel
(width,height,fsize) = size_parameter

fig, ax = plt.subplots(figsize=(width * mm, height * mm))
fig.tight_layout(rect = layout)
pcm = ax.plot(wl,loss)
ax.set_xlabel(label_list[0], fontsize=fsize)
ax.set_ylabel(label_list[1], fontsize=fsize)
ax.tick_params(axis='both', which='major', size=4, width=2, labelsize=10)
# ax.set_xticks([-2,-1,0,1,2])
# ax.set_yticks([-2,-1,0,1,2])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
plt.show(block=True)

