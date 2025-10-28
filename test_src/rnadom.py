import numpy as np
import matplotlib.pyplot as plt

n = 100
x = np.arange(-n / 2, n / 2)
mask_pos = x > 0
f = x[mask_pos]
x_p = 1 / f
y = np.arange(0, n)
xx, yy = np.meshgrid(y, x)
xp, yy = np.meshgrid(y, x_p)
zz = xx[:, mask_pos]

plt.pcolormesh(xp, yy, zz)
plt.colorbar()
plt.show(block=True)
