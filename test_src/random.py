import numpy as np
import matplotlib.pyplot as plt

n = 6
x = np.arange(-n / 2, n / 2)
y = np.arange(0, n / 2)
xx, yy = np.meshgrid(y, x)
zz = x * x[:, None]
print(xx)
print(x)
print(zz)
