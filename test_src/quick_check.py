import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

omega0 = 2 * np.pi * 2

a = [1,2,3,4,5]
n = np.size(a)
m = 2*n
b = np.pad(a,(0,m-n),mode='constant')
print(b)
