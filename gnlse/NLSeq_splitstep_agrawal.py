# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:59:38 2023

@author: nhj71
"""

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from datetime import datetime

fiblen = 10 # fiber length (in units of L_D)
beta2 = -1 # sign of GVD parameter beta_2
N =1 # soliton order

#set simulation parameters
nt = 1024 
Tmax = 32 # FFT points and window size
step_num = round(20*fiblen*N^2) # No. of z steps
deltaz = fiblen/step_num # step size in z
dtau = (2*Tmax)/nt # step size in tau
tau = np.arange(-nt/2,nt/2)*dtau # time array
omega1 = fft.fftshift(np.arange(-nt/2,nt/2))*(np.pi/Tmax) # omega array
omega = 2*np.pi*fft.fftfreq(nt,dtau)
uu = 1/np.cosh(tau) # sech pulse shape (can be modified)

#---Plot input pulse shape and spectrum
temp = fft.fftshift(fft.ifft(uu)) # Fourier transform
spect = abs(temp)**2 # input spectrum
spect = spect/max(spect) # normalize
freq = fft.fftshift(omega)/(2*np.pi) # freq. array

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (14,5))
ax1.plot(tau, abs(uu)**2)
ax1.set_xlabel('Normalized Time') 
ax1.set_ylabel('Normalized Power')

ax2.plot(freq, spect)
ax2.set_xlabel('Normalized Frequency')
ax2.set_ylabel('Spectral Power')

#---store dispersive phase shifts to speedup code
dispersion = np.exp(0.5*(1j)*beta2*omega**2*deltaz) # phase factor
hhz = (1j)*N**2*deltaz # nonlinear phase factor

#*********[ Beginning of MAIN Loop]***********
# scheme: 1/2N -> D -> 1/2N first half step nonlinear
temp = uu*np.exp(abs(uu)**2*hhz/2) # note hhz/2


for n in range(step_num):
    
    f_temp = fft.ifft(temp)*dispersion
    uu = fft.fft(f_temp)
    temp = uu*np.exp(abs(uu)**2*hhz)

uu = temp*np.exp(-abs(uu)**2*hhz/2) # Final field
#***************[ End of MAIN Loop ]**************
#----Plot output pulse shape and spectrum
temp = fft.fftshift(fft.ifft(uu)) # Fourier transform
spect = abs(temp)**2 # output spectrum
spect = spect/max(spect) # normalize

ax1.plot(tau, abs(uu)**2) 
ax2.plot(freq, spect) 

plt.show(block=True)