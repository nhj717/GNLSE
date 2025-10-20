# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:54:01 2023

@author: nhj71
"""

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

class fiber_propagation:
    
    "set initial values"    
    def __init__(self,alpha,beta2,beta3,gamma,P0,T0):
        
        self.alpha = alpha
        self.beta2 = beta2
        self.beta3 = beta3
        self.gamma = gamma
        self.T0 = 1
        self.N = 3         #order of soliton
        
        self.z_tot = np.pi       #unit of LD
        self.z_steps = round(20*self.z_tot*self.N**2) # No. of z steps
        self.delz = self.z_tot/self.z_steps
        self.z = np.linspace(0,self.z_steps,self.z_steps+1)*self.delz
        
        self.Tspan = 100    #multiple of T0
        self.tau_steps = 5000
        self.deltau = self.Tspan/self.tau_steps
        self.tau = np.linspace(-self.tau_steps/2,self.tau_steps/2,self.tau_steps+1)*self.deltau
        self.omega = fft.fftshift(self.tau/self.deltau)*(2*np.pi/self.Tspan)         # omega array
        
        
        if beta2 != 0:
            self.Ld = T0**2/abs(beta2)
        else:
            self.Ld = 1/(gamma)
            

        self.E = np.zeros((self.tau_steps+1,self.z_steps+1),dtype = "complex128")
        self.spectrum = np.zeros((self.tau_steps+1,self.z_steps+1),dtype = "complex128")
        
        
    def source(self,shape):        
        
        if shape =="gaussian":
            self.E[:,0] = np.exp(-(self.tau)**2/(2*self.T0**2))
            self.spectrum[:,0] = fft.ifft(self.E[:,0])
            
        if shape =="sech":
            self.E[:,0] = 1/np.cosh(self.tau)
            self.spectrum[:,0] = fft.ifft(self.E[:,0])
            
        if shape =="lorentzian":
            T0 = self.T0
            self.E[:,0] = (1/np.pi)(T0/2)/((self.tau)**2+(T0/2)**2)
            self.spectrum[:,0] = fft.ifft(self.E[:,0])
            
        

    def run(self,shape):
        fiber_propagation.source(self,shape)
        
        E = self.E
        spectrum = self.spectrum
        dispersion = np.exp(0.5*(1j)*self.beta2*self.omega**2*self.delz-(1/6)*(1j)*self.beta3*self.omega**3*self.delz-self.alpha/2) # phase factor
        nonlinear = (1j)*self.N**2*self.delz # nonlinear phase factor
        
        # scheme: 1/2N -> D -> 1/2N first half step nonlinear
        E[:,1] = np.exp(0.5*nonlinear*abs(E[:,0])**2)*E[:,0]
        
        for i in range(1,self.z_steps):
            spectrum[:,i] = dispersion*fft.ifft(E[:,i])
            Ei = fft.fft(spectrum[:,i])
            E[:,i+1] = np.exp(nonlinear*abs(Ei)**2)*Ei
        
        E[:,-1] = np.exp(-0.5*nonlinear*abs(E[:,-1])**2)*E[:,-1]
        spectrum[:,-1] = fft.ifft(E[:,-1])
        
        self.E = E
        self.spectrum = fft.ifftshift(spectrum,axes = 0)

    
    def draw(self):
        mag1 =  0.1
        mag2 = 0.3
        index1, index2 = int(self.tau_steps/2-mag1/2*self.tau_steps),int(self.tau_steps/2+mag1/2*self.tau_steps)
        index3, index4 = int(self.tau_steps/2-mag2/2*self.tau_steps),int(self.tau_steps/2+mag2/2*self.tau_steps)
        w = self.omega
        f = (1/(2*np.pi))*fft.ifftshift(w)
        tau,z = self.tau ,self.z
        tt,zz = np.meshgrid(tau,z)
        ff,zz_f = np.meshgrid(f,z)
        
        vis1 = 10*np.log10(np.transpose((abs(self.E)/np.max(abs(self.E)))**2))
        vis2 = 10*np.log10(np.transpose((abs(self.spectrum)/np.max(abs(self.spectrum)))**2))
        
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,15),gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('simulation')
        
        pcm1 = ax1.pcolor(tt[:,index1:index2],zz[:,index1:index2],vis1[:,index1:index2], cmap='jet',vmin = -30,vmax = 0)
        # ax1.set_aspect(30)
        ax1.set_title('Time domain')
        cb1 = plt.colorbar(pcm1,shrink = 0.75)
        
        pcm2 = ax2.pcolor(ff[:,index3:index4],zz_f[:,index3:index4],vis2[:,index3:index4],cmap = 'jet',vmin = -62,vmax = 0)
        # ax2.set_aspect(15)
        ax2.set_title('freq. domain')
        cb2 = plt.colorbar(pcm2,shrink = 0.75)
            
        ax3.plot(tau,abs(self.E[:,0])**2, label = 'before')
        ax3.plot(tau,abs(self.E[:,-1])**2, label = 'after')
        ax3.legend()
        ax3.set_title('Time domain')
        
        ax4.plot(f,abs(self.spectrum[:,0])**2, label = 'before')
        ax4.plot(f,abs(self.spectrum[:,-1])**2, label = 'after')
        ax4.legend()
        ax4.set_title('freq. domain')
        
        self.spec = abs(self.spectrum[:,0])**2

 
        
alpha = 0
beta2 = -1
beta3 = 0.04
P0 = 1
T0 = 1
gamma = 35e-30*1000/(T0)**2
# gamma = 0


A = datetime.now()

sim = fiber_propagation(alpha, beta2, beta3, gamma, P0, T0)
sim.run("sech")


B = datetime.now()
print('time : for loop', (B - A).total_seconds())

sim.draw()


