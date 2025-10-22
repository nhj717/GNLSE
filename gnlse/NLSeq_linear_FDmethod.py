# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 11:33:51 2023

@author: nhj71
"""


import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from datetime import datetime

class fiber_propagation:
    
    "set initial values"    
    def __init__(self,alpha,beta2,gamma,P0,T0,f0,num_tsteps,num_zsteps):
        self.c = 1
        self.mu = 1
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma
        self.P0 = P0
        self.T0 = T0
        self.w0 = 2*np.pi*f0
        self.factor = np.pi
        self.Lnl = 1/(gamma*P0)
        if beta2 != 0:
            self.Ld = T0**2/abs(beta2)
        self.T = np.linspace(-50*T0,50*T0,num_tsteps)
        self.Tstep = 100*T0/num_tsteps
        self.Z = np.linspace(0.0,self.factor*self.Ld,num_zsteps)
        self.Zsteps = self.factor*self.Ld/num_zsteps
        self.N = num_zsteps
        self.E = np.zeros((num_tsteps,num_zsteps),dtype = "complex128")
        self.spectrum = np.zeros((num_tsteps,num_zsteps),dtype = "complex128")
        self.test = np.zeros((num_tsteps,num_zsteps),dtype = "complex128")

    def dtt(self,j):
        #use gaus-seidel method to update E
        E = self.E
        x = (E[2:,j]-2*E[1:-1,j]+E[:-2,j])/(self.Tstep)**2        
        return x

        
    def update_periodic(self,i):
        E = self.E
        spectrum = self.spectrum
        x = fiber_propagation.dtt(self,i)
        #Efield update
        E[1:-1,i+1] = (1-self.alpha/2*self.Zsteps+(1j)*self.gamma*self.P0*(E[1:-1,i]*np.conj(E[1:-1,i]))*self.Zsteps)*E[1:-1,i]-(1j)*self.Zsteps*self.beta2/2*x
        
        self.test[1:-1,i+1] = x
        
        #boundary condition
        # E[0,i+1] = E[3,i+1]-3*E[1,i+1]-3*E[2,i+1]
        # E[-1,i+1] = E[-4,i+1]-3*E[-3,i+1]-3*E[-2,i+1]
        E[0,i+1] = E[0,i]
        E[-1,i+1] = E[-1,i]
        
        fft_func = fft.fft(E[1:-1,i+1])
        spectrum[1:-1,i+1] = fft.fftshift(fft_func)
        
        self.E = E
        self.spectrum = spectrum
    
        
    def source(self,shape):        
        
        if shape =="gaussian":
            self.E[:,0] = np.exp(-(self.T)**2/(2*self.T0**2))*np.exp((-1j)*self.w0*(self.T-self.T0))
            fft_func = fft.fft(self.E[:,0])
            self.spectrum[:,0] = fft.fftshift(fft_func)
            
        if shape =="sech":
            self.E[:,0]= 1/np.cosh(self.T/self.T0)*np.exp((-1j)*self.w0*(self.T-self.T0))
            
        if shape =="lorentzian":
            T0 = self.T0
            self.E[:,0]= (1/np.pi)(T0/2)/((self.T)**2+(T0/2)**2)*np.exp((-1j)*self.w0*(self.T-self.T0))
            

    def run(self,shape,boundary):
        fiber_propagation.source(self,shape)
        
        if boundary=="periodic":
            update = fiber_propagation.update_periodic
        
        for i in range(self.N-1):
            update(self,i)

    
    def draw(self):
        X,Y = self.T ,self.Z
        vis1 = np.transpose(abs(self.E))
        vis2 = np.transpose(abs(self.spectrum))
        
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        fig.suptitle('simulation')
        pcm = ax.pcolor(X,Y,vis1)
        ax.set_title('Time domain')
        fig.colorbar(pcm)
        # fig.colorbar(pcm,ax1)

        # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
        # fig.suptitle('simulation')
        # pcm = ax1.pcolor(X,Y,vis1)
        # ax1.set(autoscale_on=False, aspect='equal')
        # ax1.set_title('Time domain')
        # # fig.colorbar(pcm,ax1)
        
        # pcm = ax2.pcolor(X,Y,vis2)
        # ax2.set(autoscale_on=False, aspect='equal')
        # ax2.set_title('Spectrum')
        # # fig.colorbar(pcm,ax2)
        plt.show(block=True)

alpha = 0
beta2 = -35e-30*1000
# beta2 = 0
P0 = 4
T0 = 5e-14
gamma = 35e-30*1000/(T0)**2
f0 = 375e12
num_tsteps = 200
num_zsteps = 10000


A = datetime.now()

sim = fiber_propagation(alpha, beta2, gamma, P0, T0, f0, num_tsteps, num_zsteps)
sim.run("sech","periodic")


B = datetime.now()
print('time : for loop', (B - A).total_seconds())

sim.draw()



