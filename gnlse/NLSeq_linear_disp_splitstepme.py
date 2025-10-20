# -*- coding: utf-8 -*-
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
        self.Tspan = 300
        self.w0 = 2*np.pi*f0
        self.factor = 10*np.pi
        
        if gamma != 0:
            self.Lnl = 1/(gamma*P0)
        else:
            self.Lnl = T0**2/abs(beta2)
        
        if beta2 != 0:
            self.Ld = T0**2/abs(beta2)
        else:
            self.Ld = 1/(gamma*P0)
            
        self.T = np.linspace(-self.Tspan/2*T0,self.Tspan/2*T0,num_tsteps)
        self.Tstep = self.Tspan*T0/num_tsteps
        self.Z = np.linspace(0.0,self.factor*self.Ld,num_zsteps)
        self.Zsteps = self.factor*self.Ld/num_zsteps
        self.N = num_zsteps
        self.E = np.zeros((num_tsteps,num_zsteps),dtype = "complex128")
        self.spectrum = np.zeros((num_tsteps,num_zsteps),dtype = "complex128")
        self.w = fft.fftshift(np.arange(int(-num_tsteps)/2,int(num_tsteps/2))*2*np.pi/(self.Tspan*T0))

        
    def update(self,i):
        E = self.E
        spectrum = self.spectrum
        
        #Nonlinear Efield update
        Ei = np.exp((self.Zsteps/2)*(-self.alpha/2+(1j)*self.gamma*self.P0*abs(E[:,i])**2))*E[:,i] 
        #Linear Efield update
        
        spectrum[:,i+1] = np.exp((self.Zsteps)*0.5*(1j)*self.beta2*(self.w[:]-self.w0)**2)*fft.ifft(Ei)
        E[:,i+1] = fft.fft(spectrum[:,i+1])
        
        self.E = E
        self.spectrum = spectrum
    
        
    def source(self,shape):        
        
        if shape =="gaussian":
            E = np.exp(-(self.T)**2/(2*self.T0**2))
            self.E[:,0] = E/np.sum(E)
            self.spectrum[:,0] = fft.ifft(E)
            
        if shape =="sech":
            E= 1/np.cosh(self.T/self.T0)
            self.E[:,0] = E/np.sum(E)
            fft_func = fft.fft(self.E[:,0])
            self.spectrum[:,0] = fft.fftshift(fft_func)
            
        if shape =="lorentzian":
            T0 = self.T0
            E= (1/np.pi)(T0/2)/((self.T)**2+(T0/2)**2)
            self.E[:,0] = E/np.sum(E)
            fft_func = fft.fft(self.E[:,0])
            self.spectrum[:,0] = fft.fftshift(fft_func)
            

    def run(self,shape):
        fiber_propagation.source(self,shape)
        
        update = fiber_propagation.update
        
        for i in range(self.N-1):
            update(self,i)

    
    def draw(self):
        w,x,z = self.w,self.T ,self.Z
        xx,zz = np.meshgrid(x,z)
        ww,zz_w = np.meshgrid(w,z)
        
        vis1 = np.transpose(abs(self.E))
        vis2 = np.transpose(abs(self.spectrum))
        
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
        fig.suptitle('simulation')
        
        pcm1 = ax1.pcolor(xx,zz,vis1,cmap = 'jet')
        ax1.set_title('Time domain')
        cb1 = plt.colorbar(pcm1,shrink = 0.75)
        
        pcm2 = ax2.pcolor(ww,zz_w,vis2,cmap = 'jet')
        ax2.set_title('freq. domain')
        cb2 = plt.colorbar(pcm2,shrink = 0.75)
        

alpha = 0
beta2 = 35e-30*1000
# beta2 = 0
P0 = 1
T0 = 5e-14
gamma = 35e-30*1000/(T0)**2
# gamma = 0
f0 = 375e12
num_tsteps = 10000
num_zsteps = 50


A = datetime.now()

sim = fiber_propagation(alpha, beta2, gamma, P0, T0, f0, num_tsteps, num_zsteps)
sim.run("sech")


B = datetime.now()
print('time : for loop', (B - A).total_seconds())

sim.draw()



