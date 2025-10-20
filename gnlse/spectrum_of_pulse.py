import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from datetime import datetime

class fiber_propagation:
    
    "set initial values"    
    def __init__(self,alpha,beta2,gamma,P0,T0,f0,num_tsteps):
        self.c = 1
        self.mu = 1
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma
        self.P0 = P0
        self.T0 = T0
        self.Tspan = 1000
        self.w0 = 2*np.pi*f0
        
        self.T = np.linspace(-(self.Tspan/2)*T0,(self.Tspan/2)*T0,num_tsteps+1)
        self.Tstep = self.Tspan*T0/num_tsteps

        self.E = np.zeros(num_tsteps+1,dtype = "complex128")
        self.spectrum = np.zeros(num_tsteps+1,dtype = "complex128")
        self.w = np.linspace(self.w0-num_tsteps/2*2*np.pi/(self.Tspan*T0),self.w0+num_tsteps/2*2*np.pi/(self.Tspan*T0),num_tsteps+1)
    
        
    def source(self,shape):  
        T0 = self.T0
        
        if shape =="gaussian":
            # self.E = np.exp(-(self.T)**2/(2*self.T0**2))
            self.E = np.exp(-(self.T)**2/(2*T0**2))
            fft_func = fft.fft(self.E)
            self.spectrum = fft.fftshift(fft_func)
            # self.spectrum = fft_func
            
        if shape =="sech":
            self.E= 1/np.cosh(self.T/T0)
            fft_func = fft.fft(self.E)
            self.spectrum = fft.fftshift(fft_func)
            
        if shape =="lorentzian":
            self.E= (1/np.pi)*(T0/2)/((self.T)**2+(T0/2)**2)
            fft_func = fft.fft(self.E)
            self.spectrum = fft.fftshift(fft_func)
            

    def run(self,shape):
        fiber_propagation.source(self,shape)

    
    def draw(self):
        w = self.w
        x = self.T
        # vis1 = np.transpose(abs(self.E))
        spec = self.spectrum
        y1 = abs(fft.ifft(spec))
        y2 = abs(spec)
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,5))
        fig.suptitle('simulation')
        
        pcm = ax1.plot(x,y1)
        ax1.set_title('Time domain')
        
        pcm = ax2.plot(w,y2)
        ax2.set_title('Freq. domain')
        plt.show(block=True)

alpha = 100
beta2 = -35e-30*1000
# beta2 = 0
P0 = 1
T0 = 5e-14
gamma = 35e-30*1000/(T0)**2
f0 = 375e12
num_tsteps = 5000


A = datetime.now()

sim = fiber_propagation(alpha, beta2, gamma, P0, T0, f0, num_tsteps)
sim.source("gaussian")


B = datetime.now()
print('time : for loop', (B - A).total_seconds())

sim.draw()



