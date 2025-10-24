from splitstep import fiber_propagation
from functions import read_hdf5 as rdhd
from numpy import flip, imag, real,sqrt,log
from scipy.interpolate import UnivariateSpline
from scipy.constants import c, pi
from datetime import datetime
import os

folder_path = os.getcwd()
location = os.path.join(folder_path, "dispersion_data","waveguide.h5")
waveguidename = "pcf"
data_label_w, data_w = rdhd(location, waveguidename)
print(data_label_w)

#Fiber informaiton
freq = data_w[data_label_w.index("freq")]
beta1 = data_w[data_label_w.index("beta1")]
beta2 = data_w[data_label_w.index("beta2")]
beta3 = data_w[data_label_w.index("beta3")]
neff = data_w[data_label_w.index("n_eff")]
n = real(neff)
k = -imag(neff)
alpha = 4 * pi * freq * k / c
beta1_spl = UnivariateSpline(freq, beta1, k=5)
beta2_spl = UnivariateSpline(freq, beta2, k=5)
beta3_spl = UnivariateSpline(freq, beta3, k=5)
n_spl = UnivariateSpline(freq, n, k=5)
alpha_spl = UnivariateSpline(freq, alpha, k=5)
s = 0.05*0
tr = 0.05*0
gamma = 0
# gamma = 35e-30 * 1000 / (T0) ** 2
z_tot = 0.105                               #Fiber length in m

#Initial Pulse information
E_pulse = 1e-3                             #pulse energy in mJ
# P0 = 1e3                                   #peak power in W
lambda0 = 0.920                              #pump wl in um
T_FWHM = 10e-15
T0 = T_FWHM/(2*sqrt(log(2)))
P0 = E_pulse/T_FWHM                        #peak power in W

A = datetime.now()

sim = fiber_propagation(lambda0, P0,T0,z_tot, alpha, beta2, beta3, s, tr, gamma)
sim.run("gaussian")

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()
