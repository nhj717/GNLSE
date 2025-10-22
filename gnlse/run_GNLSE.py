from splitstep import fiber_propagation
from functions import read_hdf5 as rdhd
from numpy import flip, imag, real
from scipy.interpolate import UnivariateSpline
from scipy.constants import c, pi
from datetime import datetime
import os

folder_path = os.getcwd()
location = os.path.join(folder_path, "dispersion_data","waveguide.h5")
waveguidename = "pcf"
data_label_w, data_w = rdhd(location, waveguidename)
print(data_label_w)

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
P0 = 1
lamb0 = 0.920
T0 = 10e-15
gamma = 0
# gamma = 35e-30 * 1000 / (T0) ** 2

A = datetime.now()

sim = fiber_propagation(lamb0, alpha_spl, beta2_spl, beta3_spl, s, tr, gamma, P0, T0)
sim.run("sech")

B = datetime.now()
print("time : for loop", (B - A).total_seconds())

sim.draw()

# arg_dict = {"wl": c / freq * 1e6, "n_eff": neff}
# arg_dict.update(mode_dispersion(arg_dict["wl"], np.real(arg_dict["n_eff"])))
#
# hdf5_name = "waveguide.h5"
# svhd(hdf5_name, waveguidename, arg_dict)
