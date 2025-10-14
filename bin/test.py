from scipy.constants import c

N = 75

lambda_um = 0.5
initial_frq_thz = c / (lambda_um * 1e-6) * 1e-12

stop_wavelength_um = 1.5
final_frq_thz = c / (stop_wavelength_um * 1e-6) * 1e-12

del_freq = (initial_frq_thz - final_frq_thz) / N
print(del_freq)
