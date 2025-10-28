import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

# Physical constants
c = 299792458.0  # m/s


def sech(x):
    return 1.0 / np.cosh(x)


# ---- Pulse parameters ----
lambda0 = 850e-9  # central wavelength [m]
T_FWHM = 50e-15
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))  # for Sech Pulse
Npts = 2**12  # number of sample points
T_win = 100 * T0  # total time window [s]
dt = T_win / Npts
t = np.arange(-Npts / 2, Npts / 2) * dt

# Carrier frequency
f0 = c / lambda0
omega0 = 2 * np.pi * f0

# ---- Build full field with carrier ----
A_t = sech(t / T0)  # envelope
E_t = A_t * np.exp(1j * omega0 * t)  # carrier modulation

# ---- Numerical FFT spectrum ----
freq = fft.fftfreq(Npts, dt)  # Hz
spec_num = np.abs(fft.fft(E_t)) ** 2
spec_num /= spec_num.max()

# ---- Convert to wavelength axis ----
mask_pos = freq > 0
freq_pos = freq[mask_pos]
lambda_axis = c / freq_pos
jacobian = c / (lambda_axis**2)
S_num_lambda = spec_num[mask_pos] * jacobian

# ---- Analytic spectrum ----
omega_axis = 2 * np.pi * freq_pos
spec_ana_env = sech((np.pi * T0 * (omega_axis - omega0)) / 2) ** 2
spec_ana_env /= spec_ana_env.max()
S_ana_lambda = spec_ana_env * jacobian

# ---- Small floor to avoid log(0)
eps = np.finfo(float).eps

# ---- Convert to dB safely
S_num_dB = 10 * np.log10((S_num_lambda + eps) / np.max(S_num_lambda + eps))
S_ana_dB = 10 * np.log10((S_ana_lambda + eps) / np.max(S_ana_lambda + eps))

# ---- Plot ----
plt.figure(figsize=(8, 5))
plt.plot(lambda_axis * 1e9, S_num_dB, label="Numeric FFT")
plt.plot(lambda_axis * 1e9, S_ana_dB, "--", label="Analytic Sech")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Density (dB)")
plt.title(f"Sech Pulse Spectrum Centered at {lambda0*1e9:.0f} nm")
plt.grid(True, which="both")
plt.legend()
plt.xlim(500, 1500)  # +/-200 nm around centre
# plt.ylim(-50, 5)  # dynamic range to -100 dB
plt.show()
