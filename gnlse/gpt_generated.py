import numpy as np
import matplotlib.pyplot as plt

# Physical constants
c = 299792458.0  # m/s


# --------------------
# Safe sech function
# --------------------
def safe_sech(x):
    ax = np.abs(x)
    threshold = 40.0
    out = np.empty_like(ax)
    mask_large = ax > threshold
    out[mask_large] = 2.0 * np.exp(-ax[mask_large])
    out[~mask_large] = 1.0 / np.cosh(x[~mask_large])
    return out


# --------------------
# Raman response
# --------------------
def raman_response(t, tau1=12.2e-15, tau2=32e-15):
    h = np.zeros_like(t)
    pos_t = t >= 0
    h[pos_t] = (
        (tau1**2 + tau2**2)
        / (tau1 * tau2**2)
        * np.exp(-t[pos_t] / tau2)
        * np.sin(t[pos_t] / tau1)
    )
    h /= np.trapezoid(h, t)
    return h


# --------------------
# Nonlinear operator
# --------------------
def nonlinear_operator(A_t, gamma, omega0, fr, hR_t, dt):
    I_t = np.abs(A_t) ** 2
    H_R_f = np.fft.fft(hR_t)
    I_f = np.fft.fft(I_t)
    conv_R_t = np.fft.ifft(H_R_f * I_f) * dt
    power_term = (1 - fr) * I_t + fr * conv_R_t.real

    freq = np.fft.fftfreq(len(A_t), dt) * 2 * np.pi
    dA_dt = np.fft.ifft(1j * freq * np.fft.fft(A_t))

    N_t = 1j * gamma * (A_t * power_term + (1j / omega0) * dA_dt * power_term)
    return N_t


# --------------------
# RK4IP propagation
# --------------------
def propagate_RK4IP(A_t, dz, Nz, L_w, gamma, omega0, fr, hR_t, dt):
    D_half = np.exp(L_w * dz / 2)
    D_full = np.exp(L_w * dz)

    A_f = np.fft.fft(A_t)

    for _ in range(Nz):
        # k1
        U1_t = np.fft.ifft(D_half * A_f)
        k1_t = nonlinear_operator(U1_t, gamma, omega0, fr, hR_t, dt)
        k1_f = np.fft.fft(k1_t)

        # k2
        U2_t = np.fft.ifft(D_half * (A_f + dz / 2 * k1_f))
        k2_t = nonlinear_operator(U2_t, gamma, omega0, fr, hR_t, dt)
        k2_f = np.fft.fft(k2_t)

        # k3
        U3_t = np.fft.ifft(D_half * (A_f + dz / 2 * k2_f))
        k3_t = nonlinear_operator(U3_t, gamma, omega0, fr, hR_t, dt)
        k3_f = np.fft.fft(k3_t)

        # k4
        U4_t = np.fft.ifft(D_full * (A_f + dz * k3_f))
        k4_t = nonlinear_operator(U4_t, gamma, omega0, fr, hR_t, dt)
        k4_f = np.fft.fft(k4_t)

        # combine
        A_f = A_f + dz / 6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)

    return np.fft.ifft(A_f)


# --------------------
# Example run with manual β2..β7
# --------------------
# Simulation grid
Npts = 2**14
T_win = 20e-12
dt = T_win / Npts
t = np.arange(-Npts / 2, Npts / 2) * dt

# Pulse parameters
lambda0 = 850e-9
omega0 = 2 * np.pi * c / lambda0
P0 = 1.0  # W peak power
T_FWHM = 50e-15
T0 = T_FWHM / (2 * np.log(1 + np.sqrt(2)))

# Initial envelope
A_t = np.sqrt(P0) * safe_sech(t / T0)

# ---- Define dispersion coefficients ----
# Units: βk in s^k / m
beta2 = -1.276e-26  # s^2/m
beta3 = 8.119e-41  # s^3/m
beta4 = -1.321e-55  # s^4/m
beta5 = 3.032e-70  # s^5/m
beta6 = -4.196e-98  # s^6/m
beta7 = 2.57e-110  # s^7/m

# Frequency grid (relative to ω0)
freq_fft = np.fft.fftfreq(Npts, dt)  # Hz
Omega = 2 * np.pi * freq_fft  # rad/s offset from ω0

# Linear dispersion operator L_w
L_w = (
    1j * beta2 / 2 * Omega**2
    + 1j * beta3 / 6 * Omega**3
    + 1j * beta4 / 24 * Omega**4
    + 1j * beta5 / 120 * Omega**5
    + 1j * beta6 / 720 * Omega**6
    + 1j * beta7 / 5040 * Omega**7
)

# Fiber params
gamma = 1.0  # W^-1 m^-1
fr = 0.18
hR_t = raman_response(t)
dz = 1e-3  # m per step
Nz = 1000  # steps

# Propagate
A_out = propagate_RK4IP(A_t, dz, Nz, L_w, gamma, omega0, fr, hR_t, dt)

# --------------------
# Plot results in wavelength
# --------------------
spec_num = np.abs(np.fft.fft(A_out)) ** 2
spec_num /= spec_num.max()

# Positive physical frequencies for plotting
freq_phys_pos = freq_fft[freq_fft > 0] + omega0 / (2 * np.pi)
lambda_axis = c / freq_phys_pos
jacobian = c / (lambda_axis**2)
S_lambda = spec_num[freq_fft > 0] * jacobian

eps = np.finfo(float).eps
S_lambda_dB = 10 * np.log10((S_lambda + eps) / np.max(S_lambda + eps))

plt.figure(figsize=(8, 5))
plt.plot(lambda_axis * 1e9, S_lambda_dB)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Density (dB)")
plt.title(f"Output Spectrum at {lambda0*1e9:.0f} nm")
plt.grid(True, which="both")
plt.xlim(lambda0 * 1e9 - 200, lambda0 * 1e9 + 200)
plt.ylim(-80, 5)
plt.show()
