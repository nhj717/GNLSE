import numpy as np
import matplotlib.pyplot as plt


# =========================
# Raman response function
# =========================
def raman_response(t, tau1=12.2e-15, tau2=32e-15):
    """
    Raman response for silica. Agrawal NFO Eq. (2.4.18).
    Returns h_R(t) normalized such that ∫ h_R(t) dt = 1.
    """
    h = np.zeros_like(t)
    pos_t = t >= 0
    h[pos_t] = (
        (tau1**2 + tau2**2)
        / (tau1 * tau2**2)
        * np.exp(-t[pos_t] / tau2)
        * np.sin(t[pos_t] / tau1)
    )
    # Normalize area to 1
    h /= np.trapz(h, t)
    return h


# =========================
# Nonlinear operator
# =========================
def nonlinear_operator(A_t, gamma, omega0, fr, hR_t, dt):
    """
    Full nonlinear term with self-steepening and Raman.
    FFT convolution is used for Raman term.
    """
    # Instantaneous intensity
    I_t = np.abs(A_t) ** 2

    # Raman convolution (zero-padding to avoid temporal aliasing)
    padN = len(A_t)
    H_R_f = np.fft.fft(hR_t)
    I_f = np.fft.fft(I_t)
    conv_R_t = np.fft.ifft(H_R_f * I_f) * dt

    # Instantaneous + delayed term
    power_term = (1 - fr) * I_t + fr * conv_R_t.real

    # Self-steepening (shock term)
    freq = np.fft.fftfreq(len(A_t), dt) * 2 * np.pi  # angular freq grid
    dA_dt = np.fft.ifft(1j * freq * np.fft.fft(A_t))

    N_t = 1j * gamma * (A_t * power_term + (1j / omega0) * dA_dt * power_term)
    return N_t


# =========================
# RK4IP propagation
# =========================
def propagate_RK4IP(A0_t, dz, Nz, beta_w, alpha, gamma, omega0, fr, hR_t, dt):
    """
    RK4 in Interaction Picture for GNLSE.
    beta_w: dispersion in frequency domain including higher orders.
    alpha : loss coefficient (1/m)
    """
    # Precompute dispersion exponential for half-step and full-step
    D_half = np.exp((beta_w - alpha / 2) * dz / 2)
    D_full = np.exp((beta_w - alpha / 2) * dz)

    A_f = np.fft.fft(A0_t)  # start in frequency domain

    for _ in range(Nz):
        # Half-step linear operator
        A_t = np.fft.ifft(D_half * A_f)

        # 4 RK stages
        k1_t = nonlinear_operator(A_t, gamma, omega0, fr, hR_t, dt)
        k2_t = nonlinear_operator(A_t + dz / 2 * k1_t, gamma, omega0, fr, hR_t, dt)
        k3_t = nonlinear_operator(A_t + dz / 2 * k2_t, gamma, omega0, fr, hR_t, dt)
        k4_t = nonlinear_operator(A_t + dz * k3_t, gamma, omega0, fr, hR_t, dt)

        N_t = (k1_t + 2 * k2_t + 2 * k3_t + k4_t) / 6.0

        A_t += dz * N_t
        A_f = np.fft.fft(A_t)

        # Remaining half-step dispersion and loss
        A_f *= D_half

    return np.fft.ifft(A_f)


# =========================
# Example Simulation
# =========================
if __name__ == "__main__":
    # --- Simulation grid ---
    Npts = 2**13  # number of time samples
    T_win = 20e-12  # total time window (s)
    dt = T_win / Npts
    t = np.arange(-Npts / 2, Npts / 2) * dt

    freq = np.fft.fftfreq(Npts, dt)
    omega = 2 * np.pi * freq  # angular frequency grid

    # --- Initial condition: sech pulse ---
    P0 = 2.0  # peak power (W)
    T0 = 50e-15  # pulse width (s)
    lambda0 = 1550e-9  # central wavelength (m)
    c = 299792458.0
    omega0 = 2 * np.pi * c / lambda0

    A0_t = np.sqrt(P0) * 1 / np.cosh(t / T0)

    # --- Fiber parameters ---
    beta2 = -20e-27  # s^2/m
    beta3 = 0.12e-39  # s^3/m
    beta4 = -0.005e-51  # s^4/m
    alpha = 0.0  # loss coefficient (1/m)
    gamma = 1.0  # W^-1 m^-1
    fr = 0.18  # Raman fraction

    # Dispersion operator in frequency domain
    beta_w = (
        0.5j * beta2 * omega**2
        + (1j / 6) * beta3 * omega**3
        + (1j / 24) * beta4 * omega**4
    )

    # Raman response
    hR_t = raman_response(t)

    # --- Propagation ---
    dz = 1e-3  # step size (m)
    Nz = 2000  # number of steps

    Aout_t = propagate_RK4IP(A0_t, dz, Nz, beta_w, alpha, gamma, omega0, fr, hR_t, dt)

    # --- Plot results ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t * 1e12, np.abs(A0_t) ** 2, label="Input")
    plt.plot(t * 1e12, np.abs(Aout_t) ** 2, label="Output")
    plt.xlabel("Time (ps)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.title("Time Domain")

    plt.subplot(1, 2, 2)
    f_shift = np.fft.fftshift(freq)
    spec_in = np.fft.fftshift(np.abs(np.fft.fft(A0_t)) ** 2)
    spec_out = np.fft.fftshift(np.abs(np.fft.fft(Aout_t)) ** 2)
    plt.plot(f_shift * 1e-12, spec_in, label="Input")
    plt.plot(f_shift * 1e-12, spec_out, label="Output")
    plt.xlabel("Frequency (THz offset)")
    plt.ylabel("Spectral Power (a.u.)")
    plt.legend()
    plt.title("Spectrum")

    plt.tight_layout()
    plt.show(block=True)
