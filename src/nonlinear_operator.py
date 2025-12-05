import numpy as np
import scipy.fft as fft


def nonlinear_operator(A_t, gamma, omega0, omega_diff, fr, hR_t, dt):
    """
    Full nonlinear term with self-steepening and Raman.
    FFT convolution is used for Raman term.
    """
    # Instantaneous intensity
    I_t = np.abs(A_t) ** 2

    # Raman convolution (zero-padding to avoid temporal aliasing)
    HR_f = fft.fft(hR_t)
    I_f = fft.fft(I_t)
    conv_R_t = fft.ifft(HR_f * I_f) * dt

    # Instantaneous + delayed term
    S = (1 - fr) * I_t + fr * conv_R_t
    SA_t = S * A_t

    # Self-steepening (shock term)
    dSA_dt = fft.ifft(-1j * omega_diff * fft.fft(SA_t))

    return 1j * gamma * S, 1j * gamma / omega0 * dSA_dt
