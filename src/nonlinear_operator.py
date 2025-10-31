import numpy as np
import scipy.fft as fft

def nonlinear_operator(A_t, gamma, omega0, omega, fr, hR_t, dt):
    """
    Full nonlinear term with self-steepening and Raman.
    FFT convolution is used for Raman term.
    """
    # Instantaneous intensity
    I_t = np.abs(A_t) ** 2

    # Raman convolution (zero-padding to avoid temporal aliasing)
    H_R_f = fft.fft(hR_t)
    I_f = fft.fft(I_t)
    conv_R_t = fft.ifft(H_R_f*I_f)*dt

    # Self steepening term
    S = (1 - fr) * I_t + fr * conv_R_t


    dSA_dt = fft.ifft(-1j * (omega - omega0) * fft.fft(A_t*S))

    N_t = 1j * gamma * (S+ 1/ omega0 * dSA_dt)
    return N_t

