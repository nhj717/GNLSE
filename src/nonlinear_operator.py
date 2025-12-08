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
    N = A_t.size
    hR_t_aligned = fft.ifftshift(hR_t).copy()

    M = 2 * N
    HR_f = fft.fft(np.pad(hR_t, (0, M - N), mode="constant"))
    I_f = fft.fft(np.pad(I_t, (0, M - N), mode="constant"))
    conv_R_t = fft.ifft(HR_f * I_f)[:N] * dt
    conv_R_t = np.real(conv_R_t)

    # Instantaneous + delayed term
    S = (1 - fr) * I_t + fr * conv_R_t
    SA_t = S * A_t

    # Self-steepening (shock term)
    dSA_dt = fft.ifft(-1j * omega * fft.fft(SA_t))

    N_mult = 1j * gamma * S
    N_add = (gamma / omega0) * dSA_dt

    return N_mult, N_add, conv_R_t
