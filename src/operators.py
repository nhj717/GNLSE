from math import factorial
import numpy as np
import scipy.fft as fft
from scipy.constants import c, pi
import os
from scipy.interpolate import UnivariateSpline
from functions import read_hdf5 as rdhd


def nonlinear_operator(A_t, gamma, omega0, omega, fr, R_t, dt):
    """
    Full nonlinear term with self-steepening and Raman.
    FFT convolution is used for Raman term.
    """
    # Instantaneous intensity
    I_t = np.abs(A_t) ** 2

    # Raman convolution (zero-padding to avoid temporal aliasing)
    N = len(omega)
    R_w = N * fft.ifft(R_t)
    I_w = fft.ifft(I_t)
    # M = fft.next_fast_len(2 * N)
    # R_w = M*fft.ifft(fft.fftshift(R_t), n=M)
    # I_w = M*fft.ifft(fft.fftshift(I_t), n=M)

    # Convolution in frequency domain and back to time; scale by dt
    conv_R_t = fft.fft(R_w * I_w) * dt

    # Shift back to centered time origin and crop to original length
    # conv_R_t = fft.fftshift(conv_R_t)
    # start = (M - N) // 2
    # conv_R_t = np.real(conv_R_t[start : start + N])

    # # Instantaneous + delayed term
    S = (1 - fr) * I_t + fr * conv_R_t
    SA_t = S * A_t

    # Self-steepening (shock term)
    # dSA_dt = fft.fft(omega * fft.ifft(SA_t))
    # dSA_dt = np.where(
    #     abs(A_t) > 1e-15,
    #     fft.fft(omega * fft.ifft(SA_t)) / (A_t + 1e-20),
    #     0.0,
    # )

    return 1j * gamma * S


def linear_operator(alpha, beta, omega0, omega):
    if isinstance(beta, list):
        Aeff_w0 = None
        alpha_w = alpha
        beta_w = 0
        for i in range(len(beta)):
            beta_w += beta[i] / factorial(i + 2) * omega ** (i + 2)

    elif isinstance(beta, str):
        # Fiber informaiton
        folder_path = os.getcwd()
        location = os.path.join(folder_path, "dispersion_data", "waveguide.h5")
        data_label, data = rdhd(location, beta)
        print(data_label)

        # Fiber informaiton
        omega_data = data[data_label.index("omega")]
        beta1 = data[data_label.index("beta1")]
        neff = data[data_label.index("n_eff")]
        Aeff = data[data_label.index("A_eff")]
        n = np.real(neff)
        k = -np.imag(neff)
        alpha = 2 * omega_data * k / c
        beta0 = n * omega_data / c
        alpha_spl = UnivariateSpline(omega_data, alpha, k=5)
        beta0_spl = UnivariateSpline(omega_data, beta0, k=5)
        beta1_spl = UnivariateSpline(omega_data, beta1, k=5)
        Aeff_spl = UnivariateSpline(omega_data, Aeff, k=5)

        alpha_w = alpha_spl(omega)
        beta_w = beta0_spl(omega) - beta0_spl(omega0) - beta1_spl(omega0) * omega
        Aeff_w0 = Aeff_spl(omega0)

    L = 1j * beta_w - alpha_w / 2

    return L, Aeff_w0
