from numpy import size, exp, gradient
import scipy.fft as fft
from nonlinear_operator import nonlinear_operator as N_op


def RK4IP(
    alpha,
    beta,
    gamma,
    fr,
    omega0,
    omega,
    hR_t,
    dz,
    dt,
    E,
    spectrum,
):
    D_half = exp((beta - alpha / 2) * dz / 2)
    for i in range(size(E[0, :]) - 1):
        # Half-step dispersion and loss

        A_I = fft.ifft(D_half * spectrum[:, i])

        # 4 RK stages
        k1 = fft.ifft(
            D_half * fft.fft(dz * N_op(E[:, i], gamma, omega0, omega, fr, hR_t, dt))
        )

        k2 = dz * N_op(A_I + k1 / 2, gamma, omega0, omega, fr, hR_t, dt)
        k3 = dz * N_op(A_I + k2 / 2, gamma, omega0, omega, fr, hR_t, dt)
        k4 = dz * N_op(
            fft.ifft(D_half * fft.fft(A_I + k3)),
            gamma,
            omega0,
            omega,
            fr,
            hR_t,
            dt,
        )

        # save result
        E[:, i + 1] = (
            fft.ifft(D_half * fft.fft(A_I + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6
        )
        spectrum[:, i + 1] = fft.fft(E[:, i + 1])

    return E, spectrum


def SSFM_symmetric(
    alpha,
    beta,
    gamma,
    fr,
    omega0,
    omega,
    hR_t,
    dz,
    dt,
    E,
    spectrum,
):
    D_half = exp((beta - alpha / 2) * dz / 2)

    for i in range(size(E[0, :]) - 1):
        # Half-step Dispersion
        A_t_i = fft.ifft(D_half * spectrum[:, i])

        # Full-step Nonlienar
        N_mult, N_add, R_t = N_op(A_t_i, gamma, omega0, omega, fr, hR_t, dt)
        # N2 = fft.ifft(1j * omega * fft.fft(A_I * N1)) / omega0
        # N2 = -1j * gradient(N1 * A_I, dt) / omega0
        A_t_i = exp(N_mult * dz) * A_t_i +N_add * dz

        # Last half-step Dispersion
        spectrum[:, i + 1] = D_half * fft.fft(A_t_i)
        E[:, i + 1] = fft.ifft(spectrum[:, i + 1])

        if i == 0:
            R = R_t

    return E, spectrum, R
