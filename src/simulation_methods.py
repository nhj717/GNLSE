from numpy import size, exp
import scipy.fft as fft
from nonlinear_operator import nonlinear_operator as N_op


def RK4IP(
    alpha,
    beta,
    gamma,
    omega0,
    omega_diff,
    fr,
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
            D_half
            * fft.fft(dz * N_op(E[:, i], gamma, omega0, omega_diff, fr, hR_t, dt))
        )

        k2 = dz * N_op(A_I + k1 / 2, gamma, omega0, omega_diff, fr, hR_t, dt)
        k3 = dz * N_op(A_I + k2 / 2, gamma, omega0, omega_diff, fr, hR_t, dt)
        k4 = dz * N_op(
            fft.ifft(D_half * fft.fft(A_I + k3)),
            gamma,
            omega0,
            omega_diff,
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
