from fontTools.misc.psLib import endofthingRE
import numpy as np
import scipy.fft as fft
from scipy.constants import epsilon_0
from scipy.integrate import solve_ivp
from operators import nonlinear_operator as N_op
from operators import linear_operator as L_op
import tqdm

# def report(z,y,flag):
#     status = 0
#     if isempty(flag):
#         print(f'{z/}')
#
#     return status


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
    D_half = np.exp((beta - alpha / 2) * dz / 2)
    for i in range(np.size(E[0, :]) - 1):
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
    D_half = np.exp((beta - alpha / 2) * dz / 2)

    for i in range(np.size(E[0, :]) - 1):
        # Half-step Dispersion
        A_t_i = fft.fft(D_half * spectrum[:, i])

        # Full-step Nonlienar
        N_mult, N_add, R_t = N_op(A_t_i, gamma, omega0, omega, fr, hR_t, dt)
        # N2 = fft.ifft(1j * omega * fft.fft(A_I * N1)) / omega0
        # N2 = -1j * gradient(N1 * A_I, dt) / omega0
        A_t_i = np.exp(N_mult * dz) * A_t_i + 0 * N_add * dz

        # Last half-step Dispersion
        spectrum[:, i + 1] = D_half * fft.ifft(A_t_i)
        E[:, i + 1] = fft.fft(spectrum[:, i + 1])

        if i == 0:
            R = R_t

    return E, spectrum, R


def ODE_Dudley(
    alpha,
    beta,
    gamma,
    fr,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E,
    spectrum,
    atol=1e-4,
    rtol=1e-3,
):
    eps = np.finfo(float).eps
    Nz = len(E[0, :])
    Nt = len(E[:, 0])
    omega_diff = fft.fftshift(omega)
    Z = np.arange(0, Nz) * dz

    R_w = Nt * fft.ifft(fft.fftshift(R_t))
    L, Aeff = L_op(alpha, beta, omega0, omega_diff)
    if Aeff != None:
        gamma = gamma / Aeff
    if abs(omega0) > eps:
        gamma /= omega0
        W = omega_diff + omega0
        W = fft.fftshift(W)
    else:
        W = 1

    L = fft.fftshift(L)
    progress_bar = tqdm.tqdm(total=Z[-1] * 1000, unit="mm")

    def rhs(z, A_w):
        progress_bar.n = round(z * 1000, 3)
        progress_bar.update(0)
        A_t_local = fft.fft(A_w * np.exp(L * z))
        I_t = abs(A_t_local) ** 2
        if len(R_t) == 1 or abs(fr) < eps:
            M = fft.ifft(A_t_local * I_t)
        else:
            RS = dt * fr * fft.fft(fft.ifft(I_t) * R_w)
            M = fft.ifft(A_t_local * ((1 - fr) * I_t + RS))

        R = 1j * gamma * W * M * np.exp(-L * z)
        return R

    sol = solve_ivp(
        lambda t, y: rhs(t, y),
        (Z[0], Z[-1]),
        spectrum[:, 0],
        t_eval=Z,
        atol=atol,
        rtol=rtol,
        method="RK45",
    )

    progress_bar.close()

    # Convert to time domain
    for i in range(Nz):
        AW_tmp = sol.y[:, i] * np.exp(L * Z[i])
        E[:, i] = fft.fft(AW_tmp)
        spectrum[:, i] = fft.ifftshift(AW_tmp) / dt

    return E, spectrum, 0
