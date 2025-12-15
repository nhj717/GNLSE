from fontTools.misc.psLib import endofthingRE
import numpy as np
import scipy.fft as fft
from scipy.integrate import solve_ivp
from operators import nonlinear_operator_divide_At as N_op_divide
from operators import nonlinear_operator as N_op
from operators import linear_operator as L_op
import tqdm


def RK4IP(
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
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    D_half = np.exp(L * dz / 2)
    Nz = len(E[0, :])

    progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")
    for i in range(Nz - 1):
        progress_bar.n = round(i * dz * 1000, 3)
        progress_bar.update(0)

        # Half-step Dispersion
        A_I = fft.fft(D_half * spectrum[:, i])
        # 4 RK stages
        k1 = fft.fft(
            D_half * fft.ifft(dz * N_op(E[:, i], gamma, omega0, omega, fr, R_t, dt))
        )

        k2 = dz * N_op(A_I + k1 / 2, gamma, omega0, omega, fr, R_t, dt)
        k3 = dz * N_op(A_I + k2 / 2, gamma, omega0, omega, fr, R_t, dt)
        k4 = dz * N_op(
            fft.fft(D_half * fft.ifft(A_I + k3)),
            gamma,
            omega0,
            omega,
            fr,
            R_t,
            dt,
        )

        # save result
        E[:, i + 1] = (
            fft.fft(D_half * fft.ifft(A_I + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6
        )
        spectrum[:, i + 1] = fft.ifft(E[:, i + 1])

    return E, spectrum


def SSFM_symmetric(
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
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    D_half = np.exp(L * dz / 2)
    Nz = len(E[0, :])

    progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")
    for i in range(Nz - 1):
        progress_bar.n = round(i * dz * 1000, 3)
        progress_bar.update(0)
        # Half-step Dispersion
        A_t_i = fft.fft(D_half * spectrum[:, i])

        # Full-step Nonlienar
        N = N_op_divide(A_t_i, gamma, omega0, omega, fr, R_t, dt)
        A_t_i *= np.exp(N * dz)

        # Last half-step Dispersion
        spectrum[:, i + 1] = D_half * fft.ifft(A_t_i)
        E[:, i + 1] = fft.fft(spectrum[:, i + 1])

    return E, spectrum, 0


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
    Z = np.arange(0, Nz) * dz

    # Raman convolution (zero-padding to avoid temporal aliasing)
    # Nt is there to cancel the 1/N normalization of ifft, effectively making R_w equal to an unnormalized forward transform of R_t (modulo the shift).
    M = fft.next_fast_len(2 * Nt)
    start = (M - Nt) // 2
    R_w = M * fft.ifft(R_t, n=M)

    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    if abs(omega0) > eps:
        gamma /= omega0
        W = omega + omega0
    else:
        W = 1

    progress_bar = tqdm.tqdm(total=Z[-1] * 1000, unit="mm")

    def rhs(z, A_w):
        progress_bar.n = round(z * 1000, 3)
        progress_bar.update(0)
        A_t_local = fft.fft(A_w * np.exp(L * z))
        I_t = abs(A_t_local) ** 2
        if len(R_t) == 1 or abs(fr) < eps:
            SA_t = fft.ifft(A_t_local * I_t)
        else:
            conv_R_t = dt * fft.fft(fft.ifft(I_t, n=M) * R_w)
            conv_R_t = np.real(conv_R_t[start : start + Nt])
            SA_t = fft.ifft(A_t_local * ((1 - fr) * I_t + fr * conv_R_t))

        R = 1j * gamma * W * SA_t * np.exp(-L * z)
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
    spectrum = sol.y * np.exp(np.outer(L, Z))
    E = fft.fft(spectrum, axis=0)

    return E, spectrum, 0
