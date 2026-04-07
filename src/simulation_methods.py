from fontTools.misc.psLib import endofthingRE
import numpy as np
import scipy.fft as fft
from scipy.integrate import solve_ivp
from operators import nonlinear_operator_divide_At as N_op_divide
from operators import nonlinear_operator_seperated as N_op_seperated
from operators import nonlinear_operator as N_op
from operators import linear_operator as L_op
import tqdm


def RK4IP(
    alpha,
    beta,
    gamma,
    fr,
    self_steepening,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E,
    spectrum,
    update,
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    D_half = np.exp(L * dz / 2)
    Nz = len(E[0, :])

    # --- PRECOMPUTE RAMAN VARIABLES ONCE ---
    N = len(omega)
    M = fft.next_fast_len(2 * N)
    R_w = M * fft.ifft(R_t, n=M)
    start = (M - N) // 2
    # ---------------------------------------

    if update is True:
        progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")
    for i in range(Nz - 1):
        if update is True:
            progress_bar.n = round(i * dz * 1000, 3)
            progress_bar.update(0)

        # Half-step Dispersion
        A_I = fft.fft(D_half * spectrum[:, i])

        # 4 RK stages
        k1 = fft.fft(
            D_half
            * fft.ifft(
                dz
                * N_op(
                    E[:, i],
                    gamma,
                    omega0,
                    omega,
                    fr,
                    R_w,
                    M,
                    start,
                    N,
                    dt,
                    self_steepening,
                )
            )
        )

        k2 = dz * N_op(
            A_I + k1 / 2,
            gamma,
            omega0,
            omega,
            fr,
            R_w,
            M,
            start,
            N,
            dt,
            self_steepening,
        )
        k3 = dz * N_op(
            A_I + k2 / 2,
            gamma,
            omega0,
            omega,
            fr,
            R_w,
            M,
            start,
            N,
            dt,
            self_steepening,
        )
        k4 = dz * N_op(
            fft.fft(D_half * fft.ifft(A_I + k3)),
            gamma,
            omega0,
            omega,
            fr,
            R_w,
            M,
            start,
            N,
            dt,
            self_steepening,
        )

        # save result
        E[:, i + 1] = (
            fft.fft(D_half * fft.ifft(A_I + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6
        )
        spectrum[:, i + 1] = fft.ifft(E[:, i + 1])

    return E, spectrum


def RK4IP_vectorized(
    alpha,
    beta,
    gamma,
    fr,
    self_steepening,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E_2D,
    spectrum_2D,
    Nz,
    update,
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff

    # Broadcast linear operator to match 2D shape (Nt, 1)
    L = L[:, None]
    D_half = np.exp(L * dz / 2)

    # Precompute Raman Variables
    N_omega = len(omega)
    M = fft.next_fast_len(2 * N_omega)
    R_w = M * fft.ifft(R_t, n=M)
    R_w = R_w[:, None]  # Broadcast for Raman
    start = (M - N_omega) // 2
    omega_broad = omega[:, None]

    # Internal localized nonlinear operator strictly for 2D arrays
    def N_op_vec(A_t):
        I_t = np.abs(A_t) ** 2
        I_w = fft.ifft(I_t, n=M, axis=0)
        conv_R_t = fft.fft(R_w * I_w, axis=0) * dt
        conv_R_t = conv_R_t[start : start + N_omega]
        S = (1 - fr) * I_t + fr * conv_R_t
        SA_t = S * A_t
        if self_steepening is False:
            return 1j * gamma * SA_t
        else:
            return (
                1j
                * gamma
                * fft.fft((1 + omega_broad / omega0) * fft.ifft(SA_t, axis=0), axis=0)
            )

    if update is True:
        progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")

    for i in range(Nz - 1):
        if update is True:
            progress_bar.n = round(i * dz * 1000, 3)
            progress_bar.update(0)

        # Half-step Dispersion
        A_I = fft.fft(D_half * spectrum_2D, axis=0)

        # 4 RK stages with axis=0 strictly enforced
        k1 = fft.fft(D_half * fft.ifft(dz * N_op_vec(E_2D), axis=0), axis=0)
        k2 = dz * N_op_vec(A_I + k1 / 2)
        k3 = dz * N_op_vec(A_I + k2 / 2)
        k4 = dz * N_op_vec(fft.fft(D_half * fft.ifft(A_I + k3, axis=0), axis=0))

        # Save result for the next spatial step
        E_2D = (
            fft.fft(D_half * fft.ifft(A_I + k1 / 6 + k2 / 3 + k3 / 3, axis=0), axis=0)
            + k4 / 6
        )
        spectrum_2D = fft.ifft(E_2D, axis=0)

    # By the end of the loop, E_2D holds the final pulse for all P values
    return E_2D, spectrum_2D


def SSFM_Agrawal(
    alpha,
    beta,
    gamma,
    fr,
    self_steepening,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E,
    spectrum,
    update,
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    D_half = np.exp(L * dz / 2)
    Nz = len(E[0, :])

    # --- PRECOMPUTE RAMAN VARIABLES ONCE ---
    N = len(omega)
    M = fft.next_fast_len(2 * N)
    R_w = M * fft.ifft(R_t, n=M)
    start = (M - N) // 2
    # ---------------------------------------

    if update is True:
        progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")
    for i in range(Nz - 1):
        if update is True:
            progress_bar.n = round(i * dz * 1000, 3)
            progress_bar.update(0)
        # Half-step Dispersion
        A_t_i = fft.fft(D_half * spectrum[:, i])

        # Full-step Nonlienar
        N_op_val = N_op_divide(
            A_t_i, gamma, omega0, omega, fr, R_w, M, start, N, dt, self_steepening
        )
        A_t_i *= np.exp(N_op_val * dz)

        # Last half-step Dispersion
        spectrum[:, i + 1] = D_half * fft.ifft(A_t_i)
        E[:, i + 1] = fft.fft(spectrum[:, i + 1])

    return E, spectrum


def SSFM_Vishal(
    alpha,
    beta,
    gamma,
    fr,
    self_steepening,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E,
    spectrum,
    update,
):
    L, Aeff = L_op(alpha, beta, omega0, omega)
    if Aeff is not None:
        gamma = gamma / Aeff
    D_half = np.exp(L * dz / 2)
    Nz = len(E[0, :])

    # --- PRECOMPUTE RAMAN VARIABLES ONCE ---
    N = len(omega)
    M = fft.next_fast_len(2 * N)
    R_w = M * fft.ifft(R_t, n=M)
    start = (M - N) // 2
    # ---------------------------------------

    if update is True:
        progress_bar = tqdm.tqdm(total=Nz * dz * 1000, unit="mm")
    for i in range(Nz - 1):
        if update is True:
            progress_bar.n = round(i * dz * 1000, 3)
            progress_bar.update(0)
        # Half-step Dispersion
        A_t_i = fft.fft(D_half * spectrum[:, i])

        # Full-step Nonlienar
        N_mult, N_add = N_op_seperated(
            A_t_i, gamma, omega0, omega, fr, R_w, M, start, N, dt, self_steepening
        )
        A_t_i *= np.exp(N_mult * dz)
        A_t_i += N_add * dz

        # Last half-step Dispersion
        spectrum[:, i + 1] = D_half * fft.ifft(A_t_i)
        E[:, i + 1] = fft.fft(spectrum[:, i + 1])

    return E, spectrum


def ODE_Dudley(
    alpha,
    beta,
    gamma,
    fr,
    self_steepening,
    omega0,
    omega,
    R_t,
    dz,
    dt,
    E,
    spectrum,
    update,
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

    if update is True:
        progress_bar = tqdm.tqdm(total=Z[-1] * 1000, unit="mm")

    def rhs(z, A_w):
        if update is True:
            progress_bar.n = round(i * dz * 1000, 3)
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

    # Convert to time domain
    spectrum = sol.y * np.exp(np.outer(L, Z))
    E = fft.fft(spectrum, axis=0)

    return E, spectrum
