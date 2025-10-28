import numpy as np
from scipy.constants import c
import scipy.fft as fft
import matplotlib.pyplot as plt


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
    h /= np.trapezoid(h, t)
    return h


def nonlinear_operator(A_t, gamma, omega0, omega, fr, hR_t, dt):
    """
    Full nonlinear term with self-steepening and Raman.
    FFT convolution is used for Raman term.
    """
    # Instantaneous intensity
    I_t = np.abs(A_t) ** 2

    # Raman convolution (zero-padding to avoid temporal aliasing)
    padN = len(A_t)
    H_R_f = fft.fft(hR_t)
    I_f = fft.fft(I_t)
    conv_R_t = fft.ifft(H_R_f * I_f) * dt

    # Instantaneous + delayed term
    power_term = (1 - fr) * I_t + fr * conv_R_t.real

    # Self-steepening (shock term)
    dA_dt = fft.ifft(1j * (omega - omega0) * fft.fft(A_t))

    N_t = 1j * gamma * (A_t * power_term + (1j / omega0) * dA_dt * power_term)
    return N_t


class fiber_propagation:
    "set initial values"

    def __init__(
        self,
        lambda0,
        P0,
        T0,
        z_tot=0,
        C=0,
        alpha=0,
        beta=0,
        beta1=0,
        gamma=0,
    ):

        self.f0 = c / lambda0  # pump frequency in Hz
        self.omega0 = 2 * np.pi * self.f0
        self.gamma = gamma  # nonlinear coeff from the fiber in W^-1/m
        self.T0 = T0  # period of the pulse in seconds
        self.P0 = P0  # peak power of the pulse
        self.C = C  # chirp of the pulse unitless

        self.z_tot = z_tot  # total fiber length in meters
        self.z_steps = 2**10  # No. of z steps
        self.delz = self.z_tot / self.z_steps
        self.z = np.arange(0, self.z_steps) * self.delz  # z grid for simulation

        self.Tspan = 50 * T0  # total simulation grid for tau
        self.t_steps = 2**12  # No. of tau steps
        self.delt = self.Tspan / self.t_steps
        self.t = (
            np.arange(-self.t_steps / 2, self.t_steps / 2) * self.delt
        )  # tau grid for simulations
        self.hR_t = raman_response(self.t)  # Raman response
        self.f = fft.fftfreq(self.t_steps, self.delt)  # freq grid for simulation
        self.omega = 2 * np.pi * self.f  # omega array                    #angular freq

        try:
            self.alpha = alpha(self.omega)
        except:
            self.alpha = alpha  # absorption coeff

        self.beta = beta(self.omega)
        self.beta0 = beta(self.omega0)
        self.beta1 = beta1(self.omega)

        self.fr = 0.18  # Raman coefficient

    def source(self, shape):

        t, p = np.meshgrid(self.t, self.P0)

        if shape == "gaussian":
            self.E_i = (
                np.sqrt(p)
                * np.exp(-(1 + 1j * self.C) / 2 * (t / self.T0) ** 2)
                * np.exp(1j * self.omega0 * t)
            )
            self.spectrum_i = fft.fft(self.E_i)

        if shape == "sech":
            self.E_i = np.sqrt(p) / np.cosh(t / self.T0) * np.exp(1j * self.omega0 * t)
            self.spectrum_i = fft.fft(self.E_i)

        if shape == "lorentzian":
            T0 = self.T0
            self.E_i = (
                np.sqrt(p)
                * T0
                / 2
                / np.pi
                / (t**2 + (T0 / 2) ** 2)
                * np.exp(1j * self.omega0 * t)
            )
            self.spectrum_i = fft.fft(self.E_i)

    def propagate_z(self):
        omega0 = self.omega0
        omega = self.omega
        dz = self.delz
        dt = self.delt
        fr = self.fr
        hR_t = self.hR_t
        gamma = self.gamma
        beta_w = 1j * (self.beta - self.beta0 - self.beta1 * (omega - omega0))
        # beta_w = 1j * (self.beta)
        D_half = np.exp((beta_w - self.alpha / 2) * dz / 2)

        E = np.zeros(
            (self.t_steps, self.z_steps), dtype="complex128"
        )  # initial grid for the E field
        spectrum = np.zeros(
            (self.t_steps, self.z_steps), dtype="complex128"
        )  # intial grid for the spectrum

        E[:, 0] = self.E_i
        spectrum[:, 0] = self.spectrum_i

        # scheme: 1/2D -> N -> 1/2D first half step nonlinear

        for i in range(self.z_steps - 1):
            # Half-step dispersion and loss
            A_I = fft.ifft(D_half * spectrum[:, i])

            # 4 RK stages
            k1 = fft.ifft(
                D_half
                * fft.fft(
                    dz * nonlinear_operator(E[:, i], gamma, omega0, omega, fr, hR_t, dt)
                )
            )
            # k1 = dz * nonlinear_operator(A_I, gamma, omega0, omega, fr, hR_t, dt)

            k2 = dz * nonlinear_operator(
                A_I + k1 / 2, gamma, omega0, omega, fr, hR_t, dt
            )
            k3 = dz * nonlinear_operator(
                A_I + k2 / 2, gamma, omega0, omega, fr, hR_t, dt
            )
            k4 = dz * nonlinear_operator(
                fft.ifft(D_half * fft.fft(A_I + k3)), gamma, omega0, omega, fr, hR_t, dt
            )

            # save result
            E[:, i + 1] = (
                fft.ifft(D_half * fft.fft(A_I + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6
            )
            spectrum[:, i + 1] = fft.fft(E[:, i + 1])

        self.E = E
        self.spectrum = spectrum

    def propagate_P(self):
        omega0 = self.omega0
        omega = self.omega
        dz = self.delz
        dt = self.delt
        fr = self.fr
        hR_t = self.hR_t
        gamma = self.gamma
        beta_w = 1j * (self.beta - self.beta0 - self.beta1 * (omega - omega0))
        # beta_w = 1j * (self.beta)
        D_half = np.exp((beta_w - self.alpha / 2) * dz / 2)

        E = self.E_i
        spectrum = self.spectrum_i

        # scheme: 1/2D -> N -> 1/2D first half step nonlinear

        for i in range(self.z_steps - 1):
            # Half-step dispersion and loss
            A_I = fft.ifft(D_half * spectrum[:, i])

            # 4 RK stages
            k1 = fft.ifft(
                D_half
                * fft.fft(
                    dz * nonlinear_operator(E[:, i], gamma, omega0, omega, fr, hR_t, dt)
                )
            )
            # k1 = dz * nonlinear_operator(A_I, gamma, omega0, omega, fr, hR_t, dt)

            k2 = dz * nonlinear_operator(
                A_I + k1 / 2, gamma, omega0, omega, fr, hR_t, dt
            )
            k3 = dz * nonlinear_operator(
                A_I + k2 / 2, gamma, omega0, omega, fr, hR_t, dt
            )
            k4 = dz * nonlinear_operator(
                fft.ifft(D_half * fft.fft(A_I + k3)), gamma, omega0, omega, fr, hR_t, dt
            )

            # save result
            E[:, i + 1] = (
                fft.ifft(D_half * fft.fft(A_I + k1 / 6 + k2 / 3 + k3 / 3)) + k4 / 6
            )
            spectrum[:, i + 1] = fft.fft(E[:, i + 1])

        self.E = E
        self.spectrum = spectrum

    def draw_z(self):
        f = self.f
        mask_pos = f > 0
        f = f[mask_pos]
        lamb = c / f
        # sort_idx = np.argsort(lamb)
        # lambda_axis = lamb[sort_idx] * 1e9
        lambda_axis = lamb * 1e9
        # ---- factor to preserve energy conservation
        jacobian = c / lamb**2
        jacobian = np.expand_dims(jacobian, axis=-1)

        # ---- Small floor to avoid log(0)
        eps = np.finfo(float).eps

        t, z = self.t, self.z
        zz, tt = np.meshgrid(z, t)
        zz_lamb, ll = np.meshgrid(z, lambda_axis)
        zz_f, ff = np.meshgrid(z, f)

        downsample_factor = 4  # reduces number of points to plot
        tt = tt[::downsample_factor, ::downsample_factor]
        zz = zz[::downsample_factor, ::downsample_factor]
        ff = ff[::downsample_factor, ::downsample_factor]
        ll = ll[::downsample_factor, ::downsample_factor]
        zz_lamb = zz_lamb[::downsample_factor, ::downsample_factor]
        I = abs(self.E) ** 2
        I = I[::downsample_factor, ::downsample_factor]
        I_log = 10 * np.log10((I + eps) / np.max(I[:, 0] + eps))
        I_log = I_log
        S = jacobian * abs(self.spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, ::downsample_factor]
        S_log = 10 * np.log10((S + eps) / np.max(S[:, 0] + eps))
        S_log = S_log

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("simulation")

        pcm1 = ax1.pcolormesh(
            tt * 1e12,
            zz,
            I_log,
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        # ax1.set_ylim(-20 * self.T0 * 1e15, 20 * self.T0 * 1e15)
        cb1 = plt.colorbar(pcm1, shrink=0.75)

        pcm2 = ax2.pcolormesh(
            ll,
            zz_lamb,
            S_log,
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
        # ax2.set_aspect(15)
        ax2.set_title("freq. domain")
        ax2.set_xlim(735, 1095)
        cb2 = plt.colorbar(pcm2, shrink=0.75)

        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, -1]) ** 2, label="after")
        ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        ax3.set_xlabel("T [ps]")
        ax3.legend()
        ax3.set_title("Time domain")

        ax4.plot(ll[0, :], S_log[0, :], label="before")
        ax4.plot(ll[0, :], S_log[-1, :], label="after")
        ax4.set_xlim(735, 1095)
        ax4.set_ylim(-200, 10)
        ax4.legend()
        ax4.set_title("freq. domain")

        plt.show(block=True)

    def draw_P(self):
        f = self.f
        mask_pos = f > 0
        f = f[mask_pos]
        lamb = c / f
        # sort_idx = np.argsort(lamb)
        # lambda_axis = lamb[sort_idx] * 1e9
        lambda_axis = lamb * 1e9
        # ---- factor to preserve energy conservation
        jacobian = c / lamb**2
        jacobian = np.expand_dims(jacobian, axis=-1)

        # ---- Small floor to avoid log(0)
        eps = np.finfo(float).eps

        t, z = self.t, self.z
        zz, tt = np.meshgrid(z, t)
        zz_lamb, ll = np.meshgrid(z, lambda_axis)
        zz_f, ff = np.meshgrid(z, f)

        downsample_factor = 4  # reduces number of points to plot
        tt = tt[::downsample_factor, ::downsample_factor]
        zz = zz[::downsample_factor, ::downsample_factor]
        ff = ff[::downsample_factor, ::downsample_factor]
        ll = ll[::downsample_factor, ::downsample_factor]
        zz_lamb = zz_lamb[::downsample_factor, ::downsample_factor]
        I = abs(self.E) ** 2
        I = I[::downsample_factor, ::downsample_factor]
        I_log = 10 * np.log10((I + eps) / np.max(I[:, 0] + eps))
        I_log = I_log
        S = jacobian * abs(self.spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, ::downsample_factor]
        S_log = 10 * np.log10((S + eps) / np.max(S[:, 0] + eps))
        S_log = S_log

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("simulation")

        pcm1 = ax1.pcolormesh(
            tt * 1e12,
            zz,
            I_log,
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        # ax1.set_ylim(-20 * self.T0 * 1e15, 20 * self.T0 * 1e15)
        cb1 = plt.colorbar(pcm1, shrink=0.75)

        pcm2 = ax2.pcolormesh(
            ll,
            zz_lamb,
            S_log,
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
        # ax2.set_aspect(15)
        ax2.set_title("freq. domain")
        ax2.set_xlim(735, 1095)
        cb2 = plt.colorbar(pcm2, shrink=0.75)

        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, -1]) ** 2, label="after")
        ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        ax3.set_xlabel("T [ps]")
        ax3.legend()
        ax3.set_title("Time domain")

        ax4.plot(ll[0, :], S_log[0, :], label="before")
        ax4.plot(ll[0, :], S_log[-1, :], label="after")
        ax4.set_xlim(735, 1095)
        ax4.set_ylim(-200, 10)
        ax4.legend()
        ax4.set_title("freq. domain")

        plt.show(block=True)
