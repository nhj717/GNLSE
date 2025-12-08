import numpy as np
from scipy.constants import c
import scipy.fft as fft
import matplotlib.pyplot as plt

from raman_response import hollenbeck_cantrell_hr as R_t

import simulation_methods as methods


class fiber_propagation:
    "set initial values"

    def __init__(self, omega0, dz, z, dt, t, f, omega):

        self.omega0 = omega0  # pump angular frequency in rad Hz
        self.omega = omega
        self.f = f

        self.dz = dz
        self.z = z
        self.dt = dt
        self.t = t

        self.hR_t = R_t(t)  # Raman response

        self.E = np.zeros(
            (np.size(t), np.size(z)), dtype="complex128"
        )  # initial grid for the E field
        self.spectrum = np.zeros(
            np.shape(self.E), dtype="complex128"
        )  # intial grid for the spectrum

    def source(self, shape, P0, T0, C=0):
        self.T0 = T0
        if shape == "gaussian":
            self.E[:, 0] = np.sqrt(P0) * np.exp(-(1 + 1j * C) / 2 * (self.t / T0) ** 2)
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

        if shape == "sech":
            self.E[:, 0] = np.sqrt(P0) / np.cosh(self.t / T0)
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

        if shape == "lorentzian":
            self.E[:, 0] = np.sqrt(P0) * T0 / 2 / np.pi / (self.t**2 + (T0 / 2) ** 2)
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

    def propagate(self, type, alpha, beta, gamma, fr):

        if type == "RK4IP":
            # scheme: Runge Kutta 4th order, Interaction Picture method from Hult
            self.E, self.spectrum = methods.RK4IP(
                alpha,
                beta,
                gamma,
                fr,
                self.omega0,
                self.omega,
                self.hR_t,
                self.dz,
                self.dt,
                self.E,
                self.spectrum,
            )
        elif type == "SSFM_symmetric":
            # scheme: split step fourier method in 2nd order.
            self.E, self.spectrum, self.R_t = methods.SSFM_symmetric(
                alpha,
                beta,
                gamma,
                fr,
                self.omega0,
                self.omega,
                self.hR_t,
                self.dz,
                self.dt,
                self.E,
                self.spectrum,
            )

    def draw(self):
        f = fft.fftshift(self.f) + self.omega0 / 2 / np.pi
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
        tt, zz = np.meshgrid(t, z)
        ll, zz_lamb = np.meshgrid(lambda_axis, z)
        ff, zz_f = np.meshgrid(f, z)

        downsample_factor = 4  # reduces number of points to plot
        tt = tt[::downsample_factor, ::downsample_factor]
        zz = zz[::downsample_factor, ::downsample_factor]
        # ff = ff[::downsample_factor, ::downsample_factor]
        ll = ll[::downsample_factor, ::downsample_factor]
        zz_lamb = zz_lamb[::downsample_factor, ::downsample_factor]
        I = abs(self.E) ** 2
        I = I[::downsample_factor, ::downsample_factor]
        I_log = 10 * np.log10((I + eps) / np.max(I[:, 0] + eps))
        I_log = I_log.T
        spectrum = fft.fftshift(self.spectrum, axes=0)
        S = jacobian * abs(spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, ::downsample_factor]
        S_log = 10 * np.log10((S + eps) / np.max(S[:, 0] + eps))
        S_log = S_log.T

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
            cmap="jet",
            vmin=-40,
            vmax=0,
        )
        # ax2.set_aspect(15)
        ax2.set_title("freq. domain")
        ax2.set_xlim(500, 1200)
        cb2 = plt.colorbar(pcm2, shrink=0.75)

        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, -1]) ** 2, label="after")
        ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        ax3.set_xlabel("T [ps]")
        ax3.legend()
        ax3.set_title("Time domain")

        ax4.plot(ll[0, :], S_log[0, :], label="before")
        ax4.plot(ll[0, :], S_log[-1, :], label="after")
        ax4.set_xlim(500, 1400)
        ax4.set_ylim(-200, 10)
        ax4.legend()
        ax4.set_title("freq. domain")

        plt.show(block=True)

        plt.plot(self.t, self.R_t)
        plt.show(block=True)
