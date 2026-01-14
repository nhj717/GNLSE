import numpy as np
from scipy.constants import c
import scipy.fft as fft
import matplotlib.pyplot as plt
from raman_response import hollenbeck_cantrell_hr as R_t
import simulation_methods as methods
import tqdm


class fiber_propagation:
    "set initial values"

    def __init__(self, omega0, dz, z, dt, t, f, omega, pulse_shape, P0, T0, C):

        self.P = None
        self.pulse_shape = pulse_shape
        self.P0 = P0
        self.T0 = T0
        self.C = C

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

    def source(self, P0=None):
        pulse_shape = self.pulse_shape
        T0 = self.T0
        C = self.C
        if P0 is None:
            P0 = self.P0

        if pulse_shape == "gaussian":
            self.E[:, 0] = np.sqrt(P0) * np.exp(-1 / 2 * (self.t / T0) ** 2)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0]) * np.exp(
                1j / 2 * C * self.omega**2
            )
            self.E[:, 0] = fft.fft(self.spectrum[:, 0])

        if pulse_shape == "sech":
            self.E[:, 0] = np.sqrt(P0) / np.cosh(self.t / T0)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0])

        if pulse_shape == "lorentzian":
            self.E[:, 0] = np.sqrt(P0) * T0 / 2 / np.pi / (self.t**2 + (T0 / 2) ** 2)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0])

    def propagate(self, type, alpha, beta, gamma, fr, self_steepening, update=True):
        self.sim_type = type
        if type == "RK4IP":
            fun = methods.RK4IP
            # scheme: Runge Kutta 4th order, Interaction Picture method from Hult
        elif type == "SSFM_Agrawal":
            # scheme: split step fourier method in 2nd order.
            fun = methods.SSFM_Agrawal
        elif type == "SSFM_Vishal":
            # scheme: split step fourier method in 2nd order.
            fun = methods.SSFM_Vishal
        elif type == "ODE_Dudley":
            fun = methods.ODE_Dudley

        self.E, self.spectrum = fun(
            alpha,
            beta,
            gamma,
            fr,
            self_steepening,
            self.omega0,
            self.omega,
            self.hR_t,
            self.dz,
            self.dt,
            self.E,
            self.spectrum,
            update,
        )

    def draw_z(self, wl_range=[500, 1200], v_range=[-40, 0]):
        print("Now plotting...")
        f = fft.ifftshift(self.f) + self.omega0 / 2 / np.pi
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
        I_log = 10 * np.log10((I + eps) / np.max(I + eps))
        I_log = I_log.T
        spectrum = fft.ifftshift(self.spectrum, axes=0)
        S = jacobian * abs(spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, ::downsample_factor]
        S_log = 10 * np.log10((S + eps) / np.max(S + eps))
        S_log = S_log.T

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1}
        )
        fig.suptitle(f"Simulation with {self.sim_type} method")

        pcm1 = ax1.pcolormesh(
            tt * 1e12,
            zz * 1e2,
            I_log,
            cmap="jet",
            vmin=v_range[0],
            vmax=v_range[1],
            shading="gouraud",
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        ax1.set_ylabel("Distance [cm]")
        ax1.set_xlim(-20 * self.T0 * 1e12, t[-1] * 1e12)
        cb1 = plt.colorbar(pcm1, shrink=1, location="bottom")

        pcm2 = ax2.pcolormesh(
            ll,
            zz_lamb * 1e2,
            S_log,
            cmap="jet",
            vmin=v_range[0],
            vmax=v_range[1],
            shading="gouraud",
        )
        # ax2.set_aspect(15)
        ax2.set_title("Freq. domain")
        ax2.set_xlim(wl_range[0], wl_range[1])
        cb2 = plt.colorbar(pcm2, shrink=1, location="bottom")

        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, -1]) ** 2, label="after")
        ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        ax3.set_xlabel("Time [ps]")
        ax3.set_ylabel("Intensity")
        ax3.legend()

        ax4.plot(ll[0, :], S_log[0, :], label="before")
        ax4.plot(ll[0, :], S_log[-1, :], label="after")
        ax4.set_xlim(wl_range[0], wl_range[1])
        ax4.set_xlabel("Wavelength [nm]")
        ax4.set_ylim(-200, 10)
        ax4.legend()

        plt.show(block=True)

    def propagate_P(self, simulation_type, alpha, beta, gamma, fr, self_steepening, P):
        Np = len(P)
        self.P = P
        dP = P[-1] / Np
        self.E_P = np.zeros(
            (np.size(self.t), Np), dtype="complex128"
        )  # initial grid for the E field
        self.spectrum_P = np.zeros_like(
            self.E_P, dtype="complex128"
        )  # intial grid for the spectrum
        progress_bar = tqdm.tqdm(total=Np * dP * 1000, unit="mW")
        for i in range(Np):
            progress_bar.n = round(i * dP * 1000, 3)
            progress_bar.update(0)
            fiber_propagation.source(self, P[i])
            fiber_propagation.propagate(
                self,
                simulation_type,
                alpha,
                beta,
                gamma,
                fr,
                self_steepening,
                update=False,
            )
            self.E_P[:, i] = self.E[:, -1]
            self.spectrum_P[:, i] = self.spectrum[:, -1]

    def draw_P(self, R_R, wl_range=[500, 1200], v_range=[-40, 0]):
        print("Now plotting...")
        f = fft.ifftshift(self.f) + self.omega0 / 2 / np.pi
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

        t, p = self.t, np.sqrt(np.pi) * self.T0 * R_R * self.P
        tt, pp = np.meshgrid(t, p)
        ll, pp_lamb = np.meshgrid(lambda_axis, p)
        ff, pp_f = np.meshgrid(f, p)

        downsample_factor = 4  # reduces number of points to plot
        tt = tt[:, ::downsample_factor]
        pp = pp[:, ::downsample_factor]
        # ff = ff[::downsample_factor, ::downsample_factor]
        ll = ll[:, ::downsample_factor]
        pp_lamb = pp_lamb[:, ::downsample_factor]
        I = abs(self.E_P) ** 2
        I = I[::downsample_factor, :]
        I = (I + eps) / np.amax(I + eps, axis=0)[None, :]
        I_log = 10 * np.log10(I)
        I_log = I_log.T
        I = I.T
        spectrum = fft.ifftshift(self.spectrum_P, axes=0)
        S = jacobian * abs(spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, :]
        S = (S + eps) / np.amax(S + eps, axis=0)[None, :]
        S_log = 10 * np.log10(S)
        # S_log = 10 * np.log10((S + eps) / np.max(S))
        S_log = S_log.T
        S = S.T

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 8))
        fig.suptitle(f"Simulation with {self.sim_type} method")

        pcm1 = ax1.pcolormesh(
            tt * 1e12,
            pp,
            I,
            cmap="jet",
            vmin=v_range[0],
            vmax=v_range[1],
            shading="gouraud",
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        ax1.set_ylabel("Pump Power [W]")
        ax1.set_xlim(-20 * self.T0 * 1e12, t[-1] * 1e12)
        # cb1 = plt.colorbar(pcm1, shrink=1, location="bottom")

        pcm2 = ax2.pcolormesh(
            ll,
            pp_lamb,
            S,
            cmap="jet",
            vmin=v_range[0],
            vmax=v_range[1],
            shading="gouraud",
        )
        # ax2.set_aspect(15)
        ax2.set_title("Freq. domain")
        ax2.set_xlim(wl_range[0], wl_range[1])
        cb2 = plt.colorbar(pcm2, shrink=1)

        # ax3.plot(t * 1e12, 1e-9 * abs(self.E_P[:, 0]) ** 2/np.max(abs(self.E_P[:, 0]) ** 2), label="before")
        # ax3.plot(t * 1e12, 1e-9 * abs(self.E_P[:, -1]) ** 2/np.max(abs(self.E_P[:, 0]) ** 2), label="after")
        # ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        # ax3.set_xlabel("Time [ps]")
        # ax3.set_ylabel("Intensity")
        # ax3.legend()
        #
        # ax4.plot(ll[0, :], S_log[0, :], label="before")
        # ax4.plot(ll[0, :], S_log[-1, :], label="after")
        # ax4.set_xlim(wl_range[0], wl_range[1])
        # ax4.set_xlabel("Wavelength [nm]")
        # ax4.set_ylim(-200, 10)
        # ax4.legend()

        plt.show(block=True)

    def draw_P_horizontal(self, R_R, wl_range=[500, 1200], v_range=[-40, 0]):
        print("Now plotting...")
        f = fft.ifftshift(self.f) + self.omega0 / 2 / np.pi
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
        p = np.sqrt(np.pi) * self.T0 * R_R * self.P
        pp_lamb, ll = np.meshgrid(p, lambda_axis)

        downsample_factor = 1  # reduces number of points to plot
        ll = ll[::downsample_factor, :]
        pp_lamb = pp_lamb[::downsample_factor, :]
        I = abs(self.E_P) ** 2
        I = I[::downsample_factor, :]
        I = (I + eps) / np.amax(I + eps, axis=0)[None, :]
        I_log = 10 * np.log10(I)
        spectrum = fft.ifftshift(self.spectrum_P, axes=0)
        S = jacobian * abs(spectrum[mask_pos, :]) ** 2
        S = S[::downsample_factor, :]
        S = (S + eps) / np.amax(S + eps, axis=0)[None, :]
        S_log = 10 * np.log10(S)
        # S_log = 10 * np.log10((S + eps) / np.max(S))

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Simulation with {self.sim_type} method")

        pcm = ax.pcolormesh(
            pp_lamb,
            ll,
            S,
            cmap="jet",
            vmin=v_range[0],
            vmax=v_range[1],
        )
        # ax1.set_aspect(30)
        ax.set_xlabel("Pump Power [W]")
        ax.set_ylabel("Wavelength [nm]")
        ax.set_ylim(wl_range[0], wl_range[1])
        cb = plt.colorbar(pcm)
        plt.tight_layout()
        plt.show(block=True)
