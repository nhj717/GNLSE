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
    dA_dt = fft.ifft(1j * omega * fft.fft(A_t))

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
        beta2=-1e-26,
        beta3=0,
        beta4=0,
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
        self.z = np.linspace(0, self.z_tot, self.z_steps + 1)  # z grid for simulation

        self.Tspan = 100 * T0  # total simulation grid for tau
        self.t_steps = 2**13  # No. of tau steps
        self.delt = self.Tspan / self.t_steps
        self.t = np.linspace(
            -self.Tspan / 2, self.Tspan / 2, self.t_steps + 1
        )  # tau grid for simulation
        self.hR_t = raman_response(self.t)  # Raman response
        self.f = fft.fftfreq(self.t_steps + 1, self.delt)  # freq grid for simulation
        self.omega = 2 * np.pi * self.f  # omega array                    #angular freq
        self.f_D = fft.fftshift(self.f) + self.f0  # shifted freq grid for correct dispersion
        self.omega_D = 2 * np.pi * self.f_D  # shifted angular freq

        try:
            self.alpha = alpha(self.omega_D)
        except:
            self.alpha = alpha  # absorption coeff
        try:
            self.beta2 = beta2(self.omega_D)
        except:
            self.beta2 = beta2  # s^2/m
        try:
            self.beta3 = beta3(self.omega_D)
        except:
            self.beta3 = beta3  # s^3/m
        try:
            self.beta4 = beta4(self.omega_D)
        except:
            self.beta4 = beta4  # s^4/m

        self.fr = 0.18  # Raman coefficient

        self.E = np.zeros(
            (self.t_steps + 1, self.z_steps + 1), dtype="complex128"
        )  # initial grid for the E field
        self.spectrum = np.zeros(
            (self.t_steps + 1, self.z_steps + 1), dtype="complex128"
        )  # intial grid for the spectrum

    def source(self, shape):

        if shape == "gaussian":
            self.E[:, 0] = np.sqrt(self.P0) * np.exp(
                -(1 + 1j * self.C) / 2 * (self.t / self.T0) ** 2
            )
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

        if shape == "sech":
            self.E[:, 0] = np.sqrt(self.P0) / np.cosh(self.t)
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

        if shape == "lorentzian":
            T0 = self.T0
            self.E[:, 0] = (
                np.sqrt(self.P0) * T0 / 2 / np.pi / (self.t**2 + (T0 / 2) ** 2)
            )
            self.spectrum[:, 0] = fft.fft(self.E[:, 0])

    def propagate(self):
        omega0 = self.omega0
        omega = self.omega
        E = self.E
        dz = self.delz
        dt = self.delt
        fr = self.fr
        hR_t = self.hR_t
        gamma = self.gamma
        spectrum = self.spectrum
        omega_b = omega*1e-12
        beta_w = 1j * (
            self.beta2 / 2 * omega_b**2
            - self.beta3 / 6 * omega_b**3
            + self.beta4 / 24 * omega_b**4-(3.032e-10)/120*omega_b**5+(-4.169e-13)*omega_b**6-(2.57e-16)*omega_b**7
        )
        D_half = np.exp((beta_w - self.alpha / 2) * dz / 2)

        # scheme: 1/2D -> N -> 1/2D first half step nonlinear

        for i in range(self.z_steps):
            # Half-step dispersion and loss
            A_I = fft.ifft(D_half * spectrum[:, i])

            # 4 RK stages
            k1 = fft.ifft(
                D_half
                * fft.fft(
                    dz * nonlinear_operator(E[:, i], gamma, omega0, omega, fr, hR_t, dt)
                )
            )
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
        self.spectrum = fft.fftshift(spectrum, axes=0)

    def draw(self):
        f = self.f_D
        lamb = c / f * 1e9
        t, z = self.t, self.z
        zz, tt = np.meshgrid(z, t)
        zz_lamb, ll = np.meshgrid(z, lamb)
        zz_f, ff = np.meshgrid(z, f)

        downsample_factor = 4  # reduces number of points to plot
        tt = tt[::downsample_factor, ::downsample_factor]
        zz = zz[::downsample_factor, ::downsample_factor]
        ff = ff[::downsample_factor, ::downsample_factor]
        ll = ll[::downsample_factor, ::downsample_factor]
        zz_lamb = zz_lamb[::downsample_factor, ::downsample_factor]
        I = ((abs(self.E) / np.max(abs(self.E[:, 0]))) ** 2)[
            ::downsample_factor, ::downsample_factor
        ]
        I_log_safe = np.where(I == 0, 1e-60, I)
        spectrum = ((abs(self.spectrum) / np.max(abs(self.spectrum[:, 0]))))[
            ::downsample_factor, ::downsample_factor
        ]
        spectrum_log_safe = np.where(spectrum == 0, 1e-60, spectrum)

        vis1 = 10 * np.log10(I_log_safe)
        vis2 = 10 * np.log10(spectrum_log_safe)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("simulation")

        pcm1 = ax1.pcolormesh(
            zz,
            tt * 1e12,
            vis1,
            cmap="jet",
            vmin=-30,
            vmax=0,
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        # ax1.set_ylim(-20 * self.T0 * 1e15, 20 * self.T0 * 1e15)
        cb1 = plt.colorbar(pcm1, shrink=0.75)

        pcm2 = ax2.pcolormesh(
            zz_lamb,
            ff,
            vis2,
            cmap="jet",
            vmin=-40,
            vmax=0,
        )
        # ax2.set_aspect(15)
        ax2.set_title("freq. domain")
        # ax2.set_ylim(500, 1200)
        cb2 = plt.colorbar(pcm2, shrink=0.75)

        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(t * 1e12, 1e-9 * abs(self.E[:, -1]) ** 2, label="after")
        ax3.set_xlim(-20 * self.T0 * 1e12, 20 * self.T0 * 1e12)
        ax3.set_xlabel("T [ps]")
        ax3.legend()
        ax3.set_title("Time domain")

        ax4.plot(lamb, abs(self.spectrum[:, 0]) ** 2, label="before")
        ax4.plot(lamb, abs(self.spectrum[:, -1]) ** 2, label="after")
        ax4.set_xlim(500, 1200)
        ax4.legend()
        ax4.set_title("freq. domain")

        plt.show(block=True)
