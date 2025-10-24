import numpy as np
from scipy.constants import c
import scipy.fft as fft
import matplotlib.pyplot as plt


class fiber_propagation:
    "set initial values"

    def __init__(self, lambda0, P0,T0,C=0,z_tot=0, alpha=0, beta2=-1, beta3=0, s=0, tr=0, gamma=0):

        self.f0 = c / lambda0 * 1e6                                     #pump frequency in Hz
        self.s = s
        self.tr = tr
        self.gamma = gamma                                              #nonlinear coeff from the fiber in W^-1/m
        self.T0 = T0                                                    #period of the pulse in seconds
        self.P0 = P0                                                    #peak power of the pulse
        self.C = C                                                      #chirp of the pulse unitless

        self.z_tot = z_tot                                              #total fiber length in meters
        self.z_steps =2**10                                             # No. of z steps
        self.delz = self.z_tot / self.z_steps
        self.z = np.linspace(0, self.z_tot, self.z_steps + 1)       #z grid for simulation

        self.Tspan = 1000 * T0                                          # total simulation grid for tau
        self.t_steps = 2**12                                            # No. of tau steps
        self.delt = self.Tspan / self.t_steps
        self.t = (
            np.linspace(-self.Tspan/ 2, self.Tspan / 2, self.t_steps+ 1)
        )                                                                 #tau grid for simulation
        self.f= fft.fftfreq(self.t_steps+1,self.delt)                 #freq grid for simulation
        self.omega = 2 * np.pi * self.f  # omega array                    #angular freq
        self.f_D = self.f+self.f0                                         #shifted freq grid for correct dispersion
        self.omega_D = 2*np.pi*self.f_D                                   #shifted angular freq

        try:
            self.alpha = alpha(self.f_D)
        except:
            self.alpha = alpha                  #absorption coeff
        try:
            self.beta2 = beta2(self.f_D)
        except:
            self.beta2 = beta2                  #s^2/m
        try:
            self.beta3 = beta3(self.f_D)
        except:
            self.beta3 = beta3                  #s^3/m

        self.E = np.zeros((self.t_steps + 1, self.z_steps + 1), dtype="complex128")           #initial grid for the E field
        self.spectrum = np.zeros(
            (self.t_steps + 1, self.z_steps + 1), dtype="complex128"
        )                                                                                              #intial grid for the spectrum

    def source(self, shape):

        if shape == "gaussian":
            self.E[:, 0] = np.sqrt(self.P0)*np.exp(-(1+1j*self.C)/2*(self.t/self.T0)**2)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0])

        if shape == "sech":
            self.E[:, 0] = np.sqrt(self.P0) / np.cosh(self.t)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0])

        if shape == "lorentzian":
            T0 = self.T0
            self.E[:, 0] = np.sqrt(self.P0)*T0 / 2 / np.pi / (self.t**2 + (T0 / 2) ** 2)
            self.spectrum[:, 0] = fft.ifft(self.E[:, 0])

    def run(self, shape):
        fiber_propagation.source(self, shape)
        omega = self.omega
        E = self.E
        s = self.s
        tr = self.tr
        gamma = self.gamma
        spectrum = self.spectrum
        dispersion = np.exp(
            0.5 * self.delz* ((1j) * self.beta2/2 * omega**2
            - (1j) * self.beta3/6 * omega**3
            - self.alpha / 2)
        )  # phase factor

        # scheme: 1/2D -> N -> 1/2D first half step nonlinear

        for i in range(self.z_steps):
            #1/2 D step
            spectrum_i = dispersion * spectrum[:, i]
            E_i= fft.fft(spectrum_i)

            #N step
            nonlinear = 1j*gamma*(np.conjugate(E_i)*E_i+(1j*s-tr)*E_i*fft.fft(1j*omega*np.conjugate(spectrum_i))+(2j*s-tr)*np.conjugate(E_i)*fft.fft(1j*omega*spectrum_i))
            E_i+=self.delz*nonlinear

            #1/2 D step
            spectrum_i = dispersion * fft.ifft(E_i)

            #save result
            spectrum[:, i+1] = spectrum_i
            E[:, i+1] = E_i

        self.E = E
        self.spectrum = fft.ifftshift(spectrum, axes=0)

    def draw(self):
        omega = self.omega
        f = (1 / (2 * np.pi)) * fft.ifftshift(omega)
        t, z = self.t, self.z
        tt, zz = np.meshgrid(t, z)
        ff, zz_f = np.meshgrid(f, z)

        vis1 = 10 * np.log10(np.transpose((abs(self.E) / np.max(abs(self.E[:,0]))) ** 2))
        vis2 = 10 * np.log10(
            np.transpose((abs(self.spectrum) / np.max(abs(self.spectrum))) ** 2)
        )

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.suptitle("simulation")

        pcm1 = ax1.pcolor(
            tt[:, index1:index2],
            zz[:, index1:index2],
            vis1[:, index1:index2],
            cmap="jet",
            vmin=-30,
            vmax=0,
        )
        # ax1.set_aspect(30)
        ax1.set_title("Time domain")
        cb1 = plt.colorbar(pcm1, shrink=0.75)

        pcm2 = ax2.pcolor(
            ff[:, index3:index4],
            zz_f[:, index3:index4],
            vis2[:, index3:index4],
            cmap="jet",
            vmin=-40,
            vmax=0,
        )
        # ax2.set_aspect(15)
        ax2.set_title("freq. domain")
        cb2 = plt.colorbar(pcm2, shrink=0.75)

        ax3.plot(tau, abs(self.E[:, 0]) ** 2, label="before")
        ax3.plot(tau, abs(self.E[:, -1]) ** 2, label="after")
        ax3.legend()
        ax3.set_title("Time domain")

        ax4.plot(f, abs(self.spectrum[:, 0]) ** 2, label="before")
        ax4.plot(f, abs(self.spectrum[:, -1]) ** 2, label="after")
        ax4.legend()
        ax4.set_title("freq. domain")

        plt.show(block=True)
