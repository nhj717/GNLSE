import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

omega0 = 2 * np.pi * 2

N = 1000
dt = 10 / N
t = np.arange(-N / 2, N / 2) * dt
f = fft.fftfreq(N, dt)
omega = 2 * np.pi * f
omega_diff = omega - omega0


y = np.sin(omega0 * t) + np.sin(3 * omega0 * t)
answer = omega0 * np.cos(omega0 * t)
dy_dt = np.gradient(y, dt)
dy_dt_f = fft.ifft(1j * omega_diff * fft.fft(y))

# plt.plot(t, answer, label="answer")
plt.plot(t, dy_dt, label="gradient")
plt.plot(t, np.real(dy_dt_f), label="fourier")
plt.show(block=True)
