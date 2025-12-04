from math import factorial
import numpy as np


def D_manual_half(
    omega_diff, dz, alpha, beta2=-1e-26, beta3=0, beta4=0, beta5=0, beta6=0, beta7=0
):

    beta_w = 1j * (
        beta2 / factorial(2) * omega_diff**2
        + beta3 / factorial(3) * omega_diff**3
        + beta4 / factorial(4) * omega_diff**4
        + beta5 / factorial(5) * omega_diff**5
        + beta6 / factorial(6) * omega_diff**6
        + beta7 / factorial(7) * omega_diff**7
    )

    D = np.exp((beta_w - alpha / 2) * dz / 2)
    return D


def D_simulation_half(omega_diff, dz, alpha, beta, beta0, beta1):

    beta_w = 1j * (beta - beta0 - beta1 * omega_diff)

    D = np.exp((beta_w - alpha / 2) * dz / 2)
    return D
