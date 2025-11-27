import numpy as np

# Hollenbeck-Cantrell parameters (from py-fmas summary of HC2002)
# Columns: omega_n (rad/fs), A_n (relative amp), gamma_n (1/ps), Gamma_n (1/ps)
_HC_TABLE = np.array(
    [
        [0.01060, 1.00, 1.64, 4.91],
        [0.01884, 11.40, 3.66, 10.40],
        [0.04356, 36.67, 5.49, 16.48],
        [0.06828, 67.67, 5.10, 15.30],
        [0.08721, 74.00, 4.25, 12.75],
        [0.09362, 4.50, 0.77, 2.31],
        [0.11518, 6.80, 1.30, 3.91],
        [0.13029, 4.60, 4.87, 14.60],
        [0.14950, 4.20, 1.87, 5.60],
        [0.15728, 4.50, 2.02, 6.06],
        [0.17518, 2.70, 4.71, 14.13],
        [0.20343, 3.10, 2.86, 8.57],
        [0.22886, 3.00, 5.02, 15.07],
    ]
)


def hollenbeck_cantrell_hr(t, fr=0.18, normalize=True):
    global _HC_TABLE
    """
    Build the Hollenbeck-Cantrell Raman response h_R(t) on a time grid t (seconds).
    Parameters
    ----------
    t : 1D numpy array (seconds)
        Time grid (can include negative times, response will be causal).
    fr : float
        Raman fraction f_R (fractional delayed nonlinearity), default 0.18 for silica.
    normalize : bool
        If True, normalize the delayed-response so that integral(h_R) = 1
        (i.e., h_R is the delayed part used with fR in GNLSE: R(t) = (1-fr) delta(t) + fr*h_R(t)).
    Returns
    -------
    hR : numpy array of same shape as t (seconds^-1 units consistent with definitions)
    Notes
    -----
    - t must be in seconds. Parameter table uses omega in rad/fs and gammas in 1/ps;
      conversions are applied inside the function.
    """
    # Ensure numpy array
    t = np.asarray(t, dtype=float)
    # Parameter extraction
    omega_fs = _HC_TABLE[:, 0]  # rad / fs
    A = _HC_TABLE[:, 1]
    gamma_invps = _HC_TABLE[:, 2]  # 1/ps
    Gamma_invps = _HC_TABLE[:, 3]  # 1/ps

    # Convert to SI: rad/s and 1/s
    omega = omega_fs * 1e15  # rad/fs -> rad/s
    gamma = gamma_invps * 1e12  # 1/ps -> 1/s
    Gamma = Gamma_invps * 1e12  # 1/ps -> 1/s

    # Build causal response: only t >= 0 contribute
    h = np.zeros_like(t, dtype=float)
    pos_mask = t >= 0
    tp = t[pos_mask]

    # Sum modes
    # term_n(t) = A_n * exp(-gamma_n t - (Gamma_n^2 t^2)/4 ) * sin(omega_n t)
    for An, wn, gn, Gn in zip(A, omega, gamma, Gamma):
        h[pos_mask] += An * np.exp(-gn * tp - (Gn**2) * (tp**2) / 4.0) * np.sin(wn * tp)

    # Optionally normalize so the integral of h over t = 1 (makes h the delayed impulse)
    # Note: the physical R(t) used in GNLSE is R(t) = (1 - fr) delta(t) + fr * h(t)
    if normalize:
        dt = t[1] - t[0] if t.size > 1 else 1.0
        integral = np.trapezoid(h, t)
        if integral != 0.0:
            h = h / integral

    # Scale with fr is normally applied externally when building R(t): R = (1-fr)*delta + fr*h
    return h


def single_damped_HO(t, tau1=12.2e-15, tau2=32e-15):
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
