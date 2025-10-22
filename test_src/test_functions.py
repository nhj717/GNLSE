from functions import *
from saitoh import Saitoh
import numpy as np
import scipy.fft as fft

def test_add():
    assert np.add(1, 2) == 3


def test_Saitoh():
    wl = 0.5
    d = 0.39
    pitch = 1.56
    test = Saitoh(wl, d, pitch)
    x = test.neff()
    y = test.nFSM()
    z = np.sqrt(test.n_co2)
    print(f"min neff {y}")
    print(f"saito neff {x}")
    print(f"glass neff {z}")
    assert x < z and y < x

def test_fftfreq():
    a = fft.fftshift(fft.fftfreq(3000,1))
    print(a)