from functions import *
import numpy as np


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
