from functions import *
import numpy as np

def test_add():
    assert add(1, 2) == 3

def test_Saitoh():
    wl = 1.064
    d = 0.44
    pitch = 1.14
    test = Saitoh(wl,d,pitch)
    x = test.neff()
    y = test.nFSM()
    z = np.sqrt(test.n_co2)
    print(y)
    print(x)
    print(z)
    assert x<z and y<x