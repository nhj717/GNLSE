import h5py
import numpy as np


def save_dict_to_hdf5(output_filename, group_name, dic):
    with h5py.File(output_filename, "a") as f:
        group = f.create_group(group_name)
        for key, item in dic.items():
            group.create_dataset(key, data=item)
        f.close()


def read_hdf5(filename, group_name):
    df = h5py.File(filename, "r")
    print(df.keys())
    try:
        data_label = list(df[group_name].keys())
        data = []
        for name in data_label:
            data.append(np.array(df[group_name][name]))

    except:
        data_label = "data"
        data = np.array(df[group_name])

    df.close()
    return data_label, data


class Saitoh:
    def __init__(self, wavelength, d, pitch):
        # everthing in micron!!!
        self.wl_over_pitch = wavelength / pitch
        self.d_over_pitch = d / pitch
        self.aij = np.array(
            [
                [0.54808, 0.71041, 0.16904, -1.52736],
                [5.00401, 9.73491, 1.85765, 1.06745],
                [-10.43248, 47.41496, 18.96849, 1.93229],
                [8.22992, -437.50962, -42.4318, 3.89],
            ]
        ).transpose()

        self.bij = np.array(
            [[5, 1.8, 1.7, -0.84], [7, 7.32, 10, 1.02], [9, 22.8, 14, 13.4]]
        ).transpose()

        self.cij = np.array(
            [
                [-0.0973, 0.53193, 0.24876, 5.29801],
                [-16.70566, 6.70858, 2.72423, 0.05142],
                [67.13845, 52.04855, 13.28649, -5.18302],
                [-50.25518, -540.66947, -36.80372, 2.7641],
            ]
        ).transpose()
        self.dij = np.array(
            [[7, 1.49, 3.85, -2], [9, 6.58, 10, 0.41], [10, 24.8, 15, 6]]
        ).transpose()
        A = [0.6961663, 0.4079426, 0.8974794]
        B = [0.068404, 0.116241, 9.896161]
        n2 = 1
        for i in range(3):
            n2 += A[i] / (1 - (B[i] / wavelength) ** 2)
        self.n_co2 = n2

    def calc_Ai(self, i):
        Ai = self.aij[i, 0]
        for j in range(1, 4):
            Ai += self.aij[i, j] * np.power(self.d_over_pitch, self.bij[i, j - 1])
        return Ai

    def V(self):
        Ai_ = np.zeros(4)
        for i in range(0, 4):
            Ai_[i] = Saitoh.calc_Ai(self, i)
        return Ai_[0] + Ai_[1] / (1 + Ai_[2] * np.exp(Ai_[3] * self.wl_over_pitch))

    def calc_Bi(self, i):
        Bi = self.cij[i, 0]
        for j in range(1, 4):
            Bi += self.cij[i, j] * np.power(self.d_over_pitch, self.dij[i, j - 1])
        return Bi

    def W(self):
        Bi_ = np.zeros(4)
        for i in range(0, 4):
            Bi_[i] = Saitoh.calc_Bi(self, i)
        return Bi_[0] + Bi_[1] / (1 + Bi_[2] * np.exp(Bi_[3] * self.wl_over_pitch))

    def nFSM(self):
        nFSM2 = (
            self.n_co2
            - 3 * Saitoh.V(self) ** 2 / (4 * np.pi**2) * self.wl_over_pitch**2
        )
        return np.sqrt(nFSM2)

    def neff(self):
        neff2 = self.n_co2 + 3 / (4 * np.pi**2) * self.wl_over_pitch**2 * (
            Saitoh.W(self) ** 2 - Saitoh.V(self) ** 2
        )
        return np.sqrt(neff2)
