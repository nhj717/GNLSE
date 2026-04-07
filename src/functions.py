import h5py
import numpy as np


def save_dict_to_hdf5(output_filename, group_name, dic):
    with h5py.File(output_filename, "a") as f:
        if group_name in f:
            del f[group_name]
            group = f.create_group(group_name)
        else:
            group = f.create_group(group_name)

        for key, item in dic.items():
            group.create_dataset(key, data=item)
        f.close()


def read_hdf5(filename, group_name=None, read=True):
    df = h5py.File(filename, "r")
    if read is True:
        print(df.keys())
    if group_name == None:
        data_label = list(df.keys())
        data = []
        for name in data_label:
            data.append(np.array(df[name]))
    else:
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


def mode_overlap(E_i, E_ref, w):
    num = np.sum(w * E_i * np.conjugate(E_ref))
    denA = np.sum(w * E_i * np.conjugate(E_i))
    denB = np.sum(w * E_ref * np.conjugate(E_ref))
    overlap = abs(num) ** 2 / (abs(denA * denB) + 1e-300)
    return overlap
