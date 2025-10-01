import h5py
from numpy import array

def save_dict_to_hdf5(output_filename,dic):
    with h5py.File(output_filename, "a") as f:
        for key, item in dic.items():
            if isinstance(item, dict):
                try:
                    subgroup = f.create_group(key)
                    save_dict_to_hdf5(item, subgroup)
                except:
                    del f[key]
                    subgroup = f.create_group(key)
                    save_dict_to_hdf5(item, subgroup)
            else:
                 f.create_dataset(key, data=item)
        f.close()


def read_hdf5(filename,group_name):
    df = h5py.File(filename, 'r')
    print(df.keys())
    try:
        data_label = list(df[group_name].keys())
        data = []
        for name in data_label:
            data.append(array(df[group_name][name]))

    except:
        data_label = 'data'
        data = array(df[group_name])

    df.close()
    return data_label,data
