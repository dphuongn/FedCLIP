import h5py
import numpy as np
from pathlib import Path


def average_data(algorithm="", dataset="", times=10, delete=False):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, times, delete)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accurancy))
    print("mean for best accuracy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", times=10, delete=False):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete)))

    return test_acc


def read_data_then_delete(file_name, delete):
    file_path = Path("../results") / f"{file_name}.h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        file_path.unlink()
    print("Length: ", len(rs_test_acc))

    return rs_test_acc