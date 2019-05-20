# Config file to easily extract various datasets
import os.path
import numpy as np
import scipy.sparse
from data.get_data.get_datasets import all_datasets

datasets = {}
num_repeats = 10

for data_name in all_datasets.keys():
    file_name = all_datasets[data_name]['outputFileName'][3:]

    if all_datasets[data_name]['input_destination'] == 'LIBSVM':
        _ext = '.npz'
    else:
        _ext = '.npy'

    datasets[data_name] = {'filepath' : 'data/' + file_name + _ext,
                           'repeats'  : num_repeats}
    datasets[data_name]['input_destination'] = all_datasets[data_name]['input_destination']
    #print(file_name)
print(datasets)
