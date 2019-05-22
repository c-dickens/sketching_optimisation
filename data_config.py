# Config file to easily extract various datasets
import os.path
import numpy as np
import scipy.sparse
from data.get_data.get_datasets import all_datasets

datasets = {}
num_repeats = 10
for data_name in all_datasets.keys():
    datasets[data_name] = {}

for data_name in all_datasets.keys():
    file_name = all_datasets[data_name]['outputFileName'][3:]

    if all_datasets[data_name]['sparse_format'] == True:
        _ext = '.npz'
        datasets[data_name]['sparse_format'] = True
    else:
        _ext = '.npy'
        datasets[data_name]['sparse_format'] = False

    datasets[data_name]['filepath'] =  'data/' + file_name + _ext
    datasets[data_name]['repeats']  = num_repeats
    datasets[data_name]['input_destination'] = all_datasets[data_name]['input_destination']
    #print(file_name)
#print(datasets)
