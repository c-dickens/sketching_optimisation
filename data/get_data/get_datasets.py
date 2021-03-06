'''
Script to get data from UCI ML repo and convert to
.npy format for random projection exps

saves a sparse format as .npz NB need scipy.sparse.load_npz to open
files of this format.
'''


import pandas as pd
import numpy as np
import scipy
from scipy.sparse import save_npz
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import loadmat
from scipy.sparse import save_npz
from sklearn.datasets import load_svmlight_file
import os.path
import openml
import urllib.request

all_datasets = {
    "YearPredictionMSD" : {
          "url" : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
          "outputFileName" : '../YearPredictionMSD',
          'target_col' : 0,
          'input_destination' : 'UCI',
          'sparse_format' : False
         },


   "w4a" : {
               "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w4a',
               "outputFileName" : '../w4a',
               'target_col' : 'libsvm',
               'input_destination' : 'LIBSVM',
               'sparse_format' : True
              },

    "w6a" : {
        "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w6a',
        "outputFileName" : '../w6a',
        'target_col' : 'libsvm',
        'input_destination' : 'LIBSVM',
        'sparse_format' : True
       },

    "w8a" : {
        "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w8a',
        "outputFileName" : '../w8a',
        'target_col' : 'libsvm',
        'input_destination' : 'LIBSVM',
        'sparse_format' : True
       },

       ### These are the OPENML datasets.
       ### Do not need a target col as this is done in the openml wrapper.

      "APSFailure" : {
        "url" : 'https://www.openml.org/d/41138',
        "outputFileName" : '../aps',
        'input_destination' : 'OPENML',
        'openml_id' : 41138,
        'sparse_format' : False
      },

      "albert" : {
        "url" : 'https://www.openml.org/d/41147',
        "outputFileName" : '../albert',
        'input_destination' : 'OPENML',
        'openml_id' : 41147,
        'sparse_format' : False
      },

      "aloi" : {
          "url" : 'https://www.openml.org/d/1592',
          "outputFileName" : '../aloi',
          'input_destination' : 'OPENML',
          'openml_id' : 1592,
          'sparse_format' : True
      },

      "KDDCup99" : {
          "url" : 'https://www.openml.org/d/1113',
          "outputFileName" : '../kdd_cup_99',
          'input_destination' : 'OPENML',
          'openml_id' : 1113,
          'sparse_format' : False
      },

      "covertype" : {
        "url" : 'https://www.openml.org/d/293',
        "outputFileName" : '../covertype',
        'input_destination' : 'OPENML',
        'openml_id' : 293,
        'sparse_format' : True
      },

      "fars" : {
        "url" : 'https://www.openml.org/d/40672',
        "outputFileName" : '../fars',
        'input_destination' : 'OPENML',
        'openml_id' : 40672,
        'sparse_format' : False
      },

      "abtaha2" : {
        'url' : "https://sparse.tamu.edu/mat/JGD_Taha/abtaha2.mat",
        "in_path" : 'sp_abtaha2.mat',
        "outputFileName" : '../abtaha2',
        'input_destination' : 'SUITE-SPARSE',
        'sparse_format' : True
      },

      "specular" : {
        'url' : "https://sparse.tamu.edu/mat/Brogan/specular.mat",
        "in_path" : 'sp_specular.mat',
        "outputFileName" : '../specular',
        'input_destination' : 'SUITE-SPARSE',
        'sparse_format' : True
      }

}

def get_uci_data(dataset):
    print('Downloading from UCI')
    target = all_datasets[dataset]['target_col']
    try: # see if there is a header key and if so use it
        header = all_datasets[dataset]['header']
        header_loc = all_datasets[dataset]['header_location']
        df = pd.read_csv(file_url,
                         header=header_loc,
                         error_bad_lines=False)
    except KeyError: # otherwise there is a key error so just read csv
        print("No header given")
        df = pd.read_csv(file_url)
    df.fillna(0)
    data = df.values
    print("data has shape: {}".format(data.shape))
    col_selector = [id for id in range(data.shape[1]) if id != target]
    y = data[:, target]
    X = data[:,col_selector]
    np.save(out_file, np.c_[X,y])
    #return data

def get_libsvm_data(dataset):
    # Read in libsvm data. In sparse representation so
    # convert to array to concatenate (numpy throws a fit
    # otherwise) then convert back to coo_matrix to work with
    # our interface elsewhere.
    out_file = all_datasets[dataset]['outputFileName']
    print("Loading {} from libsvm format".format(dataset))
    input_file_name = dataset
    X, y = load_svmlight_file(input_file_name)
    X = X.toarray()
    data = coo_matrix(np.c_[X,y]) #np.concatenate((X, y[:,None]), axis=1)
    print(type(X),X.shape)
    print(type(data),data.shape)
    save_npz(out_file,data)
    #np.save(out_file, data)

def get_openml_data(dataset):
    '''Tool to downloads openml data and put into correct format'''
    out_file = all_datasets[dataset]['outputFileName']
    _id = all_datasets[dataset]['openml_id']
    dataset = openml.datasets.get_dataset(_id)

    # Print a summary
    print("This is dataset '%s', the target feature is '%s'" %
          (dataset.name, dataset.default_target_attribute))
    X, y = dataset.get_data( target=dataset.default_target_attribute)
    n,d = X.shape
    print(f'Shape of data: {n},{d}')
    print(f'Aspect ratio: {d/n}')
    print(f'Type of data: {type(X)}')

    if isinstance(X,csr_matrix):
        nnz = X.count_nonzero()
        Xdense = X.toarray()
        Xdense = np.nan_to_num(Xdense)
        data = coo_matrix(np.c_[Xdense,y])
        print(type(data),data.shape)
        save_npz(out_file,data)
    else:
        nnz = np.count_nonzero(X)
        data = np.c_[X,y]
        data[np.isnan(data)] = 0
        np.save(out_file, data)

    print(f'Density: {nnz/(n*d)}')

def get_suite_sparse_data(dataset):
    '''Wrapper to download the suite sparse datasets.'''
    print(f'Getting dataset {dataset}')
    print(dataset)
    data_file = all_datasets[dataset]['in_path']
    out_file = all_datasets[dataset]['outputFileName']
    mat = scipy.io.loadmat(data_file)
    X = mat['A']
    y = mat['b']

    print('Type(y) = ',type(y))
    if type(y) != np.ndarray:
        y = y.toarray()


    nnz = X.count_nonzero()
    n,d = X.shape
    print(f'Shape of data: {n},{d}, {y.shape}')
    print(f'Aspect ratio: {d/n}')
    print(f'Type of data: {type(X)}')
    print(f'Density: {nnz/(n*d)}')
    X = X.toarray()
    data = coo_matrix(np.c_[X,y])
    save_npz(out_file,data)




def get_datasets():
    '''Download data from the dictionary datasets_2_test'''
    for dataset in all_datasets:
        print(dataset)
        out_file = all_datasets[dataset]['outputFileName']
        in_dest =  all_datasets[dataset]['input_destination']

        file_url = all_datasets[dataset]['url']

        # if the .npy version doesn't exist then make one

        if not (os.path.isfile(out_file + '.npy') or os.path.isfile(out_file + '.npz')):
            if in_dest == 'UCI':
                get_uci_data(dataset)

            elif in_dest == 'LIBSVM':
                get_libsvm_data(dataset)

            elif in_dest == 'OPENML':
                get_openml_data(dataset)

            elif in_dest == 'SUITE-SPARSE':
                get_suite_sparse_data(dataset)


if __name__ == "__main__":
    get_datasets()



    #
    #

    #         print("{} data file doesn't exist so download or convert from .mat".format(dataset))
    #         file_url = all_datasets[dataset]['url']
    #
    #         if all_datasets[dataset]['matlab']:
    #             print("Converting {} from .mat".format(dataset))
    #             all_datasets[dataset]['matlab']
    #             # do matlab preprocessing
    #             input_file = all_datasets[dataset]['inputFileName']
    #             print("Reading {} from file".format(dataset))
    #
    #             matrix = scipy.io.loadmat(input_file)
    #             print(matrix) #might need this just to check keys
    #
    #             try:
    #                 data = matrix['X']
    #                 X = data[:, :-1]
    #                 y = data[:,-1]
    #             except KeyError:
    #                 try:
    #                     X = matrix['A']
    #                     try:
    #                         y = matrix['b']
    #                     except KeyError:
    #                         X = X[:,:-1]
    #                         y = X[:,-1]
    #
    #                     pass
    #                 except KeyError:
    #                     print("No keys 'A' or 'X' so check the sparse representation")
    #                     pass
    #
    #             print("Shape of {} data: {}".format(dataset, X.shape))
    #
    #             if isinstance(X, np.ndarray):
    #                 print("ndarray")
    #                 np.save(out_file, np.c_[X,y])
    #                 save_npz(out_file+'_sparse', coo_matrix(np.c_[X,y]))
    #             else:
    #                 print("Already sparse so saving as npz")
    #                 save_npz(out_file+'_sparse', X.tocoo())
    #
    #
    #         else:
    #             print("downloading from UCI/Open ML/LIBSVM")
    #
    #             if all_datasets[dataset]['target_col'] == 'libsvm':
    #                 # Read in libsvm data. In sparse representation so
    #                 # convert to array to concatenate (numpy throws a fit
    #                 # otherwise) then convert back to coo_matrix to work with
    #                 # our interface elsewhere.
    #
    #                 print("Loading {} from libsvm format".format(dataset))
    #                 input_file_name = dataset
    #                 X, y = load_svmlight_file(input_file_name)
    #                 X_dense = X.toarray()
    #                 X_y = np.concatenate((X_dense, y[:,None]), axis=1)
    #                 Z = coo_matrix(X_y)
    #
    #                 # save sparse representation of Z
    #                 save_npz(out_file+'_sparse', Z)
    #                 # Save dense representation for SRHT
    #                 np.save(out_file,X_y)
    #
    #
    #
    #
    #                 #print(np.c_[X,y_new].shape)
    #             else:
    #
    #                     # do pandas download
    #                 try: # see if there is a header key and if so use it
    #                     header = all_datasets[dataset]['header']
    #                     header_loc = all_datasets[dataset]['header_location']
    #                     df = pd.read_csv(file_url, header=header_loc,error_bad_lines=False)
    #                 except KeyError: # otherwise there is a key error so just read csv
    #                     print("No header given")
    #                     df = pd.read_csv(file_url)
    #                 df.fillna(0)
    #
    #                 try:  # dummy variables for categorical features
    #                     categorical_cols = all_datasets[dataset]['categorical columns']
    #                     dummies = pd.get_dummies(df[categorical_cols])
    #                     df = pd.concat([df, dummies],axis=1)
    #                     df.drop(categorical_cols, inplace=True,axis=1)
    #
    #                 except KeyError:
    #                     print("no categorical columns")
    #                     pass
    #
    #                 print("Downloaded {} data".format(dataset))
    #                 data = df.values
    #                 print("data has shape: {}".format(data.shape))
    #                 target_id = all_datasets[dataset]['target_col']
    #                 col_selector = [id for id in range(data.shape[1]) if id != target_id]
    #                 y = data[:, target_id]
    #                 X = data[:,col_selector]
    #                 sparsity = np.count_nonzero(X)/(X.shape[0]*X.shape[1])
    #
    #                 if sparsity < 0.4:
    #                     save_npz(out_file+'_sparse', coo_matrix(np.c_[X,y]))
    #                 # If data is sparse also save a sparse representation.
    #
    #                 # Save a dense representation for the SRHT methods.
    #                 np.save(out_file,np.c_[X,y])
    #
    #
    # #
    # # print("Downloading")
    # # file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    # # #file = "https://sparse.tamu.edu/MM/Mittelmann/rail2586.tar.gz"
    # #
    # # df = pd.read_csv(file)
    # # print("Data shape {}".format(df.shape))
    # #
    # # data = df.values
    # # X = data[:,1:]
    # # y = data[:,0]
    # # #print("Data shape is {}".format(X.shape))
    # # #print("Target shape {}".format(y.shape))
    # # #tidy_data = np.hstack((X,y),axis=1)
    # # np.save(outputFileName, data)
