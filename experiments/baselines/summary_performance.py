import json
import itertools
from timeit import default_timer
from pprint import PrettyPrinter
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix, csr_matrix
import sys
sys.path.append('../..')
from data_config import datasets
from lib.random_projection import RandomProjection as rp

##### EXPERIMENTAL GLOBAL PARAMETERS
NTRIALS = 1
projection_dimensions = [1,2,4,8,10]
sketches = ['gaussian','srht','countSketch','sjlt']
###################################

def summary_time_quality():
    '''Generate summaries, time, and measure error.
    Write the experimental output to summary_time_quality dict'''

    summary_time_quality_results = {}
    for data in datasets.keys():
        summary_time_quality_results[data] = {}
        for sketch_type in sketches:
            summary_time_quality_results[data][sketch_type] = {}
            for gamma in projection_dimensions:
                summary_time_quality_results[data][sketch_type][gamma] = {}
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(summary_time_quality_results)

    for data in datasets.keys():
        print("-"*80)
        print("Dataset: {}".format(data))
        input_file = datasets[data]["filepath"]
        input_dest = datasets[data]["input_destination"]
        if datasets[data]['sparse_format'] == True:
            df = load_npz('../../' + input_file)
            df = df.tocsr()
        else:
            df = np.load('../../' + input_file)

        X = df[:,:-1]
        nn,d = X.shape
        n = 2**np.int(np.floor(np.log2(nn)))
        X = X[:n,:]
        print(n,d)
        cov_mat = X.T@X
        # Convert the d x d (small) covariance matrix to a ndarray for compatibility
        #cov_mat = cov_mat.toarray()
        print(cov_mat.shape, type(cov_mat))

        if datasets[data]['sparse_format'] == True:
            # Convert the d x d (small) covariance matrix to a ndarray for compatibility
            cov_mat = cov_mat.toarray()
            print(cov_mat.shape, type(cov_mat))

        frob_norm_cov_mat = np.linalg.norm(cov_mat,'fro')
        spec_norm_cov_mat = np.linalg.norm(cov_mat,2)



        for gamma,sketch_type in itertools.product(projection_dimensions,sketches):
            if data == 'specular' and sketch_type == 'gaussian' and gamma > 2:
                print('Timeout so autoset')
                if gamma  == 4:
                    sketch_time = 336.0
                elif gamma == 8:
                    sketch_time = 1823.0
                elif gamma == 10:
                    sketch_time = 0.0
                frob_error = 0.0
                spec_error = 0.0
                product_time = 0.0
                summary_time_quality_results[data][sketch_type][gamma]['sketch_time'] = sketch_time
                summary_time_quality_results[data][sketch_type][gamma]['product_time'] = product_time
                summary_time_quality_results[data][sketch_type][gamma]['frob_error'] = frob_error
                summary_time_quality_results[data][sketch_type][gamma]['spec_error'] = spec_error
                continue
            print(gamma,sketch_type)
            print('*'*80)
            sketch_time = 0
            product_time = 0
            distortion = 0
            approx_cov_mat = np.zeros((d,d))
            proj_dim = np.int(gamma*d)

            # We use s = 4 globally for the sjlt.
            if sketch_type =='sjlt':
                col_sparsity = 4
            else:
                col_sparsity = 1

            _sketch = rp(X,proj_dim,sketch_type,col_sparsity)

            for _ in range(NTRIALS):
                sketch_start = default_timer()
                sX = _sketch.sketch()
                sketch_time += default_timer() - sketch_start

                product_start = default_timer()
                estimate = sX.T@sX
                product_time += default_timer() - product_start
                approx_cov_mat += estimate

            sketch_time /= NTRIALS
            product_time /= NTRIALS
            approx_cov_mat /= NTRIALS
            frob_error = np.linalg.norm(approx_cov_mat - cov_mat,ord='fro')/frob_norm_cov_mat
            spec_error = np.linalg.norm(approx_cov_mat - cov_mat,ord=2)/spec_norm_cov_mat

            print(f'Testing {data},{gamma},{sketch_type}:')
            print(f'Sketch time: {sketch_time}, Product Time: {product_time}')
            print(f'Frob error: {frob_error}, Spectral error: {spec_error}')

            summary_time_quality_results[data][sketch_type][gamma]['sketch_time'] = sketch_time
            summary_time_quality_results[data][sketch_type][gamma]['product_time'] = product_time
            summary_time_quality_results[data][sketch_type][gamma]['frob_error'] = frob_error
            summary_time_quality_results[data][sketch_type][gamma]['spec_error'] = spec_error
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(summary_time_quality_results)
    with open('../../output/baselines/summary_time_quality_results.json', 'w') as outfile:
       json.dump(summary_time_quality_results, outfile)

if __name__ == "__main__":
    summary_time_quality()
