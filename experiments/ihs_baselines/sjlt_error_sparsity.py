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
from lib.iterative_hessian_sketch import IHS as ihs
from lib.utils import mean_square_error, prediction_error
from lib.synthetic_data_generators import gaussian_design_unconstrained


##### EXPERIMENTAL GLOBAL PARAMETERS
NTRIALS = 10  # NB THIS NEEDS TO BE 10 TO MATCH WITH PAPER
sketches = ['sjlt']

# Experiment: error_vs_num_iters
# Compare the accuracy of the estimator as the nummber of iterations increases.

###################################

def sjlt_error_vs_iterations():
    n = 6_000
    d = 200
    gamma_vals = [5] #[4,6,8]
    sketch_size = int(gamma_vals[0]*d)
    col_sparsities = [1,4,16]
    number_iterations = 20 # 40 #np.asarray(np.linspace(5,40,8), dtype=np.int)
    # Output dictionaries
    error_to_lsq = {} #{sketch_name : {} for sketch_name in sketches}
    error_to_truth = {} #{sketch_name : {} for sketch_name in sketches}
    for s in col_sparsities:
        error_to_lsq[s] = []
        error_to_truth[s] = []
    print(error_to_lsq)
    print(error_to_truth)

    X, y, x_star = gaussian_design_unconstrained(n, d,variance=1.0)

    # Least squares estimator
    x_opt = np.linalg.lstsq(X,y)[0]
    lsq_vs_truth_errors = np.log(np.sqrt(prediction_error(X,x_opt,x_star)))

    for s in col_sparsities:
        col_sparsity = s
        print("Testing col sparsity: {}, num_iterations: {}".format(col_sparsity,number_iterations))
        for sketch_method in sketches:
            #lsq_error, truth_error = 0,0
            lsq_error = np.zeros((number_iterations,))
            truth_error = np.zeros_like(lsq_error)

            my_ihs = ihs(X,y,sketch_method,sketch_size,col_sparsity)
            for trial in range(NTRIALS):
                print('*'*80)
                print("{}, trial: {}".format(sketch_method, trial))
                x_ihs, x_iters = my_ihs.ols_fit_new_sketch_track_errors(number_iterations)
                for _ in range(x_iters.shape[1]):
                    lsq_error[_] += prediction_error(X,x_iters[:,_], x_opt)
                    truth_error[_] += prediction_error(X,x_iters[:,_], x_star)
                print(lsq_error)
                # lsq_error += prediction_error(X,x_ihs, x_opt)
                # truth_error += prediction_error(X,x_ihs, x_star)
            mean_lsq_error = lsq_error/NTRIALS
            mean_truth_error = truth_error/NTRIALS
            print(mean_lsq_error)
            # error_to_lsq[sketch_method][gamma].append(mean_lsq_error)
            # error_to_truth[sketch_method][gamma].append(mean_truth_error)
            error_to_lsq[s] = mean_lsq_error
            error_to_truth[s] = mean_truth_error
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(error_to_lsq)
    pretty.pprint(error_to_truth)

    # Save the dictionaries
    save_dir = '../../output/ihs_baselines//'
    np.save(save_dir+'sjlt_error_sparsity_opt',error_to_lsq)
    np.save(save_dir+'sjlt_error_sparsity_truth',error_to_truth)

def main():
    sjlt_error_vs_iterations()

if __name__ == '__main__':
    main()
