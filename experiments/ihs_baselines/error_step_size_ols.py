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
from lib.utils import mean_square_error, prediction_error, sklearn_wrapper
from lib.synthetic_data_generators import my_lasso_data, gaussian_design_unconstrained


import matplotlib.pyplot as plt


##### EXPERIMENTAL GLOBAL PARAMETERS
NTRIALS = 1  # NB THIS NEEDS TO BE 10 TO MATCH WITH PAPER
sketches = ['countSketch']

STEPSIZE = [0.4, 0.5, 0.55, 0.6, 0.7]#, 1.0]
# Experiment: error vs number of iterations with varying step size on OLS instance
# Compare the accuracy of the estimator as the nummber of iterations increases.

###################################

def error_vs_iterations():
    n = 6_000
    d = 200
    gamma_vals = [5]
    number_iterations = 30


    # Output dictionaries indexed by:
    # sketch method (sketches) --> sketch size (gamma_vals) --> STEPSIZE
    error_to_lsq = {sketch_name : {} for sketch_name in sketches}
    error_to_truth = {sketch_name : {} for sketch_name in sketches}
    for sketch_name in sketches:
        for gamma in gamma_vals:
            error_to_lsq[sketch_name][gamma] = {}
            error_to_truth[sketch_name][gamma] = {}
            for step in STEPSIZE:
                    error_to_lsq[sketch_name][gamma][step] = []
                    error_to_truth[sketch_name][gamma][step] = []

    X,y,x_star = gaussian_design_unconstrained(n,d,variance=1.0)



    # # Least squares estimator
    x_opt = np.linalg.lstsq(X,y)[0]
    print('-'*80)
    print("Beginning test")
    lsq_vs_truth_errors = np.log(np.sqrt(prediction_error(X,x_opt,x_star)))
    print(lsq_vs_truth_errors)


    for gamma in gamma_vals:
        sketch_size = int(gamma*d)
        print("Testing gamma: {}, num_iterations: {}".format(gamma,number_iterations))
        for sketch_method in sketches:
            #lsq_error, truth_error = 0,0
            lsq_error = np.zeros((number_iterations,))
            truth_error = np.zeros_like(lsq_error)
            if sketch_method == 'sjlt':
                col_sparsity = 4
            else:
                col_sparsity = 1

            my_ihs = ihs(X,y,sketch_method,sketch_size,col_sparsity)
            for step in STEPSIZE:
                lsq_error = np.zeros((number_iterations,))
                for trial in range(NTRIALS):
                    print('*'*80)
                    print("{}, trial: {}".format(sketch_method, trial))
                    print('Step size: ', step)
                    x_ihs, x_iters = my_ihs.ols_fit_one_sketch_track_errors(number_iterations, step)
                    for _ in range(x_iters.shape[1]):
                        residual = prediction_error(X,x_iters[:,_], x_opt)
                        print('Trial {}, residual {}'.format(_,residual))
                        lsq_error[_] += residual

                    # Sketching Error for this step size.
                    frob_error = my_ihs.frob_error
                    spec_error = my_ihs.spectral_error
                    print('Frobenius error: ', frob_error)
                    print('Spectral error: ', spec_error)
                mean_lsq_error = lsq_error/NTRIALS
                error_to_lsq[sketch_method][gamma][step] = mean_lsq_error
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(error_to_lsq)

    ### PLOTTING ###
    my_markers = ['.', 's', '^', 'D', '*', 'h']
    my_colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    fig,ax = plt.subplots()
    x_vals = range(1,number_iterations+1)
    for gamma in gamma_vals:
        for sketch_method in sketches:
            for i,step in enumerate(STEPSIZE):
                _marker = my_markers[i]
                _colour = my_colours[i]
                residual = error_to_lsq[sketch_method][gamma][step]
                ax.plot(x_vals, residual, label=step, marker=_marker,color=_colour)
    ax.set_yscale('log')
    ax.set_xticks(x_vals[1::2])
    ax.set_xlabel("Iterations")
    ax.set_ylabel('$\| x^t - x_{\t{opt}}\|_A^2$')
    ax.legend(title='Step sizes') # nb this only makes sense for one sketch dimension
    ax.set_title('{}, m={}d, step size varied'.format(sketches[0],gamma))
    plt.show()


    # # Save the dictionaries
    # save_dir = '../../output/ihs_baselines/'
    # np.save(save_dir+'error_vs_iters_opt_5_10',error_to_lsq)
    # np.save(save_dir+'error_vs_iters_truth_5_10',error_to_truth)

def main():
    error_vs_iterations()

if __name__ == '__main__':
    main()
