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
NTRIALS = 1  # NB THIS NEEDS TO BE 10 TO MATCH WITH PAPER
sketches = ['gaussian','srht','countSketch','sjlt']

# Experiment: solution_error_vs_row_dim
# Compare the accuracy of the estimator as the size of the data grows

ROWDIMS = [100*2**i for i in range(3,11)] #
D = 10
SKETCH_SIZE = np.int(5*D)
ROUNDS = [1 + np.int(np.ceil(np.log2(N))) for N in ROWDIMS]
CLASSICAL_SKETCH_SIZE  = [np.int(N*SKETCH_SIZE) for N in ROUNDS ]
METHODS = sketches + ['Optimal', 'Sketch & Solve']



###################################

# This script runs the baseline experiments on synthetic
# data as described in https://arxiv.org/pdf/1411.0347.pdf
# Figure 1 but with all of the random projections.

def solution_error_vs_row_dim():
    '''
    Increase `n` the input dimension of the problem and
    measure the solution error in both:
    (i) Euclidean norm (`mean_square_error`)
    (ii) Prediction norm (`prediction_error`).

    Error measurements are taken with respect to:
    (i) the optimal solution x_opt
    (ii) the ground truth

    '''
    print('Experimental setup:')
    print(f'IHS sketch size {SKETCH_SIZE}')
    print(f'Sketch and solve sketch size {CLASSICAL_SKETCH_SIZE}')
    print(f'Number of rounds {ROUNDS}')

    # Output dictionaries
    MSE_OPT = {sketches[i] : np.zeros(len(ROWDIMS),) for i in range(len(sketches)) }
    PRED_ERROR_OPT = {sketches[i] : np.zeros(len(ROWDIMS),) for i in range(len(sketches)) }
    MSE_TRUTH = {sketches[i] : np.zeros(len(ROWDIMS),) for i in range(len(sketches)) }
    PRED_ERROR_TRUTH = {sketches[i] : np.zeros(len(ROWDIMS),) for i in range(len(sketches)) }


    MSE_OPT['Sketch & Solve'] = np.zeros(len(ROWDIMS),)
    PRED_ERROR_OPT['Sketch & Solve'] = np.zeros(len(ROWDIMS),)
    MSE_TRUTH['Sketch & Solve'] = np.zeros(len(ROWDIMS),)
    PRED_ERROR_TRUTH['Sketch & Solve'] = np.zeros(len(ROWDIMS),)

    MSE_TRUTH['Exact'] = np.zeros(len(ROWDIMS),)
    PRED_ERROR_TRUTH['Exact'] = np.zeros(len(ROWDIMS),)

    ## Experiment
    for n in ROWDIMS:
        print(f'Testing {n} rows')
        experiment_index = ROWDIMS.index(n)
        _iters = ROUNDS[experiment_index]
        ihs_sketch_size = SKETCH_SIZE
        classic_sketch_size = CLASSICAL_SKETCH_SIZE[experiment_index]

        for trial in range(NTRIALS):
            print("TRIAL {}".format(trial))
            X,y, x_true = gaussian_design_unconstrained(n,D,variance=1.0)
            x_opt = np.linalg.lstsq(X,y)[0]


            for sketch_method in METHODS:
                print('*'*80)
                if sketch_method in sketches or sketch_method == 'Sketch & Solve':
                    if sketch_method == 'sjlt':
                        col_sparsity = 4
                    else:
                        col_sparsity = 1

                    if sketch_method == 'Sketch & Solve':
                        _sketch = rp(X,classic_sketch_size,'countSketch',col_sparsity)
                        SA,Sb = _sketch.sketch_data_targets(y)
                        x_ss = np.linalg.lstsq(SA,Sb)[0]
                        MSE_OPT[sketch_method][experiment_index] += mean_square_error(x_opt, x_ss)
                        PRED_ERROR_OPT[sketch_method][experiment_index] += prediction_error(X,x_opt, x_ss)
                        MSE_TRUTH[sketch_method][experiment_index] += mean_square_error(x_true, x_ss)
                        PRED_ERROR_TRUTH[sketch_method][experiment_index] += prediction_error(X,x_true, x_ss)
                    else:
                        print(f'{sketch_method} IHS')
                        my_ihs = ihs(X,y,sketch_method,ihs_sketch_size,col_sparsity)
                        x_ihs, x_iters = my_ihs.ols_fit_new_sketch_track_errors(_iters)
                        x_errors = x_opt[:,None] - x_iters
                        print(x_errors.shape)
                        MSE_OPT[sketch_method][experiment_index] += mean_square_error(x_opt, x_ihs)
                        PRED_ERROR_OPT[sketch_method][experiment_index] += prediction_error(X,x_opt, x_ihs)
                        MSE_TRUTH[sketch_method][experiment_index] += mean_square_error(x_true, x_ihs)
                        PRED_ERROR_TRUTH[sketch_method][experiment_index] += prediction_error(X,x_true, x_ihs)
                else:
                    # solve exactly
                    #x_opt = np.linalg.lstsq(X,y)[0]
                    MSE_TRUTH["Exact"][experiment_index] += mean_square_error(x_opt, x_true)
                    PRED_ERROR_TRUTH["Exact"][experiment_index] += prediction_error(X,x_opt, x_true)


    for _dict in [MSE_OPT,PRED_ERROR_OPT,MSE_TRUTH,PRED_ERROR_TRUTH]:
        for _key in _dict.keys():
            _dict[_key] /= NTRIALS

    pretty = PrettyPrinter(indent=4)
    pretty.pprint(MSE_OPT)
    pretty.pprint(PRED_ERROR_OPT)
    pretty.pprint(MSE_TRUTH)
    pretty.pprint(PRED_ERROR_TRUTH)

    save_dir = '../../output/ihs_baselines/'
    np.save(save_dir+'ihs_ols_mse_OPT',MSE_OPT)
    np.save(save_dir+'ihs_ols_pred_error_OPT',PRED_ERROR_OPT)
    np.save(save_dir+'ihs_ols_mse_TRUTH',MSE_TRUTH)
    np.save(save_dir+'ihs_ols_pred_error_TRUTH',PRED_ERROR_TRUTH)

    # with open('../../output/baselines/ihs_ols_mse.json', 'w') as outfile:
    #    json.dump(MSE, outfile)
    # with open('../../output/baselines/ihs_ols_pred_error.json', 'w') as outfile:
    #    json.dump(PRED_ERROR, outfile)





def main():
    solution_error_vs_row_dim()

if __name__ == '__main__':
    main()
