import json
import itertools
from timeit import default_timer
from pprint import PrettyPrinter
from joblib import Parallel, delayed
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
from lib.synthetic_data_generators import my_lasso_data


############################## GLOBALS ##############################
sketches = ['gaussian','srht','countSketch','sjlt']
#######################################################################


def single_exp(_trial,n,d,X,y,
                sketch_size, sketch_method,
                run_time,sklearn_lasso_bound):
    # for _trial in range(trials):
    print("Trial {}".format(_trial))
    shuffled_ids = np.random.permutation(n)
    X_train, y_train = X[shuffled_ids,:], y[shuffled_ids]
    my_ihs = ihs(X_train,y_train,sketch_method,sketch_size)
    #x_ihs, error_track = my_ihs.lasso_fit_new_sketch_timing(sklearn_lasso_bound,run_time)
    x_ihs,iters_used = my_ihs.lasso_fit_new_sketch_timing(sklearn_lasso_bound,run_time)
    #iters_used = error_track.shape[1]
    return x_ihs,iters_used

def error_vs_time_real(data,sampling_factors,trials,times2test,sklearn_lasso_bound):
    '''Show that a random lasso instance is approximated by the
    hessian sketching scheme'''
    print(80*"-")
    print("TESTING LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")


    print("Dataset: {}".format(data))
    input_file = datasets[data]["filepath"]
    input_dest = datasets[data]["input_destination"]
    if datasets[data]['sparse_format'] == True:
        df = load_npz('../../' + input_file)
        df = df.tocsr()
    else:
        df = np.load('../../' + input_file)

    X = df[:,:-1]
    y = df[:,-1]
    nn,d = X.shape
    n = 2**np.int(np.floor(np.log2(nn)))
    X = X[:n,:]
    y = y[:n,]
    print(n,d)
    cov_mat = X.T@X
    # Convert the d x d (small) covariance matrix to a ndarray for compatibility
    #cov_mat = cov_mat.toarray()
    print(cov_mat.shape, type(cov_mat))

    if datasets[data]['sparse_format'] == True:
        # Convert the d x d (small) covariance matrix to a ndarray for compatibility
        cov_mat = cov_mat.toarray()
        print(cov_mat.shape, type(cov_mat))


    ### Test Sklearn implementation
    print("Beginning test")
    x_opt, f_opt, sklearn_time = sklearn_wrapper(X,y,n,d,
                                                 sklearn_lasso_bound,1)
    print("LASSO-skl time: {}".format(sklearn_time))
    # ground Truths
    time_results = {"Sklearn" : {"objective"      : f_opt,
                                 "solve time"     : sklearn_time},}

    for sketch in sketches:
        time_results[sketch] = {}
        for gamma in sampling_factors:
            time_results[sketch][gamma] = {}

    for sketch_method in sketches:
        for gamma in sampling_factors:
            sketch_size = np.int(gamma*d)

            euclidean_error_for_iter_check = 1.0  # to check whether the error is small
                                         # enough to break out of the loop.
            for time in times2test:
                print("-"*80)
                print("Testing time: {}".format(time))
                print("int-log-error: {}".format(np.int(euclidean_error_for_iter_check)))
                if np.int(euclidean_error_for_iter_check) <= -16:
                    # continuing for longer doesn't gain anything so just use
                    # previous results.
                    time_results[sketch_method][gamma][time] = {"error to opt" : total_error2opt,
                                                         "solution error" : total_sol_error,
                                                         "num iterations" : total_iters_used}
                    print("Already converged before time {} seconds so continuing.".format(time))
                else:
                    total_error2opt       = 0
                    total_error2truth     = 0
                    total_sol_error       = 0
                    total_objective_error = 0
                    total_iters_used      = 0
                    print("IHS-LASSO ALGORITHM on ({},{}) WITH {}, gamma {}".format(n,d,sketch_method, gamma))
                    results = Parallel(n_jobs=-1,prefer="threads")(delayed(single_exp)\
                                    (_trial,n,d,X,y,sketch_size, sketch_method,time,sklearn_lasso_bound) for _trial in range(trials))
                    for i in range(trials):
                        x_ihs = results[i][0]
                        total_iters_used += results[i][1] #np.abs(results[i][0])

                        # Update dict output values
                        error2opt = prediction_error(X,x_opt,x_ihs)**2
                        euclidean_error = (1/n)*np.linalg.norm(x_ihs - x_opt)**2

                        # Update counts
                        total_error2opt += error2opt
                        total_sol_error += euclidean_error

                    total_error2opt /= trials
                    total_sol_error /= trials
                    total_iters_used /= trials
                    print("Mean log||x^* - x'||_A^2: {}".format(np.log10(total_error2opt)))
                    print("Mean log||x^* - x'||^2: {}".format(total_sol_error))
                    print("Mean number of {} iterations used".format(total_iters_used))
                    time_results[sketch_method][gamma][time] = {"error to opt" : total_error2opt,
                                                         "solution error" : total_sol_error,
                                                         "num iterations" : total_iters_used}
                    # Bookkeeping - if the error is at 10E-16 don't do another iteration.
                    euclidean_error_for_iter_check = np.log10(total_error2opt)
                    print("New sol_error_iters: {}".format(euclidean_error_for_iter_check))
        pretty = PrettyPrinter(indent=4)
        pretty.pprint(time_results)
        file_name = '../../output/ihs_timings/ihs_time_synthetic' + str(n) + '_' + str(d) + '.npy'
        #file_name = '../../output/ihs_timings/debug' + str(n) + '_' + str(d) + '.npy'
        #np.save(file_name, time_results)
        pass




def main():
    sklearn_lasso_bound = 5.0
    sampling_factors = [5,10]
    n_trials = 10
    time_range = np.linspace(0.05,3.5,10)
    for data in datasets.keys():
        #if data == 'YearPredictionMSD' or data == 'specular':
        if data != 'w8a':
            continue
        print("-"*80)
        error_vs_time_real(data,sampling_factors,n_trials, time_range,sklearn_lasso_bound)

    pass

if __name__ == "__main__":
    main()
