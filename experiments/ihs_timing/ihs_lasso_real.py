import json
import itertools
from timeit import default_timer
from pprint import PrettyPrinter
from joblib import Parallel, delayed
from time import process_time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix, csr_matrix
from sklearn.preprocessing import StandardScaler, Normalizer
import sys
sys.path.append('../..')
from data_config import datasets as all_datasets
from lib.random_projection import RandomProjection as rp
from lib.iterative_hessian_sketch import IHS as ihs
from lib.utils import mean_square_error, prediction_error, sklearn_wrapper
from lib.synthetic_data_generators import my_lasso_data
from lib.regression_solvers import lasso_solver


############################## GLOBALS ##############################
sketches = ['gaussian','srht','countSketch','sjlt']
#######################################################################


def error_vs_time_real_data(data_name,X,y,penalty,sampling_factors,trials,times,x_opt):
    '''Show that a random lasso instance is approximated by the
    hessian sketching scheme'''


    # Experimental setup
    print(80*"-")
    print("Testing dataset: {}".format(data_name))
    print("TESTING LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")
    times2test = times
    n,d = X.shape
    print("Is x_OPT all zeros? {}".format(x_opt == np.zeros_like(x_opt)))
    time_results = {}

    sparse_data = sparse.csr_matrix(X)

    for sketch in sketches:
        time_results[sketch] = {}
        for gamma in sampling_factors:
            time_results[sketch][gamma] = {}

    for sketch_method in sketches:
        for gamma in sampling_factors:

            solution_error_for_iter_check = 1.0  # to check whether the error is small
                                                 # enough to break out of the loop.

            for time_ in times2test:
            #for time_ in range(times):
                print("-"*80)
                print("Testing time: {}".format(time_))
                print("int-log-error: {}".format(np.int(solution_error_for_iter_check)))
                if np.int(solution_error_for_iter_check) <= -16:
                    # continuing for longer doesn't gain anything so just use
                    # previous results.
                    time_results[sketch_method][gamma][time_] = {"error to opt" : total_error2opt,
                                                         "solution error" : total_sol_error,
                                                         "num iterations" : total_iters_used}
                    print("Already converged before time {} seconds so continuing.".format(time_))

                else:
                    # total_error2opt       = 0
                    # total_error2truth     = 0
                    # total_sol_error       = 0
                    # total_objective_error = 0
                    # total_iters_used      = 0
                    total_error2opt       = []
                    total_sol_error       = []
                    total_objective_error = []
                    total_iters_used      = []

                    print("IHS-LASSO ALGORITHM on ({},{}) WITH {}, gamma {}".format(n,d,sketch_method, gamma))

                    for _trial in range(trials):
                        print("Trial {}".format(_trial))
                        shuffled_ids = np.random.permutation(n)
                        X_train, y_train = X[shuffled_ids,:], y[shuffled_ids]
                        sparse_X_train = sparse_data[shuffled_ids,:]
                        sparse_X_train = sparse_X_train.tocoo()
                        rows, cols, vals = sparse_X_train.row, sparse_X_train.col, sparse_X_train.data

                        my_ihs = ihs(X,y,sketch_method,np.int(gamma*d))
                        x_ihs, iters_used = my_ihs.lasso_fit_new_sketch_timing(penalty,time_)
                        my_prediction_error = prediction_error(X,x_opt,x_ihs)
                        print("Iterations completed: ", iters_used)
                        print("Prediction error: ",my_prediction_error)



                        #print("||x^OPT - x_hat||_A^2: {}".format((np.log(my_prediction_error/n))))

                        # Update dict output values
                        error2opt = my_prediction_error
                        solution_error = (1/n)*np.linalg.norm(x_ihs - x_opt)**2
                        print("Trial: {}, Error: {}".format(_trial, error2opt))
                        print("-"*80)
                        # Update counts
                        # total_error2opt  += error2opt
                        # total_sol_error  += solution_error
                        # total_iters_used += iters_used
                        total_error2opt.append(error2opt)
                        total_sol_error.append(solution_error)
                        total_iters_used.append(iters_used)

                    total_error2opt = np.median(total_error2opt)
                    total_sol_error = np.median(total_sol_error)
                    total_iters_used = np.median(total_iters_used)
                    print("Mean log||x^* - x'||_A^2: {}".format(np.log10(total_error2opt)))
                    print("Mean log||x^* - x'||^2: {}".format(total_sol_error))
                    print("Mean number of {} iterations used".format(total_iters_used))
                    time_results[sketch_method][gamma][time_] = {"error to opt" : total_error2opt,
                                                         "solution error" : total_sol_error,
                                                         "num iterations" : total_iters_used}
                    # Bookkeeping - if the error is at 10E-16 don't do another iteration.
                    solution_error_for_iter_check = np.log10(total_error2opt)
                    print("New sol_error_iters: {}".format(solution_error_for_iter_check))
    #
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(time_results)
    return time_results




def main():
    sampling_factors = [5,10] #time_error_ihs_grid['sketch_factors']
    n_trials = 5 #time_error_ihs_grid['num trials']
    time_range = [0.5,1.0,1.5,2.0,2.5] #time_error_ihs_grid['times']
    sklearn_lasso_bound = 1.0
    # for n,d in itertools.product(time_error_ihs_grid['rows'],time_error_ihs_grid['columns']):




    saved_datasets = all_datasets.keys()
    print(saved_datasets)
    new_data_2_sketch= {}
    # W8A done,
    data2test = ['specular']
    for data in all_datasets:
        print(data)
        if data not in data2test:
            continue
        else:
            if data == 'specular':
                time_range = [0.05,0.25,0.5,0.75,1.0]
            print("Dataset: {}".format(data))
            input_file = all_datasets[data]["filepath"]
            input_dest = all_datasets[data]["input_destination"]
            if all_datasets[data]['sparse_format'] == True:
                df = load_npz('../../' + input_file)
                df = df.tocsr()
                unscaled_X = df[:,:-1]
                y = df[:,-1]
                print('Type of y = ', type(y))
                y = np.squeeze(y.toarray())
                scaler = StandardScaler(with_mean=False)
                X = scaler.fit_transform(unscaled_X)


            else:
                df = np.load('../../' + input_file)
                unscaled_X = df[:,:-1]
                y = df[:,-1]
                scaler = StandardScaler()
                X = scaler.fit_transform(unscaled_X)




            n,d = X.shape
            nn,d = X.shape
            n = 2**np.int(np.floor(np.log2(nn)))
            X = X[:n,:]
            y = y[:n,]
            print('Shape of X = ', X.shape)
            print("Target shape: {}".format(y.shape))

            sklearn_time = 0
            #clf = Lasso(sklearn_lasso_bound)
            print("Fitting a lasso for data size: {}".format(X.shape))
            for i in range(1):
                lasso_start = process_time()
                #lasso_skl = clf.fit( np.sqrt(X.shape[0])*X, np.sqrt(X.shape[0])*y)
                x_opt = lasso_solver(X.astype(np.double),y.astype(np.double),sklearn_lasso_bound)
                sklearn_time += process_time() - lasso_start
            print("LASSO-QP time: {}".format(sklearn_time/1))
            print('Shape of X,y = ', X.shape,y.shape)
            print('Type of X,y = ', type(X),type(y))
            #x_LS = np.linalg.lstsq(X,y[:,None])[0]
            #print("Distance to x_LS: {}".format(np.linalg.norm(x_opt - x_LS)))
            #f_opt = original_lasso_objective(X,y, sklearn_lasso_bound,x_opt, penalty=True)
            sklearn_results =  {"x_opt"          : x_opt,
                                #"objective"      : f_opt,
                                "solve time"     : sklearn_time,
                                'lambda val'     : sklearn_lasso_bound}

            results = error_vs_time_real_data(data,X,y,sklearn_lasso_bound,
                                              sampling_factors,n_trials,time_range,x_opt)
            results['exact'] = sklearn_results
            pretty = PrettyPrinter(indent=4)
            pretty.pprint(results)
            file_name = '../../output/ihs_timings/ihs_time_' + data + '.npy'
            np.save(file_name, results)

    #pass

if __name__ == "__main__":
    main()
