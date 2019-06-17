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
sketches = ['gaussian','srht','countSketch','sjlt']

# Experiment: error_vs_num_iters
# Compare the accuracy of the estimator as the nummber of iterations increases.

###################################

def error_vs_dimensionality():
    dimension = [2**i for i in range(4,9)]
    METHODS = sketches + ['Exact', 'Sketch & Solve']

    # Output dictionaries
    error_to_truth = {_ : {} for _ in METHODS}
    for _ in METHODS:
        for d in dimension:
            error_to_truth[_][d] = 0
    print(error_to_truth)



    for d in dimension:
        n = 250*d
        print(f'TESTING {n},{d}')
        ii = dimension.index(d)
        sampling_rate = 10
        num_iterations = 1+np.int(np.log(n))
        for method in METHODS:
            if method == 'sjlt':
                col_sparsity = 4
            else:
                col_sparsity = 1
            for trial in range(NTRIALS):
                # Generate the data
                X, y, x_star = gaussian_design_unconstrained(n,d,1.0)
                if method is "Exact":
                    print('Exact method.')
                    x_hat = np.linalg.lstsq(X,y)[0]

                elif method is "Sketch & Solve":
                    sketch_size = sampling_rate*num_iterations*d
                    print(f"S&S with {sketch_size} sketch size")
                    _sketch = rp(X,sketch_size,'countSketch',col_sparsity)
                    SA,Sb = _sketch.sketch_data_targets(y)
                    x_hat = np.linalg.lstsq(SA,Sb)[0]
                else:
                    sketch_size = sampling_rate*d
                    print(f"Using {num_iterations} iterations, sketch_size {sketch_size} and {method}")
                    my_ihs = ihs(X,y,method,sketch_size,col_sparsity)
                    x_hat = my_ihs.ols_fit_new_sketch(num_iterations)

                error = prediction_error(X,x_star,x_hat)
                error_to_truth[method][d] += error
    for _ in METHODS:
        for d in dimension:
            error_to_truth[_][d] /= NTRIALS
    error_to_truth['Dimensions'] = dimension
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(error_to_truth)
    save_dir = '../../output/ihs_baselines/'
    np.save(save_dir+'error_vs_dims',error_to_truth)
def main():
    error_vs_dimensionality()

if __name__ == '__main__':
    main()
