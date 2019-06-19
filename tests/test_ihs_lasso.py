import pytest
import sys
sys.path.append("..")
import numpy as np
from lib.iterative_hessian_sketch import IHS as ihs
from lib.synthetic_data_generators import gaussian_design_unconstrained
from lib.regression_solvers import lasso_solver,iterative_lasso
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso


'''
Occasionally these tests fail but you can see from the print
functions that the two output vectors are close, it's just that
they aren't within the tolerance under the error metric.

Running a couple more times should show that they pass, or just
increase the tolerance.'''

@pytest.fixture
def all_sketch_methods():
    '''
    Returns a list of all sketch methods to test
    '''
    return ['gaussian','srht','countSketch','sjlt']

def test_lasso_solver_qp_ihs(all_sketch_methods):
    '''
    Tests that the lasso qp solver gives the same answers
    as the sklearn linear model.
    Generate the sklearn solution first, then take
    then norm and compare.

    nb. We don't compare to sklearn as there is not a
    clean matching between the regularising parameters
    so only check the global and iterative QPs agree.
    '''
    X,y,x_star = gaussian_design_unconstrained(2000,10,1.0)
    n,d = X.shape
    ell_1_bound = 100.0
    # _lambda = 100.0
    # lassoModel = Lasso(alpha=1.0 ,max_iter=1000)
    # sklearn_X, sklearn_y = np.sqrt(n)*X, np.sqrt(n)*y
    # lassoModel.fit(sklearn_X, sklearn_y)
    # x_opt = lassoModel.coef_


    x_opt = lasso_solver(X,y, ell_1_bound)
    x0 = np.zeros((d,))


    for sketch_method in all_sketch_methods:
        my_ihs = ihs(X,y,sketch_method,500)
        x_ihs = my_ihs.lasso_fit_new_sketch(20,ell_1_bound)
        x_ihs_track, error_track = my_ihs.lasso_fit_new_sketch_track_errors(20,ell_1_bound)
        final_sol_error =  (1/n)*np.linalg.norm(X@(x_ihs-x_opt))**2
        print(f'Tracking {sketch_method}, error {np.linalg.norm(x_ihs_track - x_opt)}')
        print("log Error to opt: {}".format(np.log(final_sol_error)))
        print(np.c_[x_opt,x_ihs])
        assert np.allclose(x_opt,x_ihs,1E-3)
        assert np.allclose(x_opt,x_ihs_track,1E-3)
        #np.testing.assert_array_almost_equal(x_opt,x_ihs)


def test_lasso_solver_time(all_sketch_methods):
    '''
    Tests that the lasso qp solver gives the same answers
    as the sklearn linear model.
    Generate the sklearn solution first, then take
    then norm and compare.

    nb. We don't compare to sklearn as there is not a
    clean matching between the regularising parameters
    so only check the global and iterative QPs agree.
    '''
    X,y,x_star = gaussian_design_unconstrained(2000,10,1.0)
    n,d = X.shape
    ell_1_bound = 100.0
    # _lambda = 100.0
    # lassoModel = Lasso(alpha=1.0 ,max_iter=1000)
    # sklearn_X, sklearn_y = np.sqrt(n)*X, np.sqrt(n)*y
    # lassoModel.fit(sklearn_X, sklearn_y)
    # x_opt = lassoModel.coef_


    x_opt = lasso_solver(X,y, ell_1_bound)
    x0 = np.zeros((d,))


    for sketch_method in all_sketch_methods:
        my_ihs = ihs(X,y,sketch_method,500)
        x_ihs_track, error_track = my_ihs.lasso_fit_new_sketch_timing(ell_1_bound,1.5)
        final_sol_error =  (1/n)*np.linalg.norm(X@(x_ihs_track-x_opt))**2
        print(f'Tracking {sketch_method}, error {np.linalg.norm(x_ihs_track - x_opt)}')
        print("log Error to opt: {}".format(np.log(final_sol_error)))
        print(f"{error_track.shape[1]} iterations completed")
        print(np.c_[x_opt,x_ihs_track])
        assert np.allclose(x_opt,x_ihs_track,1E-1)
        #np.testing.assert_array_almost_equal(x_opt,x_ihs)
