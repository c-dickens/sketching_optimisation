import pytest
import sys
sys.path.append("..")
import numpy as np
from lib.iterative_hessian_sketch import IHS as ihs
from lib.synthetic_data_generators import gaussian_design_unconstrained
from sklearn.datasets import make_regression

@pytest.fixture
def all_sketch_methods():
    '''
    Returns a list of all sketch methods to test
    '''
    return ['gaussian','srht','countSketch','sjlt']

def test_ihs_initialises(all_sketch_methods):
    '''Checks that the __init__ function is correctly entered and exited from
    the ihs functions'''
    X,y = make_regression(1000,2)
    sketch_dimension = 100
    for sketch_method in all_sketch_methods:
        my_ihs = ihs(X,y,sketch_method,sketch_dimension)
        assert np.array_equal(my_ihs.A,X)
        assert np.array_equal(my_ihs.b,y)
        assert my_ihs.sketch_method == sketch_method
        assert my_ihs.sketch_dimension == sketch_dimension

def test_ols_new_sketch_per_iteration(all_sketch_methods):
    '''
    Test that using IHS and generating a new sketch every iteration yields
    an approximation close to the true estimator.'''
    X,y,_ = gaussian_design_unconstrained(2**13,50,variance=2.5)
    x_opt = np.linalg.lstsq(X,y,rcond=None)[0] # rcond just to suppres warning as per docs
    for sketch_method in all_sketch_methods:
        my_ihs = ihs(X,y,sketch_method,500)
        x_ihs = my_ihs.ols_fit_new_sketch(iterations=20)
        print(sketch_method, np.linalg.norm(x_ihs - x_opt))
        assert np.allclose(x_opt,x_ihs)

def test_ols_one_sketch_per_iteration(all_sketch_methods):
    '''
    Test that using IHS and generating *A SINGLE* sketch yields
    an approximation close to the true estimator.

    Need a larger sketch compared to the test with a new sketch for every
    iteration'''
    X,y,_ = gaussian_design_unconstrained(2**13,50,variance=2.5)
    x_opt = np.linalg.lstsq(X,y,rcond=None)[0] # rcond just to suppres warning as per docs
    for sketch_method in all_sketch_methods:
        my_ihs = ihs(X,y,sketch_method,1000)
        x_ihs = my_ihs.ols_fit_one_sketch(iterations=75)
        print(sketch_method, np.linalg.norm(x_ihs - x_opt))
        #assert np.isclose(x_opt,x_ihs)
        np.testing.assert_array_almost_equal(x_ihs,x_opt)
