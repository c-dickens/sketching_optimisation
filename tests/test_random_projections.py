import pytest
import sys
sys.path.append("..")
import numpy as np
import scipy.sparse as sparse
from lib.random_projection import RandomProjection as rp


@pytest.fixture
def data_to_test():
    '''
    Generates a sample of test data for the tests.
    This can be replaced by pulling a real dataset.

    To ensure the data-target test passes for the SRHT
    then length of the input should be a power of 2.
    '''
    data = np.random.randn(2**14,100) #/ np.random.randn(2**12,50)
    return data

@pytest.fixture
def all_sketch_methods():
    '''
    Returns a list of all sketch methods to test
    '''
    return ['gaussian','srht','countSketch','sjlt']


def test_raises_exception_proj_dim_larger_than_n(data_to_test,all_sketch_methods):
    '''Ensures that the projection dimension is smaller than n'''
    n,d = data_to_test.shape
    for sketch_method in all_sketch_methods:
        with pytest.raises(Exception):
            print('Testing ', sketch_method)
            sX = rp(data_to_test,n+1,sketch_method)

def test_accepts_non_power2(data_to_test,all_sketch_methods):
    '''Ensures that the projection dimension is smaller than n'''
    n,d = data_to_test.shape
    noise = np.random.randn(10,d)
    _data = np.concatenate((data_to_test,noise),axis=0)
    for sketch_method in all_sketch_methods:
        sX = rp(_data,5*d,sketch_method)
        _sketch = sX.sketch()
        assert _sketch.shape == (5*d,d)

def test_summary_method(data_to_test,all_sketch_methods):
    '''
    Tests that the correct sketch method
    will be executed.'''
    sketch_dim = 100
    for sketch_method in all_sketch_methods:
        summary = rp(data_to_test,sketch_dim,sketch_method)
        assert summary.sketch_type == sketch_method


def test_summary_size(data_to_test,all_sketch_methods):
    '''
    Tests that the summary returned has number of rows equal
    to the required projection dimension'''
    sketch_dim = 100

    for sketch_method in all_sketch_methods:
        if sketch_method == 'sjlt':
            col_sparsity = 2
        else:
            col_sparsity = 1
        summary = rp(data_to_test,sketch_dim,sketch_method, col_sparsity)
        _sketch = summary.sketch()
        print('Sketch size is ', _sketch.shape)
        assert _sketch.shape[0] == sketch_dim

    def test_accept_dense_data(all_sketch_methods):
        '''
        Tests that
        (i) a dense numpy ndarray can be accepted and
        (ii) sparse matrices within the method can be accessed.
        (iii). All sketch methods can act upon dense input data.
        '''
        dense_data = data_to_test()
        sparse_data = sparse.coo_matrix(dense_data)
        n,d = dense_data.shape
        for sketch_method in all_sketch_methods:
            summary = rp(dense_data,5*d,sketch_method)

            # could just check the coo_data arrays but then run into sparsity
            # implementation issues so do array-wise instead.
            assert np.array_equal(sparse_data.row,summary.rows)
            assert np.array_equal(sparse_data.col, summary.cols)
            assert np.array_equal(sparse_data.data, summary.vals)
            _sketch = summary.sketch()

def test_embedding_improves_with_proj_dim(data_to_test,all_sketch_methods):
    '''
    Test that error decreases as we increase projection dimension.

    nb. This test should be used as a calibration tool as *not all* tests
    preserve the ordering on the error as the sketch dimension increases.
    As a result "failing" the test isn't necessarily bad provided it doesn't
    happen too regularly.
    Note that the errors are generaly relatively quite similar.
    '''
    n,d = data_to_test.shape
    sketch_dims = [d,10*d,20*d]
    errors = [0,0,0]
    trials = 5
    covariance = data_to_test.T@data_to_test

    for sketch_method in all_sketch_methods:
        for idx in range(len(sketch_dims)):
            sketch_dim = sketch_dims[idx]
            print(idx)
            error = 0
            for i in range(trials):
                summary = rp(data_to_test,sketch_dim,sketch_method)
                SA = summary.sketch()
                sketch_covariance = SA.T@SA
                error += np.linalg.norm(sketch_covariance - covariance,ord='fro')/np.linalg.norm(covariance,ord='fro')
            errors[idx] = error / trials

        print('Errors for {}\n'.format(sketch_method))
        print(errors)
        assert errors[2] <= errors[1]
        assert errors[1] <= errors[0]

def test_sketch_data_targets(data_to_test,all_sketch_methods):
    '''
    Test that the output is correct dimensionality
    when the input is the data-target pair (A,b).

    Note that this test will fail when the input has been extended
    for the SRHT and the same extension has not been applied to y.

    '''
    n,d = data_to_test.shape
    y = np.random.randn(n)
    sketch_dim = 5*d
    for sketch_method in all_sketch_methods:
        summary = rp(data_to_test,sketch_dim,sketch_method)
        SA,Sb = summary.sketch_data_targets(y)
        assert SA.shape == (sketch_dim,d)
        assert Sb.shape == (sketch_dim,)
