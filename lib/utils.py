'''utility functions used throughout'''

import numpy as np
import scipy as sp

# Error metrics
def mean_square_error(x1, x2):
    '''compute ||x2 - x1||_2^2'''
    return np.linalg.norm(x2-x1)**2

def prediction_error(data,x1,x2):
    '''compute np.sqrt(1/n)*||A(x1-x2)||_2'''
    return (1/data.shape[0])*np.linalg.norm(data@(x1-x2),2)**2
