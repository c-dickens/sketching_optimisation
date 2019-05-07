'''
Synthetic data generators
'''
import numpy as np
import scipy.stats
from scipy.sparse import random
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler




def gaussian_design_unconstrained(nsamples, nfeatures, variance, density=None):#, random_seed=100):
    '''
    Generate data as described in https://arxiv.org/pdf/1411.0347.pdf 3.1
    1. Generate A in R^{n \times d} with A_ij inn N(0,1)
    2. Choose x^* from S^{d-1}
    3. Set y = Ax^* + w where w ~ N(0,variance*I)
    '''
    #np.random.seed(random_seed)
    if density is not None:
        A = random(nsamples, nfeatures, density)
    else:
        A = np.random.randn(nsamples, nfeatures)
    x_true = np.random.randn(nfeatures)
    x_true /= np.linalg.norm(x_true)
    noise = np.random.normal(loc=0.0,scale=variance,size=(nsamples,))
    y = A@x_true + noise
    return A, y, x_true
