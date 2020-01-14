import numpy as np
from lib.random_projection import RandomProjection as rp
from lib.regression_solvers import iterative_lasso
from lib.regression_solvers import iterative_lasso_step_size
from time import process_time
from numba import njit,jit
from scipy import sparse
from scipy.sparse import coo_matrix
import datetime
from timeit import default_timer as timer

class IHS:
    '''Implementation of the iterative hessian sketching scheme of
    Pilanci and Wainwright (https://arxiv.org/pdf/1411.0347.pdf)
    '''

    def __init__(self,data,targets,sketch_method,sketch_dimension,
                col_sparsity=1):

        # optimisation setup
        self.A = data
        self.b = targets

        # Need to deal with sparse type
        if isinstance(self.A, np.ndarray):
            self.ATb = self.A.T@self.b
        else:
            self.ATb = sparse.csr_matrix.dot(self.A.T,self.b)
            #self.ATb = np.squeeze(self.ATb.toarray())

        self.n, self.d = self.A.shape
        self.x = np.zeros((self.d,)) # initialise the startin point.

        self.sketch_method    = sketch_method
        self.sketch_dimension = sketch_dimension
        self.col_sparsity = col_sparsity
        # initialise the sketch to avoid the repeated costs
        self.sketcher = rp(self.A,self.sketch_dimension,
                           self.sketch_method,self.col_sparsity)
        self.coo_data = coo_matrix(data)
        self.rows = self.coo_data.row
        self.cols = self.coo_data.col
        self.vals = self.coo_data.data

    ############# OLS (VANILLA) ##########################
    def ols_fit_new_sketch(self,iterations):
            '''Solve the ordinary least squares problem iteratively using ihs
            generating a fresh sketch at every iteration.'''
            for ii in range(iterations):
                _sketch = self.sketcher.sketch()
                H = _sketch.T@_sketch
                grad_term = self.ATb - self.A.T@(self.A@self.x)
                u = np.linalg.solve(H,grad_term)
                self.x = u + self.x
            return self.x

    def ols_fit_one_sketch(self,iterations):
            '''Solve the ordinary least squares problem iteratively using ihs
            generating a fresh sketch at every iteration.

            This needs a larger sketch than if we generate a fresh sketch
            for every iteration.'''
            _sketch = self.sketcher.sketch()
            H = _sketch.T@_sketch
            for ii in range(iterations):
                grad_term = self.ATb - self.A.T@(self.A@self.x)
                u = np.linalg.solve(H,grad_term)
                self.x = u + self.x
            return self.x


    ############# OLS WITH ERROR TRACKING ##########################
    def ols_fit_new_sketch_track_errors(self,iterations):
            '''Solve the ordinary least squares problem iteratively using ihs
            generating a fresh sketch at every iteration and tracking the error
            after every iteration.

            step_size:

            '''
            self.sol_errors = np.zeros((self.d,iterations))
            for ii in range(iterations):
                _sketch = self.sketcher.sketch()
                H = _sketch.T@_sketch
                grad_term = self.ATb - self.A.T@(self.A@self.x)

                # if choose_step:
                #     eigs = np.linalg.eig(H)[0]
                #     alpha = 2.0/(eigs[0]+eigs[-1])
                # else:
                #     alpha = 1.0
                #
                #
                # grad_term *= alpha
                u = np.linalg.solve(H,grad_term)
                self.x = u + self.x
                self.sol_errors[:,ii] = self.x
            return self.x, self.sol_errors

    def ols_fit_one_sketch_track_errors(self,iterations, step_size):
            '''Solve the ordinary least squares problem iteratively using ihs
            generating a fresh sketch at every iteration.

            This needs a larger sketch than if we generate a fresh sketch
            for every iteration.

            '''
            self.sol_errors = np.zeros((self.d,iterations))
            _sketch = self.sketcher.sketch()
            H = _sketch.T@_sketch
            cov_mat = self.A.T@self.A


            # Frobenius and pointwise spectral guarantee
            _,Ss,_ = np.linalg.svd(_sketch)
            _,SigmaA,_ = np.linalg.svd(self.A)
            self.frob_error = np.linalg.norm(H - cov_mat,ord='fro') / np.linalg.norm(cov_mat,ord='fro')
            self.spectral_error = np.abs(SigmaA[0] - Ss[0])/SigmaA[0]


            #self.frob_error = np.linalg.norm(H - self.A.T@self.A,ord='fro') / np.linalg.norm(self.A.T@self.A,ord='fro')
            #self.spec_error = np.linalg.norm(H - self.A.T@self.A,ord=2) / np.linalg.norm(self.A.T@self.A,ord=2)
            # if choose_step:
            #     eigs = np.linalg.eig(H)[0]
            #     alpha = 2.0/(eigs[0]+eigs[-1])
            # else:
            #     alpha=1.0

            for ii in range(iterations):
                grad_term = step_size*(self.ATb - self.A.T@(self.A@self.x))
                u = np.linalg.solve(H,grad_term)
                self.x = u + self.x
                self.sol_errors[:,ii] = self.x
            return self.x, self.sol_errors


    ############# LASSO ##########################
    ##############################################
    ##############################################
    ##############################################

    def lasso_fit_new_sketch(self,iterations,ell1Bound):
            '''
            fit the lasso model with the ell1Bound constraint.'''
            for ii in range(iterations):
                 _sketch = self.sketcher.sketch()
                 self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                    self.b,self.x,ell1Bound)
                 #self.x = u + self.x
            return self.x

    def lasso_fit_new_sketch_track_errors(self,ell1Bound,iterations):
            '''
            fit the lasso model with the ell1bound constraint
            and return the iterates for error check.'''
            print(f'Using X{self.A.shape},y{self.b.shape}')
            print('Using ell1Bound = ', ell1Bound)
            self.sol_errors = np.zeros((self.d,iterations))
            for ii in range(iterations):
                 _sketch = self.sketcher.sketch()
                 self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                    self.b,self.x,ell1Bound)
                 self.sol_errors[:,ii] = self.x
            return self.x,self.sol_errors

    def lasso_fit_new_sketch_speedup(self,ell1Bound,iterations):
            '''
            fit the lasso model with the ell1bound constraint
            and return the iterates for error check until the
            allotted time period runs out.
            '''
            time_used = 0
            self.sol_errors = np.zeros((self.d,1))
            start_time = timer()
            for ii in range(iterations):
                 _sketch = self.sketcher.sketch()
                 self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                    self.b,self.x,ell1Bound)
            end_time = timer()
            t_elapsed = end_time - start_time
            return self.x, t_elapsed


    def lasso_fit_new_sketch_timing(self,ell1Bound,timeBound):
            '''
            fit the lasso model with the ell1bound constraint
            and return the iterates for error check until the
            allotted time period runs out.
            '''
            print("RUNNING FOR {} SECONDS".format(timeBound))
            iterations = 0
            time_used = 0
            self.sol_errors = np.zeros((self.d,1))
            endTime = datetime.datetime.now() + datetime.timedelta(seconds=timeBound)
            while True:
                if datetime.datetime.now() >= endTime:
                    break
                    print("IHS ran for {} seconds".format(timeBound))
                else:
                    iterations += 1
                    _sketch = self.sketcher.sketch()
                    self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                        self.b,self.x,ell1Bound)
            return self.x, iterations

    def lasso_fit_one_sketch_track_errors(self, ell1Bound,iterations, step_size=1.0):
            '''
            Fit the Lasso model with a single sketch and step size equal to
            step_size to descend to optimum
            '''
            _sketch = self.sketcher.sketch()
            self.sol_errors = np.zeros((self.d,iterations))
            for ii in range(iterations):
                self.x = step_size*(iterative_lasso_step_size(_sketch, self.ATb, self.A,
                                       self.b,self.x,ell1Bound, step_size))
                self.sol_errors[:,ii] = self.x
            return self.x, self.sol_errors
