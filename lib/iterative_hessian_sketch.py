import numpy as np
from lib.random_projection import RandomProjection as rp
from lib.regression_solvers import iterative_lasso
from time import process_time

class IHS:
    '''Implementation of the iterative hessian sketching scheme of
    Pilanci and Wainwright (https://arxiv.org/pdf/1411.0347.pdf)
    '''

    def __init__(self,data,targets,sketch_method,sketch_dimension,
                col_sparsity=1):

        # optimisation setup
        self.A = data
        self.b = targets
        self.ATb = self.A.T@self.b
        self.n, self.d = self.A.shape
        self.x = np.zeros((self.d,)) # initialise the startin point.

        self.sketch_method    = sketch_method
        self.sketch_dimension = sketch_dimension
        self.col_sparsity = col_sparsity
        # initialise the sketch to avoid the repeated costs
        self.sketcher = rp(self.A,self.sketch_dimension,
                           self.sketch_method,self.col_sparsity)

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

    def ols_fit_one_sketch_track_errors(self,iterations):
            '''Solve the ordinary least squares problem iteratively using ihs
            generating a fresh sketch at every iteration.

            This needs a larger sketch than if we generate a fresh sketch
            for every iteration.

            '''
            self.sol_errors = np.zeros((self.d,iterations))
            _sketch = self.sketcher.sketch()
            H = _sketch.T@_sketch
            # if choose_step:
            #     eigs = np.linalg.eig(H)[0]
            #     alpha = 2.0/(eigs[0]+eigs[-1])
            # else:
            #     alpha=1.0

            for ii in range(iterations):
                grad_term = 0.6*(self.ATb - self.A.T@(self.A@self.x))
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

    def lasso_fit_new_sketch_track_errors(self,iterations,ell1Bound):
            '''
            fit the lasso model with the ell1bound constraint
            and return the iterates for error check.'''
            self.sol_errors = np.zeros((self.d,iterations))
            for ii in range(iterations):
                 _sketch = self.sketcher.sketch()
                 self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                    self.b,self.x,ell1Bound)
                 self.sol_errors[:,ii] = self.x
            return self.x,self.sol_errors

    def lasso_fit_new_sketch_timing(self,ell1Bound,timeBound):
            '''

            fit the lasso model with the ell1bound constraint
            and return the iterates for error check until the
            allotted time period runs out.'''
            print("RUNNING FOR {} SECONDS".format(timeBound))
            iterations = 0
            time_used = 0
            self.sol_errors = np.zeros((self.d,1))
            while time_used < timeBound:
                sketch_start = process_time()
                _sketch = self.sketcher.sketch()
                sketch_time = process_time() - sketch_start

                if sketch_time + time_used > timeBound:
                    print('Sketch step exceeded time so break.')
                    break

                opt_start = process_time()
                self.x = iterative_lasso(_sketch, self.ATb, self.A,
                                    self.b,self.x,ell1Bound)
                opt_time = process_time() - opt_start
                self.sol_errors = np.concatenate((self.sol_errors,self.x[:,None]),
                                                axis=1)
                if opt_time + 2*sketch_time > timeBound:
                    print('An extra sketch step will exceed time so break.')
                    break

                time_used += opt_time + sketch_time
            # remove the initial col of zeros
            self.sol_errors = self.sol_errors[:,1:]
            return self.x,self.sol_errors
