import numpy as np
from lib.random_projection import RandomProjection as rp

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
