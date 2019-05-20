import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse import coo_matrix
import pyfht

# Helper functions

@njit(fastmath=True)
def fast_countSketch(SA,row,col,data,sign,row_map):
    N = len(data)
    for idx in range(N):
        SA[row_map[row[idx]],col[idx]]+=data[idx]*sign[row[idx]]
    return SA

def shift_bit_length(x):
    '''Given int x find next largest power of 2.
    If x is a power of 2 then x is returned '''
    return 1<<(x-1).bit_length()

class RandomProjection:
    '''
    Class based implementation of different
    Random projection methods for RNLA.

    Current examples are taken from [1]
    however there are various sampling methods
    which could also be implemented.

    [1] - https://researcher.watson.ibm.com/researcher/files/us-dpwoodru/wNow.pdf'''

    def __init__(self,data,proj_dim,
                 sketch_type,col_sparsity=1):
        '''
        data - ndarray of the data (or vector) to sketch
        proj_dim - int --> Value to sketch down to
        sketch_type - str determining which sketch to use

        Sketch Type support args:
        1. gaussian
        2. srht
        3. sjlt
        '''
        self.data = data
        self.n, self.d = data.shape
        self.proj_dim = proj_dim
        self.sketch_type = sketch_type

        # Create an attribute for dense data for later reference.
        # If data is ndarray then just make new reference, otherwise, if input
        # is sparse then make reference for dense data.
        if isinstance(self.data, np.ndarray):
            self.dense_data = self.data
        elif isinstance(self.data,sparse.coo.coo_matrix) or isinstance(self.data, sparse.csr.csr_matrix):
            print('Converting sparse to dense data.')
            self.dense_data = self.data.toarray()

        # Convert data to sparse data for cross-comparison between sketches
        # and accept sparse inputs
        # Note that the second arg for `isinstance` depends on how the scipy
        # import is written. If it is `import scipy` then we need
        # `scipy.sparse...` but if we have `import scipy.sparse as sparse`
        # then `sparse.coo....` will suffice.

        # LOGIC: if self.data is sparse just make references for later
        # otherwise, convert to sparse data.
        if isinstance(self.data, sparse.coo.coo_matrix):
            self.coo_data = self.data
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data
        elif isinstance(self.data, sparse.csr.csr_matrix):
            self.coo_data = self.data.tocoo()
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data
        else:
            self.coo_data = coo_matrix(data)
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data



        if self.proj_dim > self.n:
            raise Exception(f'Sketching with projection dimension\
                             $self.proj_dim > $self.n is not supported')

        ## For SRHT do the bit shift here so not timed in call
        ## to sketch
        if self.sketch_type is 'srht':
            # preprocess data so length is correct and repeat for
            # the intermediate function after hadamard transform
            next_power2_data = shift_bit_length(self.n)
            deficit = next_power2_data - self.n
            self.dense_data = np.concatenate((self.dense_data,
                                        np.zeros((deficit,self.d),
                                        dtype=self.dense_data.dtype)), axis=0)
            # set the new n for later use although
            # sampling will only ever be from self.n
            # as the power of 2 extension only necessary for
            # the hadamard transform
            self.new_n = self.dense_data.shape[0]

        ######################## SPARSE SKETCHES ##############################


        ### For sparse sketches generate hash functions here so
        # that they aren't timed in the call to sketch.
        # Would it be better to only accept sparse datasets for the sparse
        # sketches?
        elif self.sketch_type is 'sjlt' or 'countSketch':
            self.col_sparsity = col_sparsity
            if self.sketch_type is 'sjlt' and self.col_sparsity == 1:
                self.col_sparsity = 2

        ## Function dictionary to call later on.

        self.fct_dict = {'gaussian'    : self.GaussianSketch,
                         'srht'        : self.SRHT,
                         'countSketch' : self.CountSketch,
                         'sjlt'        : self.SparseJLT}


    def GaussianSketch(self):
        '''Compute the Gaussian random transform
        S_ij = G_ij ~ N(0,1) / sqrt(proj_dim)'''
        S = np.random.randn(self.proj_dim,self.n)
        S /= np.sqrt(self.proj_dim)
        return S@self.dense_data

    def SRHT(self):
        '''
        Compute the Subsampled Randomized Hadamard
        Transform (aka Fast Johnson Lindesntrauss Transform).

        Compute:
        PHDA where
        - DA is a diagonal matrix whose entries are
        {±1} chose uar
        - H (DA) applies the Hadamard transform on DA
        - P (HDA) uniformly subsamples HDA.


        Notes:
        The pyfht.fht_inplace doesn't seem to work so we
        need a little bit of working space (to store the
        FFTed versions of the columns) to do the
        entire transform'''
        diag = np.random.choice([1,-1], self.new_n)[:,None]
        signed_data = diag*self.dense_data

        # the [:,None] syntax is just to add a 2nd dimension
        # so that the columns after hadamard transform can
        # be easily appended.
        # It is slightly quicker to generate lists, call
        # np.array, then reshape than initialising a zero
        # array to store the fhted columns

        #Y = np.zeros((self.new_n,self.d))
        Y = []
        # perform the in place fht on each column
        for _col in range(self.d):
            # Y[:,_col] = pyfht.fht(self.data[:,_col])
            Y.append(pyfht.fht(self.dense_data[:,_col]))
        Y = np.array(Y)
        Y = Y.T
        #print(type(Y),Y.shape)
        #print(Y)
        # number from num_rows_data universe
        sample = np.sort(np.random.choice(self.n,
                                  self.proj_dim,
                                  replace=False))

        S = Y[sample] * (np.sqrt(1/self.proj_dim))
        return S

    def CountSketch(self):
        '''Compute the CountSketch transform of the data.
        This is just the  SJLT but with column sparsity 1.
        Given its own method to ensure speed is not
        implicated during later testing.'''
        self.SA = np.zeros((self.proj_dim,self.d))
        self._row_map = np.random.choice(self.proj_dim,
                                     self.n,
                                     replace=True)
        self._sign_map = np.random.choice(2, self.n, replace=True) * 2 - 1
        return fast_countSketch(self.SA,
                    self.rows,
                    self.cols,
                    self.vals,
                    self._sign_map,
                    self._row_map)

    def SparseJLT(self):
        '''Compute the SparseJLT of the data
        using the Kane-Nelson construction of
        concatenated CountSketches.

        1. Generate `s` independent countsketches
        each of size m/s x n and concatenate them.

        2. Use initial hash functions as decided above in the
        class definition and then generate new hashes for
        subsequent countsketch calls.'''
        # set the new projection dimension for sjlt
        # this is because the sjlt is an m x n sketch
        # composed of s*(m/s) x n shorter countSketches.
        self.sjlt_proj_dim = self.proj_dim // self.col_sparsity
        self.SA = np.zeros((self.sjlt_proj_dim,self.d))
        self._row_map = np.zeros((self.col_sparsity,self.n))
        self._sign_map = np.zeros((self.col_sparsity,self.n))
        # Generate array whose rows are lists for :
        # 1. row_map
        # 2. sign_map
        # Generate single array self.sjlt_proj_dim
        # to populate for the sketch to which new local
        # sketches will be added.

        for _ in range(self.col_sparsity):
            self._row_map[_,:] = np.random.choice(self.sjlt_proj_dim,
                                         self.n,
                                         replace=True)
            self._sign_map[_,:] = np.random.choice(2, self.n, replace=True) * 2 - 1
        self._row_map = self._row_map.astype(int)

        #print('size of SA ', self.SA.shape)
        #print('dType of row map ', self._row_map.dtype)
        local_row_map = self._row_map[0,:]
        local_sign_map = self._sign_map[0,:]
        global_summary = fast_countSketch(self.SA,
                    self.rows,
                    self.cols,
                    self.vals,
                    local_sign_map,
                    local_row_map)
        #print('global summary \n', global_summary)
        for sketch_id in range(1,self.col_sparsity):
            #print('sketch_id ', sketch_id+1)
            #print(self.SA)
            local_row_map = self._row_map[sketch_id,:]
            local_sign_map = self._sign_map[sketch_id,:]
            local_summary = fast_countSketch(np.zeros_like(self.SA),
                    self.rows,
                    self.cols,
                    self.vals,
                    local_sign_map,
                    local_row_map)
            global_summary = np.concatenate((global_summary,
                                             local_summary),axis=0)
        global_summary *= 1 / np.sqrt(self.col_sparsity)
        return global_summary

    def sketch(self):
        '''
        Perform the transform sketch(A) = SA
        '''
        return self.fct_dict[self.sketch_type]()

    def sketch_data_targets(self,targets):
        '''
        For classic sketching aka sketch-and-solve
        we need to sketch the matrix X where X is the
        matrix which has data|targets appended'''
        X = np.c_[self.data,targets]
        S_Ab = RandomProjection(X,self.proj_dim,self.sketch_type).sketch()
        SA = S_Ab[:,:-1]
        Sb = S_Ab[:,-1]
        #print(SA.shape, Sb.shape)
        return SA, Sb
