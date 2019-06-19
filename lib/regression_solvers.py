import numpy as np
import cvxopt as cp


def lasso_solver(data,targets, constraint):
    '''solve using cvxopt'''
    n,d = data.shape
    Q = data.T@data
    c = data.T@targets

    # Expand the problem
    big_Q = np.vstack((np.c_[Q, -1.0*Q], np.c_[-1.0*Q, Q]))
    big_c = np.concatenate((c,-c))

    # penalty term
    constraint_term = constraint*np.ones((2*d,))
    big_linear_term = constraint_term - big_c

    # nonnegative constraints
    G = -1.0*np.eye(2*d)
    h = np.zeros((2*d,))

    P = cp.matrix(big_Q)
    q = cp.matrix(big_linear_term)
    G = cp.matrix(G)
    h = cp.matrix(h)


    res = cp.solvers.qp(P,q,G,h)
    w = np.squeeze(np.array(res['x']))
    w[w < 1E-8] = 0
    x = w[:d] - w[d:]
    return(x)



def iterative_lasso(sketch_data,ATy, data, targets, x0, ell_1_bound):
    '''solve the lasso through repeated calls to a smaller quadratic program'''

    # Expand the problem
    n,d = data.shape
    Q = sketch_data.T@sketch_data #+ 1E-10*np.eye(d)
    big_Q = np.vstack((np.c_[Q, -1.0*Q], np.c_[-1.0*Q, Q])) #+ 1E-3*np.eye(2*d)
    #print('Rank of Q: {}'.format(np.linalg.matrix_rank(Q)))


    linear_term = Q@x0 + ATy - data.T@(data@x0)
    big_c = np.concatenate((linear_term,-linear_term))

    # penalty term
    constraint_term = ell_1_bound*np.ones((2*d,))
    big_linear_term = constraint_term - big_c

    # nonnegative constraints
    G = -1.0*np.eye(2*d,dtype=np.float64)
    h = np.zeros((2*d,),dtype=np.float64)


    P = cp.matrix(big_Q)
    q = cp.matrix(big_linear_term)
    G = cp.matrix(G)
    h = cp.matrix(h)

    res = cp.solvers.qp(P,q,G,h)
    #w = qpsolvers.solve_qp(big_Q,big_linear_term,G,h,solver='cvxopt')
    #w = quadprog.solve_qp(big_Q,-1.0*big_linear_term,-1.0*G,h) # using quadprog
    #print(solve_time)
    w = np.squeeze(np.array(res['x']))
    x = w[:d] - w[d:]

    return x
