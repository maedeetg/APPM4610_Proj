import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import ChebDiffMatrix
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la2

def eval_pqr1(x):
    # y'' = p(x)y'+q(x)y + r(x)
    p = np.zeros(len(x))
    q = np.zeros(len(x))
    r = 4*np.exp(2*x)
    return p, q, r

def eval_pqr2(x):
    # y'' = p(x)y'+q(x)y + r(x)
    p = -4*np.ones(len(x))
    q = -np.exp(x)
    r = np.sin(8*x)
    return p, q, r

def eval_pqr3(x):
    # y'' = p(x)y'+q(x)y + r(x)
    p = np.zeros(len(x))
    q = np.zeros(len(x))
    r = np.exp(4*x)
    return p, q, r

def eval_pqr4(x):
    p = np.zeros(len(x))
    q = 4*np.ones(len(x))
    r = -4*x
    return p, q, r

def eval_pqr5(x):
    p = np.zeros(len(x))
    q = -4*np.ones(len(x))
    r = np.cos(x)
    return p, q, r

def spectral(p, q, r, N, a, b, alpha, beta):
    [D2, x_nodes] = cheb2_ab(a, b, N)
    
    A = D2
    
    for i in range(1, N+1):
        A[0, i] = 0
        
    for j in range(0, N):
        A[N, j] = 0

    A[0, 0] = 1
    A[N, N] = 1
    
    rhs = r
    rhs[0] = beta
    rhs[-1] = alpha

    yapp = la.inv(A)@rhs
   
    return yapp

# def spectral2(p, q, r, N, a, b, alpha, beta):
#     [D2, x_nodes] = cheb2_ab(a, b, N)
#     # print(la.inv(D2)@r)

#     A = D2[1:(N), 1:(N)]
#     rhs = r[1:(N),]
    
#     yapp = np.zeros(N+1)
#     yapp[1:N,] = la.inv(A)@rhs
#     yapp[0] = 0
#     yapp[-1] = 0
#     yapp = yapp + (x_nodes + 1)/2
   
#     return yapp

