import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import ChebDiffMatrix
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cent_diff, cent_diff2
import scipy.sparse

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

def spectral_test(x, N, alpha, beta):
    p, q, r = eval_pqr1(x)
   
    [D, x_nodes] = cheb(N)
    [D2, x_nodes] = cheb2(N)
   
    # Construct matrix A and vector rhs
    A = D2[1:N, 1:N] + np.diag(p[1:N]) @ D[1:N, 1:N] + np.diag(q[1:N])
    rhs = r[1:N]
   
    # Apply boundary conditions
    rhs[0] -= (1/(x_nodes[1]-x_nodes[0])**2 - (1/(2*(x_nodes[1]-x_nodes[0])))*(-p[1])) * alpha
    rhs[-1] -= (1/(x_nodes[-1]-x_nodes[-2])**2 + (1/(2*(x_nodes[-1]-x_nodes[-2])))*(-p[-1])) * beta

   
    # Solve the linear system
    yapp = sp.linalg.spsolve(A, rhs)
   
    # Append boundary values to the solution
    yapp = np.concatenate(([alpha], yapp, [beta]))
   
    return x_nodes, yapp


def spectral(x, N, alpha, beta):
    p, q, r = eval_pqr2(x)
   
    [D, x_nodes] = cheb(N)
    [D2, x_nodes] = cheb2(N)
   
    # Construct matrix A and vector rhs
    A = D2[1:N, 1:N] + np.diag(p[1:N]) @ D[1:N, 1:N] + np.diag(q[1:N])
    rhs = r[1:N]
   
    # Apply boundary conditions
    rhs[0] -= (1/(x_nodes[1]-x_nodes[0])**2 - (1/(2*(x_nodes[1]-x_nodes[0])))*(-p[1])) * alpha
    rhs[-1] -= (1/(x_nodes[-1]-x_nodes[-2])**2 + (1/(2*(x_nodes[-1]-x_nodes[-2])))*(-p[-1])) * beta

   
    # Solve the linear system
    yapp = sp.linalg.spsolve(A, rhs)
   
    # Append boundary values to the solution
    yapp = np.concatenate(([alpha], yapp, [beta]))
   
    return x_nodes, yapp