import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import csc_matrix
import scipy.sparse as sp
import numpy.linalg as la1

import fd_bvp_demoSP
from fd_bvp_demoSP import eval_pqr1, eval_pqr2, make_FDmatDir_SP,make_FDmatDir_SP2

import fem_general_dir
from fem_general_dir import eval_k, eval_q, eval_f, eval_stiffD, eval_stiffO, eval_rhsInt1, eval_rhsInt2, make_Matrix, make_rhs

import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, spectral, spectral2, spectral3
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2
import scipy

def homog_BC():
    a = -1
    b = 1
    alpha = 0
    beta = 0

    N = np.arange(2, 51)
    h = (b - a)/N

    y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    

    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2(N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la1.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la1.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FEM
        x = np.linspace(a, b, N[i]+1)
        A = make_Matrix(x,h[i],N[i])
        rhs = make_rhs(x,h[i],N[i])
        sol = sp.linalg.spsolve(A,rhs)
        yapp = np.zeros(N[i]+1)
         
        for j in range(1,N[i]):
            yapp[j] = sol[j-1]

        error_fem[i] = la1.norm(yapp - y(x))

    ############################################################################

    plt.loglog(N, error_spectral, '-go', label = 'spectral')
    plt.loglog(N, error_fd, '-bo', label = 'fd')
    plt.loglog(N, error_fem, '-ro', label = 'fem')
    plt.legend()
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Convergence of BVP"
    plt.title(title1)
    plt.show()

def inhomog_BC():
    a = -1
    b = 1
    alpha = 0
    beta = 1

    N = np.arange(2, 51)
    h = (b - a)/N

    lp1 = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha
    lp = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha
    y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2(N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral(p_s, q_s, r_s, N[i], a, b, 0, 0) + lp(x)
        error_spectral[i] = la1.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la1.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FEM
        x = np.linspace(a, b, N[i]+1)
        A = make_Matrix(x,h[i],N[i])
        rhs = make_rhs(x,h[i],N[i])
        sol = sp.linalg.spsolve(A,rhs)
        yapp = np.zeros(N[i]+1)
         
        for j in range(1,N[i]):
            yapp[j] = sol[j-1]

        yapp = yapp + lp(x)

        error_fem[i] = la1.norm(yapp - y(x))

    ############################################################################

    plt.loglog(N, error_spectral, '-go', label = 'spectral')
    plt.loglog(N, error_fd, '-bo', label = 'fd')
    plt.loglog(N, error_fem, '-ro', label = 'fem')
    plt.legend()
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Convergence of BVP"
    plt.title(title1)
    plt.show()

def mixed_BC():
    a = -1
    b = 1
    alpha = 0
    beta = 0

    N = np.arange(2, 51)
    h = (b - a)/N

    y = lambda x: (np.exp(4*x) - 4*np.exp(-4)*(x - 1) - np.exp(4))/16

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2(N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral2(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la1.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        #x = np.linspace(a, b, N[i] + 1)
        x = np.linspace(a-h[i], b-h[i], N[i]+1)
        x = np.append(x, b)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP2(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la1.norm(yapp - y(x))

    ############################################################################

    plt.loglog(N, error_spectral, '-go', label = 'spectral')
    plt.loglog(N, error_fd, '-bo', label = 'fd')
    plt.legend()
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Convergence of BVP"
    plt.title(title1)
    plt.show()
    
homog_BC()
inhomog_BC()
mixed_BC()