import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import csc_matrix
import scipy.sparse as sp
import numpy.linalg as la

import fd_bvp_demoSP
from fd_bvp_demoSP import eval_pqr1, eval_pqr2, make_FDmatDir_SP,make_FDmatDir_SP2

import fem_general_dir
from fem_general_dir import eval_k, eval_q, eval_f, eval_stiffD, eval_stiffO, eval_rhsInt1, eval_rhsInt2, make_Matrix, make_rhs

import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, eval_pqr2, spectral, spectral2, spectral3, spectral4
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2
import scipy

##############################################################################################################################################################################################################################################################################################################################################################################################

# HOMOGENOUS DIRICHLET BC
def homog_dir_BC1():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: np.exp(1)*x-x-np.exp(x)+1

    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr1(x1)
    p2, q2, r2 = eval_pqr1(x2)
    p3, q3, r3 = eval_pqr1(x3)
    p4, q4, r4 = eval_pqr1(x4)

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta)

    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # axs[0, 0].plot(x1, yapp1t, color = 'green')
    # #axs[0, 0].plot(x1, y(x1))
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # #axs[0, 1].plot(x2, y(x2))
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # #axs[1, 0].plot(x3, y(x3))
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # #axs[1, 1].plot(x4, y(x4))
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x', ylabel = 'f(x)')

    # fig.tight_layout()
    
    #plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def homog_dir_BC_conv1():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: np.exp(1)*x-x-np.exp(x)+1

    N = np.arange(2, 51)
    h = (b - a)/N

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    

    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FEM
        x = np.linspace(a, b, N[i]+1)
        A = make_Matrix(x,h[i],N[i])
        rhs = make_rhs(x,h[i],N[i])
        sol = sp.linalg.spsolve(A,rhs)
        yapp = np.zeros(N[i]+1)
         
        for j in range(1,N[i]):
            yapp[j] = sol[j-1]

        error_fem[i] = la.norm(yapp - y(x))

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

# INHOMOGENOUS DIRICHLET BC
def inhomog_dir_BC1():
    a = 0
    b = 1
    y0 = -1
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    y = lambda x: np.exp(1)*x-x-np.exp(x)+1 + lp(x)
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr1(x1)
    p2, q2, r2 = eval_pqr1(x2)
    p3, q3, r3 = eval_pqr1(x3)
    p4, q4, r4 = eval_pqr1(x4)

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta) + lp(x1)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta) + lp(x2)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta) + lp(x3)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta) + lp(x4)
   
    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # axs[0, 0].plot(x1, yapp1t, color = 'green')
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x')

    # fig.tight_layout()
       
    #plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def inhomog_dir_BC_conv1():
    a = 0
    b = 1
    y0 = -1
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    y = lambda x: np.exp(1)*x-x-np.exp(x)+1 + lp(x)

    N = np.arange(2, 51)
    h = (b - a)/N

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral(p_s, q_s, r_s, N[i], a, b, 0, 0) + lp(x)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))
        
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

        error_fem[i] = la.norm(yapp - y(x))

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

# MIXED BC
def mixed_BC1():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: np.exp(1)*x-np.exp(x)+1
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr1(x1)
    p2, q2, r2 = eval_pqr1(x2)
    p3, q3, r3 = eval_pqr1(x3)
    p4, q4, r4 = eval_pqr1(x4)

    yapp1t = spectral2(p1, q1, r1, N1, a, b, alpha, beta) 
    yapp2t = spectral2(p2, q2, r2, N2, a, b, alpha, beta) 
    yapp3t = spectral2(p3, q3, r3, N3, a, b, alpha, beta) 
    yapp4t = spectral2(p4, q4, r4, N4, a, b, alpha, beta)
   
    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # axs[0, 0].plot(x1, yapp1t, color = 'green')
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x')

    # fig.tight_layout()
       
    # plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def mixed_BC_conv1():
    a = 0
    b = 1
    alpha = 0
    beta = 0

    N = np.arange(2, 51)
    h = (b - a)/N

    y = lambda x: np.exp(1)*x-np.exp(x)+1

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr1(x)
        yapp = spectral2(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        #x = np.linspace(a, b, N[i] + 1)
        x = np.linspace(a-h[i], b-h[i], N[i]+1)
        x = np.append(x, b)
        p_fd, q_fd, r_fd = eval_pqr1(x)
        yapp = make_FDmatDir_SP2(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))

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

##############################################################################################################################################################################################################################################################################################################################################################################################

# HOMOGENOUS DIRICHLET BC
def homog_dir_BC2():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: (-(np.exp(4)*x-x+np.exp(2-2*x)-np.exp(2*x+2)))/(1-np.exp(4))

    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr2(x1)
    p2, q2, r2 = eval_pqr2(x2)
    p3, q3, r3 = eval_pqr2(x3)
    p4, q4, r4 = eval_pqr2(x4)

    yapp1t = spectral3(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral3(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral3(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral3(p4, q4, r4, N4, a, b, alpha, beta)

    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # axs[0, 0].plot(x1, yapp1t, color = 'green')
    # #axs[0, 0].plot(x1, y(x1))
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # #axs[0, 1].plot(x2, y(x2))
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # #axs[1, 0].plot(x3, y(x3))
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # #axs[1, 1].plot(x4, y(x4))
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x', ylabel = 'f(x)')

    # fig.tight_layout()
    
    # plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def homog_dir_BC_conv2():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: (-(np.exp(4)*x-x+np.exp(2-2*x)-np.exp(2*x+2)))/(1-np.exp(4))

    N = np.arange(2, 51)
    h = (b - a)/N

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    

    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr2(x)
        yapp = spectral3(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr2(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FEM
        x = np.linspace(a, b, N[i]+1)
        A = make_Matrix(x,h[i],N[i])
        rhs = make_rhs(x,h[i],N[i])
        sol = sp.linalg.spsolve(A,rhs)
        yapp = np.zeros(N[i]+1)
         
        for j in range(1,N[i]):
            yapp[j] = sol[j-1]

        error_fem[i] = la.norm(yapp - y(x))

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

# INHOMOGENOUS DIRICHLET BC
def inhomog_dir_BC2():
    a = 0
    b = 1
    y0 = -2
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    y = lambda x: (-(np.exp(4)*x-x+np.exp(2-2*x)-np.exp(2*x+2)))/(1-np.exp(4)) + lp(x)
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr2(x1)
    p2, q2, r2 = eval_pqr2(x2)
    p3, q3, r3 = eval_pqr2(x3)
    p4, q4, r4 = eval_pqr2(x4)

    yapp1t = spectral3(p1, q1, r1, N1, a, b, alpha, beta) + lp(x1)
    yapp2t = spectral3(p2, q2, r2, N2, a, b, alpha, beta) + lp(x2)
    yapp3t = spectral3(p3, q3, r3, N3, a, b, alpha, beta) + lp(x3)
    yapp4t = spectral3(p4, q4, r4, N4, a, b, alpha, beta) + lp(x4)
   
    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # #axs[0, 0].plot(x1, yapp1t, color = 'green')
    # axs[0, 0].plot(x1, y(x1))
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # #axs[0, 1].plot(x2, y(x2))
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # #axs[1, 0].plot(x3, y(x3))
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # #axs[1, 1].plot(x4, y(x4))
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x')

    # fig.tight_layout()
       
    # plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def inhomog_dir_BC_conv2():
    a = 0
    b = 1
    y0 = -2
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    y = lambda x: (-(np.exp(4)*x-x+np.exp(2-2*x)-np.exp(2*x+2)))/(1-np.exp(4)) + lp(x)

    N = np.arange(2, 51)
    h = (b - a)/N

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    error_fem = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr2(x)
        yapp = spectral3(p_s, q_s, r_s, N[i], a, b, 0, 0) + lp(x)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        x = np.linspace(a, b, N[i]+1)
        p_fd, q_fd, r_fd = eval_pqr2(x)
        yapp = make_FDmatDir_SP(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))
        
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

        error_fem[i] = la.norm(yapp - y(x))

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

# MIXED BC
def mixed_BC2():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    
    y = lambda x: (2*np.exp(4)*x + 2*x + np.exp(2-2*x) - np.exp(2*x + 2))/(2 + 2*np.exp(4))
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr2(x1)
    p2, q2, r2 = eval_pqr2(x2)
    p3, q3, r3 = eval_pqr2(x3)
    p4, q4, r4 = eval_pqr2(x4)

    yapp1t = spectral4(p1, q1, r1, N1, a, b, alpha, beta) 
    yapp2t = spectral4(p2, q2, r2, N2, a, b, alpha, beta) 
    yapp3t = spectral4(p3, q3, r3, N3, a, b, alpha, beta) 
    yapp4t = spectral4(p4, q4, r4, N4, a, b, alpha, beta)

    # fig, axs = plt.subplots(2, 2)
    # title = "Spectral Collocation Approximation"
    # fig.suptitle(title)
    # axs[0, 0].plot(x1, yapp1t, color = 'green')
    # #axs[0, 0].plot(x1, y(x1))
    # axs[0, 0].set_title("N = 10")
    # axs[0, 1].plot(x2, yapp2t, color = 'green')
    # #axs[0, 1].plot(x2, y(x2))
    # axs[0, 1].set_title("N = 20")
    # axs[1, 0].plot(x3, yapp3t, color = 'green')
    # axs[1, 0].set_title("N = 50")
    # axs[1, 1].plot(x4, yapp4t, color = 'green')
    # axs[1, 1].set_title("N = 100")

    # for ax in axs.flat:
    #     ax.set(xlabel = 'x')

    # fig.tight_layout()
       
    # plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    fig.tight_layout()
       
    plt.show()

def mixed_BC_conv2():
    a = 0
    b = 1
    alpha = 0
    beta = 0

    N = np.arange(2, 51)
    h = (b - a)/N

    y = lambda x: (2*np.exp(4)*x + 2*x + np.exp(2-2*x) - np.exp(2*x + 2))/(2 + 2*np.exp(4))
   

    error_spectral = np.zeros(49)
    error_fd = np.zeros(49)
    
    for i in range(len(N)):
        # SPECTRAL
        [D2_N, x] = cheb2_ab(a, b, N[i])
        p_s, q_s, r_s = eval_pqr2(x)
        yapp = spectral4(p_s, q_s, r_s, N[i], a, b, alpha, beta)
        error_spectral[i] = la.norm(yapp - y(x))
        
    for i in range(len(N)):
        # FD
        #x = np.linspace(a, b, N[i] + 1)
        x = np.linspace(a-h[i], b-h[i], N[i]+1)
        x = np.append(x, b)
        p_fd, q_fd, r_fd = eval_pqr2(x)
        yapp = make_FDmatDir_SP2(x, p_fd, q_fd, r_fd, h[i], N[i], alpha, beta)
        error_fd[i] = la.norm(yapp - y(x))

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
    
homog_dir_BC1()
homog_dir_BC_conv1()
inhomog_dir_BC1()
inhomog_dir_BC_conv1()
mixed_BC1()
mixed_BC_conv1()

homog_dir_BC2()
homog_dir_BC_conv2()
inhomog_dir_BC2()
inhomog_dir_BC_conv2()
mixed_BC2()
mixed_BC_conv2()