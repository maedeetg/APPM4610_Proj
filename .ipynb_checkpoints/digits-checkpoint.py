import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import csc_matrix
import scipy.sparse as sp
import numpy.linalg as la1

import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, eval_pqr2, eval_pqr3, eval_pqr4, eval_pqr5, spectral
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2

import fd_bvp_demo
from fd_bvp_demo import eval_pqr, make_FDmatDir

import fd_bvp_demoSP
from fd_bvp_demoSP import eval_pqr1, eval_pqr2, make_FDmatDir_SP

import fem_general_dir
from fem_general_dir import eval_k, eval_q, eval_f, eval_stiffD, eval_stiffO, eval_rhsInt1, eval_rhsInt2, make_Matrix, make_rhs

a = -1
b = 1
alpha = 0
beta = 0

y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16

def digits_spectral():
    N = 2
    [D_N, x] = cheb2(N)
    p1, q1, r1 = eval_pqr3(x)
    yapp1t = spectral(p1, q1, r1, N, a, b, alpha, beta)
    yapp_0 = yapp1t[len(yapp1t)//2]
    yex_0 = y(x)[len(yapp1t)//2]
    rel_err = abs(yapp_0 - yex_0)/abs(yex_0)

    while (rel_err > 10**(-20)):
        N = N + 1
        if (N % 2 == 0):
            [D_N, x] = cheb2(N)
            p1, q1, r1 = eval_pqr3(x)
            yapp1t = spectral(p1, q1, r1, N, a, b, alpha, beta)
            yapp_0 = yapp1t[len(yapp1t)//2]
            yex_0 = y(x)[len(yapp1t)//2]
            rel_err = abs(yapp_0 - yex_0)/abs(yex_0)
        else:
            [D_N, x] = cheb2(N)
            p1, q1, r1 = eval_pqr3(x)
            yapp1t = spectral(p1, q1, r1, N, a, b, alpha, beta)
            yapp_0 = (yapp1t[len(yapp1t)//2] + yapp1t[len(yapp1t)//2]) / 2
            yex_0 = (y(x)[len(yapp1t)//2] + y(x)[len(yapp1t)//2]) / 2
            rel_err = abs(yapp_0 - yex_0)/abs(yex_0)
    print("SPECTRAL COLLOCATION N =", N)
    return

def digits_fd():
    N = 2
    h = int((b-a)/N)
    x = np.linspace(a, b, N+1)
    yapp = make_FDmatDir(x, h, N, alpha, beta)
        
    yapp_0 = yapp[len(yapp)//2]
    yex_0 = y(x)[len(yapp)//2]

    rel_err = abs(yapp_0 - yex_0) / abs(yex_0)

    while (rel_err > 10**(-9)):
        N = N + 1
        if (N % 2 == 0):
            h = int((b-a)/N)
            x = np.linspace(a, b, N+1)
            yapp = make_FDmatDir(x, h, N, alpha, beta)
        
            yapp_0 = yapp[len(yapp)//2]
            yex_0 = y(x)[len(yapp)//2]
        
            rel_err = abs(yapp_0 - yex_0) / abs(yex_0)
        else:
            h = int((b-a)/N)
            x = np.linspace(a, b, N+1)
            yapp = make_FDmatDir(x, h, N, alpha, beta)
        
            yapp_0 = (yapp[len(yapp)//2] + yapp[len(yapp)//2 - 1]) / 2
            yex_0 = (y(x)[len(yapp)//2] + y(x)[len(yapp)//2 - 1]) / 2
        
            rel_err = abs(yapp_0 - yex_0) / abs(yex_0)
    print("FINITE DIFFERENCE N =", N)
    return

def digits_fd_SP():
    N = 2
    h = int((b-a)/N)
    x = np.linspace(a, b, N+1)
    yapp = make_FDmatDir_SP(x, h, N, alpha, beta)
        
    yapp_0 = yapp[len(yapp)//2]
    yex_0 = y(x)[len(yapp)//2]

    rel_err = abs(yapp_0 - yex_0) / abs(yex_0)

    while (rel_err > 10**(-9)):
        N = N + 1
        if (N % 2 == 0):
            h = int((b-a)/N)
            x = np.linspace(a, b, N+1)
            yapp = make_FDmatDir_SP(x, h, N, alpha, beta)
        
            yapp_0 = yapp[len(yapp)//2]
            yex_0 = y(x)[len(yapp)//2]
        
            rel_err = abs(yapp_0 - yex_0) / abs(yex_0)
        else:
            h = int((b-a)/N)
            x = np.linspace(a, b, N+1)
            yapp = make_FDmatDir_SP(x, h, N, alpha, beta)
        
            yapp_0 = (yapp[len(yapp)//2] + yapp[len(yapp)//2 - 1]) / 2
            yex_0 = (y(x)[len(yapp)//2] + y(x)[len(yapp)//2 - 1]) / 2
        
            rel_err = abs(yapp_0 - yex_0) / abs(yex_0)
    
    print("FINITE DIFFERENCE N =", N)
    return

def digits_FEM():
    N = 2
    h = int((b-a)/N)
    xh = np.linspace(a, b, N+1)
    A = make_Matrix(xh,h,N)
    rhs = make_rhs(xh,h,N)
    sol = sp.linalg.spsolve(A,rhs)

    uapp = np.zeros(N+1)
     
    for j in range(1,N):
        uapp[j] = -sol[j-1]

    uex = y(xh)
    
    yapp_0 = uapp[len(xh)//2]
    yex_0 = uex[len(xh)//2]

    rel_err = abs(yapp_0 - yex_0)/abs(yex_0)

    while (rel_err > 10**(-20)):
        N = N + 1
        if (N % 2 == 0):
            h = int((b-a)/N)
            xh = np.linspace(a, b, N+1)
            A = make_Matrix(xh,h,N)
            rhs = make_rhs(xh,h,N)
            sol = sp.linalg.spsolve(A,rhs)
        
            uapp = np.zeros(N+1)
             
            for j in range(1,N):
                uapp[j] = -sol[j-1]
        
            uex = u(xh)
    
            yapp_0 = uapp[len(xh)//2]
            yex_0 = uex[len(xh)//2]
        
            rel_err = abs(yapp_0 - yex_0)/abs(yex_0)
             
        else:
            h = int((b-a)/N)
            xh = np.linspace(a, b, N+1)
            A = make_Matrix(xh,h,N)
            rhs = make_rhs(xh,h,N)
            sol = sp.linalg.spsolve(A,rhs)
        
            uapp = np.zeros(N+1)
             
            for j in range(1,N):
                uapp[j] = -sol[j-1]
        
            uex = u(xh)

            yapp_0 = (uapp[len(xh)//2] + uapp[len(xh)//2 - 1]) / 2
            yex_0 = (uex[len(xh)//2] + uex[len(xh)//2]) / 2
        
            rel_err = abs(yapp_0 - yex_0)/abs(yex_0)
    print("FINITE ELEMENT N =", N)
    return

digits_spectral()
# digits_fd()
# digits_fd_SP()
digits_FEM()