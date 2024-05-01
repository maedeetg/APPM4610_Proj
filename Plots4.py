import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import csc_matrix
import scipy.sparse as sp
import numpy.linalg as la1

import fd_bvp_demo
from fd_bvp_demo import eval_pqr1, eval_pqr2, make_FDmatDir

import fd_bvp_demoSP
from fd_bvp_demoSP import eval_pqr1, eval_pqr2, make_FDmatDir_SP

import fem_general_dir
from fem_general_dir import eval_k, eval_q, eval_f, eval_stiffD, eval_stiffO, eval_rhsInt1, eval_rhsInt2, make_Matrix, make_rhs

def fd_homog_BC():
     a = -1
     b = 1
     alpha = 0
     beta = 0

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)

     p1, q1, r1 = eval_pqr1(x1)
     p2, q2, r2 = eval_pqr1(x2)
     p3, q3, r3 = eval_pqr1(x3)
     p4, q4, r4 = eval_pqr1(x4)
     
     yapp1 = make_FDmatDir(x1,p1,q1,r1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir(x2,p2,q2,r2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir(x3,p3,q3,r3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir(x4,p4,q4,r4,h4,N4,alpha,beta)

     fig, axs = plt.subplots(2, 2)
     title = r"Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, yapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, yapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, yapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    ######################################################################

     y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
     
     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)
    
     for j in range(0,N1+1):
          err1[j] = abs(yapp1[j]-yex1[j])
         
     for j in range(0,N2+1):
          err2[j] = abs(yapp2[j]-yex2[j])
         
     for j in range(0,N3+1):
          err3[j] = abs(yapp3[j]-yex3[j])
         
     for j in range(0,N4+1):
          err4[j] = abs(yapp4[j]-yex4[j])

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    #############################################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         h = (b-a)/N[i]
         x = np.linspace(a, b, N[i]+1)
         p, q, r = eval_pqr1(x)
         yapp = make_FDmatDir(x, p, q, r, h, N[i], alpha, beta)
         err_norms[i] = la1.norm(yapp - y(x))

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Difference")
     plt.show()

     return

def fd_SP_homog_BC():
     a = -1
     b = 1
     alpha = 0
     beta = 0

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)

     p1, q1, r1 = eval_pqr1(x1)
     p2, q2, r2 = eval_pqr1(x2)
     p3, q3, r3 = eval_pqr1(x3)
     p4, q4, r4 = eval_pqr1(x4)
     
     yapp1 = make_FDmatDir_SP(x1,p1,q1,r1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir_SP(x2,p2,q2,r2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir_SP(x3,p3,q3,r3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir_SP(x4,p4,q4,r4,h4,N4,alpha,beta)

     fig, axs = plt.subplots(2, 2)
     title = r"Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, yapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, yapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, yapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    ######################################################################

     y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
     
     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)
    
     for j in range(0,N1+1):
          err1[j] = abs(yapp1[j]-yex1[j])
         
     for j in range(0,N2+1):
          err2[j] = abs(yapp2[j]-yex2[j])
         
     for j in range(0,N3+1):
          err3[j] = abs(yapp3[j]-yex3[j])
         
     for j in range(0,N4+1):
          err4[j] = abs(yapp4[j]-yex4[j])

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    #############################################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         h = (b-a)/N[i]
         x = np.linspace(a, b, N[i]+1)
         p, q, r = eval_pqr1(x)
         yapp = make_FDmatDir(x, p, q, r, h, N[i], alpha, beta)
         err_norms[i] = la1.norm(yapp - y(x))

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Difference")
     plt.show()

     return

def FEM_homog_BC():
     a = -1
     b = 1
     alpha = 0
     beta = 0

     u = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16
     
     # N = number of nodes +1;
     N1 = 2
     N2 = 20
     N3 = 50
     N4 = 100
    
    # space between nodes
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     xh1 = np.linspace(a,b,N1+1)
     xh2 = np.linspace(a,b,N2+1)
     xh3 = np.linspace(a,b,N3+1)
     xh4 = np.linspace(a,b,N4+1)

     A1 = make_Matrix(xh1,h1,N1)
     A2 = make_Matrix(xh2,h2,N2)
     A3 = make_Matrix(xh3,h3,N3)
     A4 = make_Matrix(xh4,h4,N4)
    
     rhs1 = make_rhs(xh1,h1,N1)
     rhs2 = make_rhs(xh2,h2,N2)
     rhs3 = make_rhs(xh3,h3,N3)
     rhs4 = make_rhs(xh4,h4,N4)
     
    # solve for the approximate solution 
    # at the interior nodes
     sol1 = sp.linalg.spsolve(A1,rhs1)
     sol2 = sp.linalg.spsolve(A2,rhs2)
     sol3 = sp.linalg.spsolve(A3,rhs3)
     sol4 = sp.linalg.spsolve(A4,rhs4)

    # create the vector with the approximations at the 
    # nodes     
     uapp1 = np.zeros(N1+1)
     uapp2 = np.zeros(N2+1)
     uapp3 = np.zeros(N3+1)
     uapp4 = np.zeros(N4+1)
    
     for j in range(1,N1):
         uapp1[j] = sol1[j-1]

     for j in range(1,N2):
         uapp2[j] = sol2[j-1]

     for j in range(1,N3):
         uapp3[j] = sol3[j-1]

     for j in range(1,N4):
         uapp4[j] = sol4[j-1]
         
     uex1 = u(xh1)
     print('exact', uex1)
     print('app', uapp1)
     uex2 = u(xh2)
     uex3 = u(xh3)
     uex4 = u(xh4)

     fig, axs = plt.subplots(2, 2)
     title = "Finite Element Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(xh1, uapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(xh2, uapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(xh3, uapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(xh4, uapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
       
     plt.show()

    #######################################################

     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)

     for j in range(0,N1+1):
          err1[j] = uapp1[j]-uex1[j]
         
     for j in range(0,N2+1):
          err2[j] = uapp2[j]-uex2[j]
         
     for j in range(0,N3+1):
          err3[j] = uapp3[j]-uex3[j]
         
     for j in range(0,N4+1):
          err4[j] = uapp4[j]-uex4[j]

     fig, axs = plt.subplots(2, 2)
     title = "Error in Finite Element Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(xh1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(xh2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(xh3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(xh4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
       
     plt.show()

    #######################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         
         h = (b-a)/N[i]
         xh = np.linspace(a, b, N[i]+1)
         A = make_Matrix(xh,h,N[i])
         rhs = make_rhs(xh,h,N[i])
         sol = sp.linalg.spsolve(A,rhs)

         uapp = np.zeros(N[i]+1)
         
         for j in range(1,N[i]):
             uapp[j] = sol[j-1]

         uex = u(xh)
         err_norms[i] = la1.norm(uapp - uex)

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Element")
     plt.show()

     return

#############################################################################################################################################

def fd_inhomog_BC1():
     a = -1
     b = 1
     alpha = 0
     beta = 1

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)

     p1, q1, r1 = eval_pqr1(x1)
     p2, q2, r2 = eval_pqr1(x2)
     p3, q3, r3 = eval_pqr1(x3)
     p4, q4, r4 = eval_pqr1(x4)
     
     yapp1 = make_FDmatDir(x1,p1,q1,r1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir(x2,p2,q2,r2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir(x3,p3,q3,r3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir(x4,p4,q4,r4,h4,N4,alpha,beta)

     fig, axs = plt.subplots(2, 2)
     title = r"Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, yapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, yapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, yapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    ######################################################################

     lp = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha
    
     y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
     
     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)
    
     for j in range(0,N1+1):
          err1[j] = yapp1[j]-yex1[j]
         
     for j in range(0,N2+1):
          err2[j] = yapp2[j]-yex2[j]
         
     for j in range(0,N3+1):
          err3[j] = yapp3[j]-yex3[j]
         
     for j in range(0,N4+1):
          err4[j] = yapp4[j]-yex4[j]

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    #############################################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         h = (b-a)/N[i]
         x = np.linspace(a, b, N[i]+1)
         p, q, r = eval_pqr1(x)
         yapp = make_FDmatDir(x, p, q, r, h, N[i], alpha, beta)
         err_norms[i] = la1.norm(yapp - y(x))

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Difference")
     plt.show()

     return

def fd_SP_inhomog_BC1():
     a = -1
     b = 1
     alpha = 0
     beta = 1

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)

     p1, q1, r1 = eval_pqr1(x1)
     p2, q2, r2 = eval_pqr1(x2)
     p3, q3, r3 = eval_pqr1(x3)
     p4, q4, r4 = eval_pqr1(x4)
     
     yapp1 = make_FDmatDir_SP(x1,p1,q1,r1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir_SP(x2,p2,q2,r2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir_SP(x3,p3,q3,r3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir_SP(x4,p4,q4,r4,h4,N4,alpha,beta)

     fig, axs = plt.subplots(2, 2)
     title = r"Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, yapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, yapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, yapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    ######################################################################

     lp = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha
    
     y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
     
     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)
    
     for j in range(0,N1+1):
          err1[j] = yapp1[j]-yex1[j]
         
     for j in range(0,N2+1):
          err2[j] = yapp2[j]-yex2[j]
         
     for j in range(0,N3+1):
          err3[j] = yapp3[j]-yex3[j]
         
     for j in range(0,N4+1):
          err4[j] = yapp4[j]-yex4[j]

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    #############################################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         h = (b-a)/N[i]
         x = np.linspace(a, b, N[i]+1)
         p, q, r = eval_pqr1(x)
         yapp = make_FDmatDir(x, p, q, r, h, N[i], alpha, beta)
         err_norms[i] = la1.norm(yapp - y(x))

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Difference")
     plt.show()

     return

def FEM_inhomog_BC1():
     a = -1
     b = 1
     alpha = 0
     beta = 1

     lp = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha

     u = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)
     
     # N = number of nodes +1;
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
    # space between nodes
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     xh1 = np.linspace(a,b,N1+1)
     xh2 = np.linspace(a,b,N2+1)
     xh3 = np.linspace(a,b,N3+1)
     xh4 = np.linspace(a,b,N4+1)

     A1 = make_Matrix(xh1,h1,N1)
     A2 = make_Matrix(xh2,h2,N2)
     A3 = make_Matrix(xh3,h3,N3)
     A4 = make_Matrix(xh4,h4,N4)
    
     rhs1 = make_rhs(xh1,h1,N1)
     rhs2 = make_rhs(xh2,h2,N2)
     rhs3 = make_rhs(xh3,h3,N3)
     rhs4 = make_rhs(xh4,h4,N4)
     
    # solve for the approximate solution 
    # at the interior nodes
     sol1 = sp.linalg.spsolve(A1,rhs1) 
     sol2 = sp.linalg.spsolve(A2,rhs2) 
     sol3 = sp.linalg.spsolve(A3,rhs3)
     sol4 = sp.linalg.spsolve(A4,rhs4)

    # create the vector with the approximations at the 
    # nodes     
     uapp1 = np.zeros(N1+1)
     uapp2 = np.zeros(N2+1)
     uapp3 = np.zeros(N3+1)
     uapp4 = np.zeros(N4+1)
    
     for j in range(1,N1):
         uapp1[j] = -sol1[j-1]

     for j in range(1,N2):
         uapp2[j] = -sol2[j-1]

     for j in range(1,N3):
         uapp3[j] = -sol3[j-1]

     for j in range(1,N4):
         uapp4[j] = -sol4[j-1]

     uapp1 = uapp1 + lp(xh1)
     uapp2 = uapp2 + lp(xh2)
     uapp3 = uapp3 + lp(xh3)
     uapp4 = uapp4 + lp(xh4)
         
     uex1 = u(xh1)
     uex2 = u(xh2)
     uex3 = u(xh3)
     uex4 = u(xh4)

     fig, axs = plt.subplots(2, 2)
     title = "Finite Element Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(xh1, uapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(xh2, uapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(xh3, uapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(xh4, uapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
       
     plt.show()

    #######################################################

     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)

     for j in range(0,N1+1):
          err1[j] = uapp1[j]-uex1[j]
         
     for j in range(0,N2+1):
          err2[j] = uapp2[j]-uex2[j]
         
     for j in range(0,N3+1):
          err3[j] = uapp3[j]-uex3[j]
         
     for j in range(0,N4+1):
          err4[j] = uapp4[j]-uex4[j]

     fig, axs = plt.subplots(2, 2)
     title = "Error in Finite Element Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(xh1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(xh2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(xh3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(xh4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
       
     plt.show()

    #######################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         
         h = (b-a)/N[i]
         xh = np.linspace(a, b, N[i]+1)
         A = make_Matrix(xh,h,N[i])
         rhs = make_rhs(xh,h,N[i])
         sol = sp.linalg.spsolve(A,rhs)

         uapp = np.zeros(N[i]+1)
         
         for j in range(1,N[i]):
             uapp[j] = -sol[j-1]

         uapp = uapp + lp(xh)

         uex = u(xh)
         err_norms[i] = la1.norm(uapp - uex)

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Element")
     plt.show()

     return

#############################################################################################################################################

def fd_neumann_BC():
     a = -1
     b = 1
     alpha = 0
     beta = 0

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 = (b-a)/N1
     h2 = (b-a)/N2
     h3 = (b-a)/N3
     h4 = (b-a)/N4
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)

     p1, q1, r1 = eval_pqr1(x1)
     p2, q2, r2 = eval_pqr1(x2)
     p3, q3, r3 = eval_pqr1(x3)
     p4, q4, r4 = eval_pqr1(x4)
     
     yapp1 = make_FDmatDir(x1,p1,q1,r1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir(x2,p2,q2,r2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir(x3,p3,q3,r3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir(x4,p4,q4,r4,h4,N4,alpha,beta)

     fig, axs = plt.subplots(2, 2)
     title = r"Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, yapp2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, yapp3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, yapp4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    ######################################################################

     lp = lambda x: ((x - a)/(b - a))*beta + ((x - b)/(a - b))*alpha
    
     y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
     
     err1 = np.zeros(N1+1)
     err2 = np.zeros(N2+1)
     err3 = np.zeros(N3+1)
     err4 = np.zeros(N4+1)
    
     for j in range(0,N1+1):
          err1[j] = yapp1[j]-yex1[j]
         
     for j in range(0,N2+1):
          err2[j] = yapp2[j]-yex2[j]
         
     for j in range(0,N3+1):
          err3[j] = yapp3[j]-yex3[j]
         
     for j in range(0,N4+1):
          err4[j] = yapp4[j]-yex4[j]

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, err1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, err4, color = 'green')
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()

    #############################################################################

     N = np.arange(2, 51)
     err_norms = np.zeros(49)

     for i in range(len(N)):
         h = (b-a)/N[i]
         x = np.linspace(a, b, N[i]+1)
         p, q, r = eval_pqr1(x)
         yapp = make_FDmatDir(x, p, q, r, h, N[i], alpha, beta)
         err_norms[i] = la1.norm(yapp - y(x))

     plt.semilogy(N, err_norms, '-go')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Accuracy of BVP with Finite Difference")
     plt.show()

     return

#fd_homog_BC()
fd_SP_homog_BC()
FEM_homog_BC()
# fd_inhomog_BC1()
fd_SP_inhomog_BC1()
FEM_inhomog_BC1()