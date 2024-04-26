import numpy as np
import math
import matplotlib.pyplot as plt

def driver():

# this code illustrates how to solve a mixed BC ODE 
# via spectral collocation

#    this code solves  y'' = f(x), y(-1) = alpha, y'(1) = beta
     uex = lambda x: (np.exp(4*x) -x*math.sinh(4)-
         math.cosh(4))/16
     uex_p = lambda x: (4*np.exp(4*x)-math.sinh(4))/16
     alpha = 0
     beta = 0
     print(alpha, beta)

# Ntest = number of tests done
     Ntest = 7
     err = np.zeros(Ntest)
# NN = orders to test
     N = 5
     (A, x) = mixedBCend_system(N)
     rhs = make_rhs(x,alpha,beta,N)
     Ainv = np.linalg.inv(A)
     yapp = np.matmul(Ainv,rhs)
     print('app', yapp)
#      NN = [5, 10, 20, 40,80,160,320]
     
#      for j in range(0,Ntest):    
     
# #     N = order of approx.
#        N = NN[j]

# # create the linear system     
#        (A,x) = mixedBCend_system(N)
#        print('A', A)
     
# # create the right hand side     
#        rhs = make_rhs(x,alpha,beta,N)
#        print('rhs', rhs)
     
# #   Create the approximate solution     
#        Ainv = np.linalg.inv(A)
#        yapp = np.matmul(Ainv,rhs)
     
#        uu = uex(x)
#        err[j] = np.linalg.norm(uu-yapp)
     
#      print(err)
#      plt.semilogy(NN,err,label = 'Spectral approx')
#      plt.xlabel('N')
#      plt.xlabel('norm absolute error')
#      plt.legend(loc = 'upper left')
#      plt.show()
     return
     
def make_rhs(x,alpha,beta,N):
     f = lambda x: np.exp(4*x)
     
     rhs = np.zeros(N+1)
     
     rhs = f(x)
     rhs[0] = beta
     rhs[N] = alpha

     return(rhs)     

def mixedBCend_system(N):

     [x,D] = cheb(N)
     D2 = np.matmul(D, D)
     #print('D2', D2)

     A = np.zeros((N+1,N+1))
     A[1:N,:] = D2[1:N,:]
     A[0,:] = D[0,:]
     A[N,N] = 1
     
     return(A,x)

def cheb(n):
    N = n
    x = np.cos((np.pi * np.arange(N + 1)) / N)
    diag = np.array(-x[1:-1]/(2*(1 - x[1:-1]**2)))
    diag = np.append(x[1], diag)
    diag = np.append(diag, x[-1])
    
    D_N = np.diag(diag)

    # first and last enteries
    D_N[0, 0] = (2*N**2 + 1) / 6
    D_N[N, N] = -(2*N**2 + 1) / 6

    if (N == 1):
        D_N[0, N] = (1/2)*(-1)**N
        D_N[N, 0] = -(1/2)*(-1)**N
        
        return (D_N, x)
    else:
        for i in range(N+1):
            for j in range(N+1):
                if (i != j):
                    if ((i == 0) and (j != N)):
                        D_N[i, j] = (2*(-1)**j) / (1 - x[j])
                    elif ((i == N) and (j != N)):
                        D_N[i, j] = -(2*(-1)**(N+j)) / (1 + x[j])
                    elif ((j == 0) and (i != N)):
                        D_N[i, j] = (-1/2)*((-1)**i / (1 - x[i]))
                    elif ((j == N) and (i != N)):
                        D_N[i, j] = (1/2)*((-1)**(N+i) / (1 + x[i]))
                    elif ((1 < i < N) or (1 < j < N)):
                        D_N[i, j] = (-1)**(i+j) / (x[i] - x[j])
        
        D_N[0, N] = (1/2)*(-1)**N
        D_N[N, 0] = -(1/2)*(-1)**N

    return (x, D_N)
# def cheb(n):
# # build chebychev nodes and derivative 
# # matrix

#      if n == 0:
#          x = 1
#          D = 0 
#          return(x,D)
     
#      x = np.cos(np.pi*np.linspace(0,1,n+1))
#      c = np.ones(n+1)*alt(n+1)
#      c[0] = 2
#      c[n] = 2
#      X = np.outer(x, np.ones(n+1))
#      dX = X-X.T
#      D = np.outer(c, np.array([1]*(n+1))/c) / (dX 
#       + np.identity(n+1))
#      D = D - np.diag(np.sum(D,axis=1))
#      return(x,D) 

def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1)**i)
    return np.array(alt)	     
     
driver()         