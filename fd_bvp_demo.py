import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la1
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la2

def eval_pqr1(x):
     p = 0*x
     q = 0*x
     r = np.exp(4*x)
     
     return(p,q,r)    

def eval_pqr2(x):
     p = 0*x
     q = 0*x
     r = 4*np.exp(2*x)
     
     return(p,q,r)  

def make_FDmatDir(x, p, q, r, h, N, alpha, beta):
# create the finite difference matrix  
     # A = np.diag(np.ones(N + 1))
     # A 
     Matypp = (1/(h**2))*(np.diag(2*np.ones(N+1)) -np.diag(np.ones(N),-1) - 
           np.diag(np.ones(N),1))
    
     Matyp = (1/(2*h))*(np.diag(np.ones(N),1)-np.diag(np.ones(N),-1))

     B = np.matmul(np.diag(p),Matyp)
     
     A = Matypp + B + np.diag(q)

     A[0,] = np.zeros(len(A))
     A[-1,] = np.zeros(len(A))

     A[0, 0] = 1
     A[-1, -1] = 1

# create the right hand side rhs: (N-1) in size
     rhs = np.zeros(N+1)
     rhs[0] = alpha
     rhs[-1] = beta
     rhs[1:len(rhs)-1] = -r[1:N]
#  update with boundary data   
     # rhs[0] = rhs[0] + ((1/h**2) - (1/(2*h)) - p[1])*alpha
     # rhs[N-2] = rhs[N-2] + ((1/h**2) + (1/(2*h)) - p[N-1])*beta
     
# solve for the approximate solution

     Ainv = np.linalg.inv(A)
     yapp = np.matmul(Ainv,rhs)
     
     # yapp = np.zeros(N+1)
     # for j in range(1,N):
     #     yapp[j] = sol[j-1]   

     return yapp