import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la1
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la2

def eval_pqr(x):
     p = 0*x
     q = 0*x
     r = np.exp(4*x)
     
     return(p,q,r)      

def make_FDmatDir(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr(x)
    
      
# create the finite difference matrix     
     Matypp = (1/h**2)*(np.diag(2*np.ones(N-1)) -np.diag(np.ones(N-2),-1) - 
           np.diag(np.ones(N-2),1))
           
     Matyp = (1/(2*h))*(np.diag(np.ones(N-2),1)-np.diag(np.ones(N-2),-1))
     
     B = np.matmul(np.diag(p[1:N],0),Matyp)
     
     A = Matypp + B+ np.diag(q[1:N])

# create the right hand side rhs: (N-1) in size
     rhs = -r[1:N]
#  update with boundary data   
     rhs[0] = rhs[0] + ((1/h**2) - (1/(2*h)) - p[1])*alpha
     rhs[N-2] = rhs[N-2] + ((1/h**2) + (1/(2*h)) - p[N-1])*beta
     
# solve for the approximate solution

     Ainv = np.linalg.inv(A)
     sol = np.matmul(Ainv,rhs)
     
     yapp = np.zeros(N+1)
     yapp[0] = alpha
     for j in range(1,N):
         yapp[j] = sol[j-1]
     yapp[N] = beta    

     return yapp