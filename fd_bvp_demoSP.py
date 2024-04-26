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

def make_FDmatDir_SP(x, p, q, r, h, N, alpha, beta):
# create the finite difference matrix     
     Matypp = (1/h**2)*(sp.diags(2*np.ones(N-1)) -sp.diags(np.ones(N-2),-1) - 
           sp.diags(np.ones(N-2),1))
    
           
     Matyp = (1/(2*h))*(sp.diags(np.ones(N-2),1)-sp.diags(np.ones(N-2),-1))
     
     A = Matypp +sp.diags(p[1:N],0)@Matyp + sp.diags(q[1:N])

# create the right hand side rhs: (N-1) in size
     rhs = -r[1:N]
#  update with boundary data   
     rhs[0] = rhs[0] + ((1/h**2) - (1/(2*h))* - p[1])*alpha
     rhs[N-2] = rhs[N-2] + ((1/h**2) + (1/(2*h))* - p[N-1])*beta
   
# solve for the approximate solution
     sol = sp.linalg.spsolve(A,rhs)
     
     yapp = np.zeros(N+1)
     yapp[0] = alpha
    
     for j in range(1,N):
         yapp[j] = sol[j-1]
         
     yapp[-1] = beta  

     return yapp

def make_FDmatDir_SP2(x, p, q, r, h, N, alpha, beta):
# mixed BC
# create the finite difference matrix     
     Matypp = -(1/h**2)*(np.diag(2*np.ones(N+2)) -np.diag(np.ones(N+1),-1) - 
           np.diag(np.ones(N+1),1))
     
     Matyp = (1/(2*h))*(np.diag(np.ones(N+1),1)-np.diag(np.ones(N+1),-1))
     Matypp[-1,] = np.zeros(N+2)
     Matypp[-1, -1] = 1
     Matypp[0, ] = np.zeros(N+2)
     Matypp[0, 0] = -1/(2*h)
     Matypp[0, 2] = 1/(2*h)
     A = Matypp

     rhs = np.zeros(N+2)
     rhs[0] = alpha
     rhs[-1] = beta
     rhs[1:len(rhs)-1] = r[:N]

   
# solve for the approximate solution
     yapp = sp.linalg.spsolve(A,rhs)
     
     # yapp = np.zeros(N+1)
     # yapp[0] = alpha
    
     # for j in range(1,N):
     #     yapp[j] = sol[j-1]
         
     # yapp[-1] = beta  

     return yapp
