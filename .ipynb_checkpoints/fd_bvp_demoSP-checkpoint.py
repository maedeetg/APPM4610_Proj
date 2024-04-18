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
     r = 4*np.exp(2*x)
     
     return(p,q,r)    

def eval_pqr2(x):
     p = 0*x
     q = 0*x
     r = np.exp(4*x)
     
     return(p,q,r)    

def make_FDmatDir_SP(x,h,N,alpha,beta):
# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr2(x)
     
# create the finite difference matrix     
     Matypp = (N/2)**2*(sp.diags(2*np.ones(N-1)) -sp.diags(np.ones(N-2),-1) - 
           sp.diags(np.ones(N-2),1))
     #print(Matypp)
           
     Matyp = (N/4)*(sp.diags(np.ones(N-2),1)-sp.diags(np.ones(N-2),-1))
     #print(Matypp)

     A = Matypp +sp.diags(p[1:N],0)@Matyp + sp.diags(q[1:N])

# create the right hand side rhs: (N-1) in size
     rhs = -r[1:N]
#  update with boundary data   
     rhs[0] = rhs[0] + ((N/2)**2 - (N/4)*-p[1])*alpha
     rhs[N-2] = rhs[N-2] + ((N/2)**2+(N/4)*-p[N-1])*beta
   
# solve for the approximate solution
     sol = sp.linalg.spsolve(A,rhs)
     
     yapp = np.zeros(N+1)
     yapp[0] = alpha
    
     for j in range(1,N):
         yapp[j] = sol[j-1]
         
     yapp[N] = beta  

     return yapp