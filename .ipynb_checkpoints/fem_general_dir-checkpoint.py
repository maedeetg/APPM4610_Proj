import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.sparse import csc_matrix
import scipy.sparse as sp
import numpy.linalg as la1

def eval_k(x):
    k = -1
    return k
      
def eval_q(x):
    q = 0
    return q        

def eval_f(x):
    f = np.exp(4*x)
    return f
      
def eval_stiffD(x,xj):
       # evaluates integrand for diagonal of 
       # stiffness matrix
     val = eval_q(x)*(x-xj)**2
     return val        

def eval_stiffO(x,xj,xk):
      # evaluates integrand of off-diag of 
      # stiffness matrix
     val = eval_q(x)*(x-xj)*(xk-x)
     return val        

def eval_rhsInt1(x,xk):
     # evaluate the integrand for the rhs 
     # left side of hat
     val = eval_f(x)*(x-xk)
     return val
     
def eval_rhsInt2(x,xk):
     # evaluate the integrand for the rhs 
     # right side of hat
     val = eval_f(x)*(xk-x)
     return val
          
def make_Matrix(x,h,N):
     # make the entries of the mass matrix
     d_main = np.zeros(N-1)
     d_sub = np.zeros(N-2)
     d_sup = np.zeros(N-2)
     
     # make the diagonal entries
     for j in range(0,N-1):
         a = x[j]
         b = x[j+1]
         c = x[j+2]
         tmp1 = quad(eval_k,a,b)
         tmp2 = quad(eval_k,b,c)
         tmp = tmp1[0] + tmp2[0]
         d_main[j] = (1/h**2)*tmp
     
     for j in range(0,N-2):
        b = x[j+1]
        c = x[j+2]     
        tmp1 = quad(eval_k,b,c)
        tmp2 = quad(eval_k,b,c) 
        d_sub[j] = -(1/h**2)*tmp1[0]
        d_sup[j] = -(1/h**2)*tmp2[0]
        
# make entries for the stiffness matrix 
     for j in range(0,N-1):
         a = x[j]
         b = x[j+1]
         c = x[j+2]
         tmp1 = quad(eval_stiffD,a,b,args = (a,))
         tmp2 = quad(eval_stiffD,b,c,args = (c,))
         tmp = tmp1[0] + tmp2[0]
         d_main[j] = d_main[j]+(1/h**2)*tmp

     for j in range(0,N-2):
        a = x[j]
        b = x[j+1]
        c = x[j+2]     
        tmp1 = quad(eval_stiffO,a,b, args = (a,b,))
        tmp2 = quad(eval_stiffO,b,c, args = (b,c,) ) 
        d_sub[j] = d_sub[j]+(1/h**2)*tmp1[0]
        d_sup[j] = d_sup[j]+(1/h**2)*tmp2[0]

# list of all the data
     data = [d_sub, d_main, d_sup]   
 # which diagonal each vector goes into
     diags = [-1,0,1]   
 # create the matrix                
     A = sp.diags(data,diags,format='csc') 
     
     return(A)

def make_rhs(x,h,N):

     rhs = np.zeros(N-1)
     
     for j in range(0,N-1):
        a = x[j]
        b = x[j+1]
        c = x[j+2]     
        tmp1 = quad(eval_rhsInt1,a,b,args = (a,))
        tmp2 = quad(eval_rhsInt2,b,c,args = (c,))
        rhs[j] = (1/h)*tmp1[0] + (1/h)*tmp2[0]
     
     return(rhs)