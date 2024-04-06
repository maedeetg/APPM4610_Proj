import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la1
import scipy.sparse as sp
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la2

def driver():

# this demo code considers the boundary value problem
# y'' = p(x)y'+q(x)y + r(x)
#y(a) = alpha, y(b) = beta

# boundary data
     a = 0
     b = 1
     alpha = 1
     beta = np.exp(2)

     # step size
     # h = np.array([10**(-2), 10**(-3), 10**(-4), 10**(-5)])
     h1 = 10**(-2)
     h2 = 10**(-3)
     h3 = 10**(-4)
     h4 = 10**(-5)
     # N = np.zeros(len(h))
     # for i in range(len(h)):
     #     N[i] = int((b-a)/h[i])
     N1 =  int((b-a)/h1)
     N2 =  int((b-a)/h2)
     N3 =  int((b-a)/h3)
     N4 =  int((b-a)/h4)
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)
     
     yapp1 = make_FDmatDir(x1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir(x2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir(x3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir(x4,h4,N4,alpha,beta)
          
#  exact solution 
     # c2 = 1/70*(8-12*np.sin(np.log(2))-4*np.cos(np.log(2)))
     # c1 = 11/10-c2
     y = lambda x: np.exp(2*x)
             
     yex1 = y(x1)
     yex2 = y(x2)
     yex3 = y(x3)
     yex4 = y(x4)
    
     # plt.plot(x1,yapp1,label = 'h = 10^(-2)')
     # plt.plot(x1,yex1,label = 'Exact')
    
     # plt.plot(x2,yapp2,label = 'h = 10^(-3)')
     # plt.plot(x2,yex2,label = 'Exact')
    
     # plt.plot(x3,yapp3,label = 'h = 10^(-4)')
     # plt.plot(x3,yex3,label = 'Exact')
    
     # plt.plot(x4,yapp4,label = 'h = 10^(-5)')
     # plt.plot(x4,yex4,label = 'Exact')
     # plt.xlabel('x')
     # plt.legend(loc = 'upper left')
    
     # plt.show()
     
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
          
     print('err1 = ', err1)
     print('err2 = ', err2)
     print('err3 = ', err3)
     print('err4 = ', err4)
          
     plt.semilogy(x1,err1,label = 'h = 10^(-2)')
     plt.semilogy(x2,err2,label = 'h = 10^(-3)')
     plt.semilogy(x3,err3,label = 'h = 10^(-4)')
     plt.semilogy(x4,err4,label = 'h = 10^(-5)')
     plt.xlabel('x')
     plt.ylabel('absolute error')
     plt.legend(loc = 'upper left')
     plt.show()

     err_norm1 = la1.norm(err1)
     err_norm2 = la1.norm(err2)
     err_norm3 = la1.norm(err3)
     err_norm4 = la1.norm(err4)

     h = [h1, h2, h3, h4]
     err_norms = [err_norm1, err_norm2, err_norm3, err_norm4]

     plt.semilogy(h, err_norms)
     plt.xlabel("h")
     plt.ylabel("Log of norm of absolute error")
     plt.title("Problem 1 Plot")
     plt.show()
          
     return

def driver2():

# this demo code considers the boundary value problem
# y'' = p(x)y'+q(x)y + r(x)
#y(a) = alpha, y(b) = beta

# boundary data
     a = -1
     b = 1
     alpha = 0
     beta = 0

     # step size
     N1 = 10
     N2 = 20
     N3 = 50
     N4 = 100
    
     h1 =  int((b-a)/N1)
     h2 =  int((b-a)/N2)
     h3 =  int((b-a)/N3)
     h4 =  int((b-a)/N4)
     
     x1 = np.linspace(a,b,N1+1)
     x2 = np.linspace(a,b,N2+1)
     x3 = np.linspace(a,b,N3+1)
     x4 = np.linspace(a,b,N4+1)
     
     yapp1 = make_FDmatDir(x1,h1,N1,alpha,beta)
     yapp2 = make_FDmatDir(x2,h2,N2,alpha,beta)
     yapp3 = make_FDmatDir(x3,h3,N3,alpha,beta)
     yapp4 = make_FDmatDir(x4,h4,N4,alpha,beta)
          
#  exact solution 
     # c2 = 1/70*(8-12*np.sin(np.log(2))-4*np.cos(np.log(2)))
     # c1 = 11/10-c2
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
          err1[j] = (yapp1[j]-yex1[j])
         
     for j in range(0,N2+1):
          err2[j] = (yapp2[j]-yex2[j])
         
     for j in range(0,N3+1):
          err3[j] = (yapp3[j]-yex3[j])
         
     for j in range(0,N4+1):
          err4[j] = (yapp4[j]-yex4[j])
          
     print('err1 = ', err1)
     print('err2 = ', err2)
     print('err3 = ', err3)
     print('err4 = ', err4)

     err_norm1 = la1.norm(err1)
     err_norm2 = la1.norm(err2)
     err_norm3 = la1.norm(err3)
     err_norm4 = la1.norm(err4)

     N = [N1, N2, N3, N4]
     err_norms = [err_norm1, err_norm2, err_norm3, err_norm4]

     plt.semilogy(N, err_norms, color = 'green')
     plt.xlabel("N")
     plt.ylabel("error")
     plt.title("Finite Difference Approxmiation BVP")
     plt.show()

     fig, axs = plt.subplots(2, 2)
     title = r"Error in Finite Difference Approximation"
     fig.suptitle(title)
     axs[0, 0].plot(x1, yapp1 - yex1, color = 'green')
     axs[0, 0].set_title("N = 10")
     axs[0, 1].plot(x2, -err2, color = 'green')
     axs[0, 1].set_title("N = 20")
     axs[1, 0].plot(x3, -err3, color = 'green')
     axs[1, 0].set_title("N = 50")
     axs[1, 1].plot(x4, -err4, color = 'green')
    # print("deriv approx", D_N4@h3(x4), "actual deriv", hp3(x4), 'func', h3(x4))
     axs[1, 1].set_title("N = 100")

     for ax in axs.flat:
         ax.set(xlabel = 'x', ylabel = 'error')

     for ax in fig.get_axes():
         ax.label_outer()
        
     plt.show()
          
     return
     
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

def make_FDmatDir(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr2(x)
     
 
# create the finite difference matrix     
     Matypp = (N/2)**2*(sp.diags(2*np.ones(N-1)) -sp.diags(np.ones(N-2),-1) - 
           sp.diags(np.ones(N-2),1))
     #print(Matypp)
           
     Matyp = (N/4)*(sp.diags(np.ones(N-2),1)-sp.diags(np.ones(N-2),-1))
     #print(Matypp)

     
     A = Matypp +sp.diags(p[1:N],0)@Matyp + sp.diags(q[1:N])
     # eig_vals_vect= la.eigs(A)
     # eig_vals = eig_vals_vect[0]
     # cond = eig_vals[-1] / eig_vals[0]
     # print(cond)
     # cond = max(eig_vals) / min(eig_vals)
     # print(cond)

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
     
driver2()     
