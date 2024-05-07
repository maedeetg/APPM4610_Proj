import numpy as np
import math
import matplotlib.pyplot as plt

def driver():

# boundary data
    a = -1
    b = 1
    alpha = 0
    beta = 0

    plotApproximation(a,b,alpha,beta)
    #plotConvergence(a,b,alpha,beta)
          
    return
     
def eval_pqr(x):
     p = 0*x
     q = 0*x
     r = np.exp(4*x)
     
     return(p,q,r)  



def make_FDmatNeu1st(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr(x)
 
# create the finite difference matrix     
     Matypp = 1/h**2*(np.diag(2*np.ones(N+1)) -np.diag(np.ones(N),-1) - 
           np.diag(np.ones(N),1))
           
     Matyp = 1/(2*h)*(np.diag(np.ones(N),1)-np.diag(np.ones(N),-1))
     
     B = np.matmul(np.diag(p,0),Matyp)
     
     A = Matypp + B+ np.diag(q)

# correct the entries to enforce the boundary condition
     A[0,0] = 1/h
     A[0,1] = -1/h
     A[N,N-1] = 1/h
     A[N,N] = -1/h
    
     A = -A
     
# create the right hand side rhs: (N) in size
     rhs = r
#  update with boundary data   
     rhs[0] = alpha
     rhs[N] = beta
     
     
# solve for the approximate solution

     Ainv = np.linalg.inv(A)
     yapp = np.matmul(Ainv,rhs)
     
     return yapp
     
def make_FDmatNeu2nd(x,h,N,alpha,beta):

# evaluate coefficients of differential equation
     (p,q,r) = eval_pqr(x)

     p = np.append(np.insert(p,0,0), 0)
     q = np.append(np.insert(q,0,0), 0)
     r = np.append(np.insert(r,0,0), 0)
 
# create the finite difference matrix     
     Matypp = 1/h**2*(np.diag(2*np.ones(N+3)) -np.diag(np.ones(N+2),-1) - 
           np.diag(np.ones(N+2),1))
           
     Matyp = 1/(2*h)*(np.diag(np.ones(N+2),1)-np.diag(np.ones(N+2),-1))
     
     B = np.matmul(np.diag(p,0),Matyp)
     
     A = Matypp + B+ np.diag(q)

# correct the entries to enforce the boundary condition
     A[0,0] = 1/(2*h)
     A[0,1] = 0
     A[0,2] = -1/(2*h)
     A[N+2,N] = 1/(2*h)
     A[N+2,N+1] = 0
     A[N+2,N+2] = -1/(2*h)
    
     A = -A
     
# create the right hand side rhs: (N) in size
     rhs =  r
     rhs[0] = alpha
     rhs[N+2] = beta
#  update with boundary datas
     
     
# solve for the approximate solution

     Ainv = np.linalg.inv(A)
     yapp = np.matmul(Ainv,rhs)
     
     return yapp[1:yapp.size - 1]


def plotApproximation(a,b,alpha,beta):
     h = 0.025
     N = int((b-a)/h)
     
     x = np.linspace(a,b,N+1)
     
     yapp = make_FDmatNeu2nd(x,h,N,alpha,beta)


     
#  exact solution 
     y = lambda x: np.sin(math.pi*x)
             
     yex = y(x)
    
     plt.plot(x,yapp,label = 'FD aprox')
     plt.plot(x,yex,label = 'Exact')
     plt.xlabel('x')
     plt.legend(loc = 'upper left')
     plt.show()
     
     err = np.zeros(N+1)
     for j in range(0,N+1):
          err[j] = abs(yapp[j]-yex[j])
          
          
     plt.plot(x,err,label = 'FD aprox')
     plt.xlabel('x')
     plt.ylabel('absolute error')
     plt.legend(loc = 'upper left')
     plt.show()


def plotConvergence(a,b,alpha,beta):

     point = 0.5

     ypoint = np.zeros(10)
     hvals = np.zeros(10)
     
     for i in range(10):

          h = 0.1 * (1/2)**i

          N = int((b-a)/h)

          x = np.linspace(a,b,N+1)
     
          yapp = make_FDmatNeu2nd(x,h,N,alpha,beta)

          idx = int(3 * 2**i)

          
          ypoint[i] = yapp[idx]
          hvals[i] = h

     y = lambda x: np.sin(math.pi*x)
             
     yex = y(point)

     err = abs(ypoint - yex)

     print("hvals:", hvals)
     print("Errors", err)


     plt.loglog(hvals,err)
     plt.xlabel("h value")
     plt.ylabel("Error Magnitude")
     plt.title("Convergence of Neumann Finite Difference")
     plt.show()
     return
     

     
driver()     