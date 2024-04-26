import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver1():
    hp2 = lambda x: 3*x**5/(abs(x**3))

    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
    ''' interval'''
    a = -1
    b = 1
   
    ''' create equispaced interpolation nodes'''
    xint1 = np.cos((np.pi * np.arange(N1 + 1)) / N1)
    xint2 = np.cos((np.pi * np.arange(N2 + 1)) / N2)
    xint3 = np.cos((np.pi * np.arange(N3 + 1)) / N3)
    xint4 = np.cos((np.pi * np.arange(N4 + 1)) / N4)

    ''' create interpolation data'''
    yint1 = hp2(xint1)
    yint2 = hp2(xint2)
    yint3 = hp2(xint3)
    yint4 = hp2(xint4)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l1= np.zeros(Neval+1)
    yeval_l2= np.zeros(Neval+1)
    yeval_l3= np.zeros(Neval+1)
    yeval_l4= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y1 = np.zeros( (N1+1, N1+1) )
    y2 = np.zeros( (N2+1, N2+1) )
    y3 = np.zeros( (N3+1, N3+1) )
    y4 = np.zeros( (N4+1, N4+1) )
     
    for j in range(N1+1):
       y1[j][0]  = yint1[j]

    for j in range(N2+1):
       y2[j][0]  = yint2[j]

    for j in range(N3+1):
       y3[j][0]  = yint3[j]

    for j in range(N4+1):
       y4[j][0]  = yint4[j]

    y1 = dividedDiffTable(xint1, y1, N1+1)
    y2 = dividedDiffTable(xint2, y2, N2+1)
    y3 = dividedDiffTable(xint3, y3, N3+1)
    y4 = dividedDiffTable(xint4, y4, N4+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l1[kk] = eval_lagrange(xeval[kk],xint1,yint1,N1)
        yeval_l2[kk] = eval_lagrange(xeval[kk],xint2,yint2,N2)
        yeval_l3[kk] = eval_lagrange(xeval[kk],xint3,yint3,N3)
        yeval_l4[kk] = eval_lagrange(xeval[kk],xint4,yint4,N4)
        #yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex1 = hp2(xeval)
    fex2 = hp2(xeval)
    fex3 = hp2(xeval)
    fex4 = hp2(xeval)

    fig, axs = plt.subplots(2, 2)
    title = r"Lagrange Interpolation with Chebyshev Nodes for $\frac{d}{dx}|x^3|$"
    fig.suptitle(title)
    axs[0, 0].plot(xeval, (yeval_l1 - fex1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(xeval, (yeval_l2 - fex2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(xeval, (yeval_l3 - fex3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(xeval, (yeval_l4 - fex4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
        
def driver2():
    hp2 = lambda x: 2*np.exp(-x**(-2))/x**3
    
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
    
    ''' interval'''
    a = -1
    b = 1
   
    ''' create equispaced interpolation nodes'''
    xint1 = np.cos((np.pi * np.arange(N1 + 1)) / N1)
    xint2 = np.cos((np.pi * np.arange(N2 + 1)) / N2)
    xint3 = np.cos((np.pi * np.arange(N3 + 1)) / N3)
    xint4 = np.cos((np.pi * np.arange(N4 + 1)) / N4)

    ''' create interpolation data'''
    yint1 = hp2(xint1)
    yint2 = hp2(xint2)
    yint3 = hp2(xint3)
    yint4 = hp2(xint4)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l1= np.zeros(Neval+1)
    yeval_l2= np.zeros(Neval+1)
    yeval_l3= np.zeros(Neval+1)
    yeval_l4= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y1 = np.zeros( (N1+1, N1+1) )
    y2 = np.zeros( (N2+1, N2+1) )
    y3 = np.zeros( (N3+1, N3+1) )
    y4 = np.zeros( (N4+1, N4+1) )
     
    for j in range(N1+1):
       y1[j][0]  = yint1[j]

    for j in range(N2+1):
       y2[j][0]  = yint2[j]

    for j in range(N3+1):
       y3[j][0]  = yint3[j]

    for j in range(N4+1):
       y4[j][0]  = yint4[j]

    y1 = dividedDiffTable(xint1, y1, N1+1)
    y2 = dividedDiffTable(xint2, y2, N2+1)
    y3 = dividedDiffTable(xint3, y3, N3+1)
    y4 = dividedDiffTable(xint4, y4, N4+1)
    
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l1[kk] = eval_lagrange(xeval[kk],xint1,yint1,N1)
        yeval_l2[kk] = eval_lagrange(xeval[kk],xint2,yint2,N2)
        yeval_l3[kk] = eval_lagrange(xeval[kk],xint3,yint3,N3)
        yeval_l4[kk] = eval_lagrange(xeval[kk],xint4,yint4,N4)
        #yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex1 = hp2(xeval)
    fex2 = hp2(xeval)
    fex3 = hp2(xeval)
    fex4 = hp2(xeval)

    fig, axs = plt.subplots(2, 2)
    title = r"Lagrange Interpolation with Chebyshev Nodes for $\frac{d}{dx}e^{-x^{-2}}$"
    fig.suptitle(title)
    axs[0, 0].plot(xeval, (yeval_l1 - fex1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(xeval, (yeval_l2 - fex2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(xeval, (yeval_l3 - fex3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(xeval, (yeval_l4 - fex4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver3():
    hp2 = lambda x: 6*abs(x)
    
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
    
    ''' interval'''
    a = -1
    b = 1
   
    ''' create equispaced interpolation nodes'''
    xint1 = np.cos((np.pi * np.arange(N1 + 1)) / N1)
    xint2 = np.cos((np.pi * np.arange(N2 + 1)) / N2)
    xint3 = np.cos((np.pi * np.arange(N3 + 1)) / N3)
    xint4 = np.cos((np.pi * np.arange(N4 + 1)) / N4)

    ''' create interpolation data'''
    yint1 = hp2(xint1)
    yint2 = hp2(xint2)
    yint3 = hp2(xint3)
    yint4 = hp2(xint4)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l1= np.zeros(Neval+1)
    yeval_l2= np.zeros(Neval+1)
    yeval_l3= np.zeros(Neval+1)
    yeval_l4= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y1 = np.zeros( (N1+1, N1+1) )
    y2 = np.zeros( (N2+1, N2+1) )
    y3 = np.zeros( (N3+1, N3+1) )
    y4 = np.zeros( (N4+1, N4+1) )
     
    for j in range(N1+1):
       y1[j][0]  = yint1[j]

    for j in range(N2+1):
       y2[j][0]  = yint2[j]

    for j in range(N3+1):
       y3[j][0]  = yint3[j]

    for j in range(N4+1):
       y4[j][0]  = yint4[j]

    y1 = dividedDiffTable(xint1, y1, N1+1)
    y2 = dividedDiffTable(xint2, y2, N2+1)
    y3 = dividedDiffTable(xint3, y3, N3+1)
    y4 = dividedDiffTable(xint4, y4, N4+1)
    
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l1[kk] = eval_lagrange(xeval[kk],xint1,yint1,N1)
        yeval_l2[kk] = eval_lagrange(xeval[kk],xint2,yint2,N2)
        yeval_l3[kk] = eval_lagrange(xeval[kk],xint3,yint3,N3)
        yeval_l4[kk] = eval_lagrange(xeval[kk],xint4,yint4,N4)
        #yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex1 = hp2(xeval)
    fex2 = hp2(xeval)
    fex3 = hp2(xeval)
    fex4 = hp2(xeval)

    fig, axs = plt.subplots(2, 2)
    title = r"Lagrange Interpolation with Chebyshev Nodes for $\frac{d^2}{dx^2}|x^3|$"
    fig.suptitle(title)
    axs[0, 0].plot(xeval, (yeval_l1 - fex1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(xeval, (yeval_l2 - fex2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(xeval, (yeval_l3 - fex3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(xeval, (yeval_l4 - fex4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver4():
    hp2 = lambda x: (4*np.exp(-x**(-2)) - 6*x**2*np.exp(-x**(-2)))/x**6
    
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
    ''' interval'''
    a = -1
    b = 1
   
    ''' create equispaced interpolation nodes'''
    xint1 = np.cos((np.pi * np.arange(N1 + 1)) / N1)
    xint2 = np.cos((np.pi * np.arange(N2 + 1)) / N2)
    xint3 = np.cos((np.pi * np.arange(N3 + 1)) / N3)
    xint4 = np.cos((np.pi * np.arange(N4 + 1)) / N4)

    ''' create interpolation data'''
    yint1 = hp2(xint1)
    yint2 = hp2(xint2)
    yint3 = hp2(xint3)
    yint4 = hp2(xint4)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l1= np.zeros(Neval+1)
    yeval_l2= np.zeros(Neval+1)
    yeval_l3= np.zeros(Neval+1)
    yeval_l4= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y1 = np.zeros( (N1+1, N1+1) )
    y2 = np.zeros( (N2+1, N2+1) )
    y3 = np.zeros( (N3+1, N3+1) )
    y4 = np.zeros( (N4+1, N4+1) )
     
    for j in range(N1+1):
       y1[j][0]  = yint1[j]

    for j in range(N2+1):
       y2[j][0]  = yint2[j]

    for j in range(N3+1):
       y3[j][0]  = yint3[j]

    for j in range(N4+1):
       y4[j][0]  = yint4[j]

    y1 = dividedDiffTable(xint1, y1, N1+1)
    y2 = dividedDiffTable(xint2, y2, N2+1)
    y3 = dividedDiffTable(xint3, y3, N3+1)
    y4 = dividedDiffTable(xint4, y4, N4+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
        yeval_l1[kk] = eval_lagrange(xeval[kk],xint1,yint1,N1)
        yeval_l2[kk] = eval_lagrange(xeval[kk],xint2,yint2,N2)
        yeval_l3[kk] = eval_lagrange(xeval[kk],xint3,yint3,N3)
        yeval_l4[kk] = eval_lagrange(xeval[kk],xint4,yint4,N4)
        #yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex1 = hp2(xeval)
    fex2 = hp2(xeval)
    fex3 = hp2(xeval)
    fex4 = hp2(xeval)

    fig, axs = plt.subplots(2, 2)
    title = r"Lagrange Interpolation with Chebyshev Nodes for $\frac{d^2}{dx^2}e^{-x^{-2}}$"
    fig.suptitle(title)
    axs[0, 0].plot(xeval, (yeval_l1 - fex1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(xeval, (yeval_l2 - fex2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(xeval, (yeval_l3 - fex3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(xeval, (yeval_l4 - fex4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  

''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint, hp2, N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + hp2[0][j]*ptmp[j]  

    return yeval

# driver1()        
driver2()
# driver3()
# driver4()