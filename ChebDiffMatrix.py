import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la

h1 = lambda x: np.exp(x)*np.sin(5*x)
hp1 = lambda x: 5*np.exp(x)*np.cos(5*x) + np.exp(x)*np.sin(5*x)
hpp1 = lambda x: -24*np.exp(x)*np.sin(5*x) + 10*np.exp(x)*np.cos(5*x)

h2 = lambda x: abs(x**3)
hp2 = lambda x: 3*x*abs(x)
hpp2 = lambda x: 6*abs(x)

h3 = lambda x: np.exp(-x**(-2))
hp3 = lambda x: 2*np.exp(-x**(-2))/x**3
hpp3 = lambda x: (4*np.exp(-x**(-2)) - 6*x**2*np.exp(-x**(-2)))/x**6

h4 = lambda x: 1/(1+x**2)
hp4 = lambda x: (-2*x)/(1+x**2)**2
hpp4 = lambda x: (6*x**2 - 2)/(1+x**2)**3

h5 = lambda x: x**10
hp5 = lambda x: 10*x**9
hpp5 = lambda x: 90*x**8

def cheb(N):
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

    return [D_N, x]

def cheb2(N):
    [D_N, x] = cheb(N)
    D2_N = np.dot(D_N, D_N)
    return [D2_N, x]

def cheb_ab(a, b, N):
    x = np.cos((np.pi * np.arange(N + 1)) / N)
    x = ((b - a)/2)*x+((b+a)/2) # need to transform x-values for general [a, b]
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

    return [D_N, x]

def cent_diff(x, h):
    hp1_app = (h1(x+h)-h1(x-h))/(2*h)
    hp2_app = (h2(x+h)-h2(x-h))/(2*h)
    hp3_app = (h3(x+h)-h3(x-h))/(2*h)
    hp4_app = (h4(x+h)-h4(x-h))/(2*h)
    hp5_app = (h5(x+h)-h5(x-h))/(2*h)
      
    return [hp1_app, hp2_app, hp3_app, hp4_app, hp5_app]

def cent_diff2(x, h):
    hp1_app = (h1(x+h)-2*h1(x) + h1(x-h))/(h**2)
    hp2_app = (h2(x+h)-2*h2(x) + h2(x-h))/(h**2)
    hp3_app = (h3(x+h)-2*h3(x) + h3(x-h))/(h**2)
    hp4_app = (h4(x+h)-2*h4(x) + h4(x-h))/(h**2)
    hp5_app = (h5(x+h)-2*h5(x) + h5(x-h))/(h**2)
      
    return [hp1_app, hp2_app, hp3_app, hp4_app, hp5_app]