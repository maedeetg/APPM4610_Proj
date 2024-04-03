import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la

h1 = lambda x: np.exp(x)*np.sin(5*x)
hp1 = lambda x: np.exp(x)*np.cos(5*x)*5 + np.exp(x)*np.sin(5*x)

h2 = lambda x: abs(x**3)
hp2 = lambda x: 3*x**5/(abs(x**3))

h3 = lambda x: np.exp(-x**(-2))
hp3 = lambda x: 2*np.exp(-x**(-2))/x**3

h4 = lambda x: 1/(1+x**2)
hp4 = lambda x: -2*x/(1+x**2)**2

h5 = lambda x: x**10
hp5 = lambda x: 10*x**9

def driver1():
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
    
    [D_N1, x1] = cheb(N1)
    [D_N2, x2] = cheb(N2)
    [D_N3, x3] = cheb(N3)
    [D_N4, x4] = cheb(N4)
    
    fig, axs = plt.subplots(2, 2)
    title = r"Error in f'(x) = $5e^xcos(5x)+e^xsin(5x)$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D_N1@h1(x1)-hp1(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D_N2@h1(x2)-hp1(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D_N3@h1(x3)-hp1(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D_N4@h1(x4)-hp1(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = r"Error in f'(x) = $\frac{3x^5}{|x^3|}$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D_N1@h2(x1)-hp2(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D_N2@h2(x2)-hp2(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D_N3@h2(x3)-hp2(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D_N4@h2(x4)-hp2(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = r"Error in f'(x) = $\frac{2e^{-x^{-2}}}{x^2}$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D_N1@h3(x1)-hp3(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D_N2@h3(x2)-hp3(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D_N3@h3(x3)-hp3(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D_N4@h3(x4)-hp3(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()

    fig, axs = plt.subplots(2, 2)
    title = r"Error in f'(x) = $-\frac{2x}{(1+x^2)^2}$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D_N1@h4(x1)-hp4(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D_N2@h4(x2)-hp4(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D_N3@h4(x3)-hp4(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D_N4@h4(x4)-hp4(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = r"Error in f'(x) = $10x^9$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D_N1@h5(x1)-hp5(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D_N2@h5(x2)-hp5(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D_N3@h5(x3)-hp5(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D_N4@h5(x4)-hp5(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
        
    plt.show()

def driver2():
    N = np.arange(1, 51)

    error1 = np.zeros(50)
    error2 = np.zeros(50)
    error3 = np.zeros(50)
    error4 = np.zeros(50)
    error5 = np.zeros(50)

    for i in range(len(N)):
        [D_N, x] = cheb(N[i])
        error1[i] = la.norm(D_N@h1(x) - hp1(x), np.inf)
        error2[i] = la.norm(D_N@h2(x) - hp2(x), np.inf)
        error3[i] = la.norm(D_N@h3(x) - hp3(x), np.inf)
        error4[i] = la.norm(D_N@h4(x) - hp4(x), np.inf)
        error5[i] = la.norm(D_N@h5(x) - hp5(x), np.inf)

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title1 = r"f(x) = $e^xsin(5x)$"
    plt.title(title1)
    plt.show()

    plt.semilogy(N, error2, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title2 = r"f(x) = $|x^3|$"
    plt.title(title2)
    plt.show()

    plt.semilogy(N, error3, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title3 = r"f(x) = $e^{-x^{-2}}$"
    plt.title(title3)
    plt.show()

    plt.semilogy(N, error4, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title4 = r"f(x) = $\frac{1}{1+x^2}$"
    plt.title(title4)
    plt.show()

    plt.semilogy(N, error5, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title5 = r"f(x) = $x^{10}$"
    plt.title(title5)
    plt.show()
    
# create D_N matrix and chebyshev nodes

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

driver1()
driver2()