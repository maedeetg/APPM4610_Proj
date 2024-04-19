import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, eval_pqr2, eval_pqr3, eval_pqr4, eval_pqr5, spectral
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2
import scipy

def driver1(): # for u_xx = e^(4x)
    # Boundary data
    a = -1
    b = 1
    alpha = 0
    beta = 0
   
    # Exact Solution
    yex = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16
   
    # Chebyshev nodes
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2(N1)
    [D2_N2, x2] = cheb2(N2)
    [D2_N3, x3] = cheb2(N3)
    [D2_N4, x4] = cheb2(N4)
    
    p1, q1, r1 = eval_pqr3(x1)
    p2, q2, r2 = eval_pqr3(x2)
    p3, q3, r3 = eval_pqr3(x3)
    p4, q4, r4 = eval_pqr3(x4)

    # plt.plot(x4, yex(x4), color = 'green')
    # plt.xlabel('x')
    # plt.title("Actual Derivative")
    # plt.show()
   
    # Approximate solution
    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta)
   
    # Plot results
    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t, color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t, color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t, color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t, color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - yex(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - yex(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - yex(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - yex(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver2():
    a = -1
    b = 1
    alpha = 0
    beta = 0
    
    N = np.arange(2, 51)

    error1 = np.zeros(49)

    yex = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16

    for i in range(len(N)):
        [D_N, x] = cheb2(N[i])
        p1, q1, r1 = eval_pqr3(x)
        yapp1t = spectral(p1, q1, r1, N[i], a, b, alpha, beta)
        error1[i] = la.norm(yapp1t - yex(x))

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Accuracy of Spectral Collocation BVP"
    plt.title(title1)
    plt.show()

def driver3(): # for u_xx = e^(4x)
    # Boundary data
    a = -1
    b = 1
    alpha = 0
    beta = 1
   
    # Exact Solution
    yex = lambda x: ((np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16) + ((x+1)/2)
   
    # Chebyshev nodes
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2(N1)
    [D2_N2, x2] = cheb2(N2)
    [D2_N3, x3] = cheb2(N3)
    [D2_N4, x4] = cheb2(N4)
    
    p1, q1, r1 = eval_pqr3(x1)
    p2, q2, r2 = eval_pqr3(x2)
    p3, q3, r3 = eval_pqr3(x3)
    p4, q4, r4 = eval_pqr3(x4)

    # plt.plot(x4, yex(x4), color = 'green')
    # plt.xlabel('x')
    # plt.title("Actual Derivative")
    # plt.show()

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta)
   
    # Plot results
    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t, color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t, color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t, color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t, color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - yex(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - yex(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - yex(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - yex(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver4():
    a = -1
    b = 1
    alpha = 0
    beta = 1
    
    N = np.arange(2, 51)

    error1 = np.zeros(49)

    yex = lambda x: ((np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16) + ((x+1)/2)

    for i in range(len(N)):
        [D_N, x] = cheb2(N[i])
        p1, q1, r1 = eval_pqr3(x)
        yapp1t = spectral(p1, q1, r1, N[i], a, b, alpha, beta)
        error1[i] = la.norm(yapp1t - yex(x), np.inf)

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Accuracy of Spectral Collocation BVP"
    plt.title(title1)
    plt.show()


# driver1()
# driver2()
driver3()
driver4()