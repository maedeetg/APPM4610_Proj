import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, spectral, spectral2
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cheb2_ab, cent_diff, cent_diff2
import scipy

# HOMOGENOUS DIRICHLET BC
def homog_dir_BC():
    a = -1
    b = 1
    alpha = 0
    beta = 0

    y = lambda x: (np.exp(4*x) - x*np.sinh(4) - np.cosh(4))/16

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

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta)

    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t, color = 'green')
    axs[0, 0].plot(x1, y(x1))
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t, color = 'green')
    axs[0, 1].plot(x2, y(x2))
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t, color = 'green')
    axs[1, 0].plot(x3, y(x3))
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t, color = 'green')
    axs[1, 1].plot(x4, y(x4))
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def homog_dir_BC_conv():
    a = -1
    b = 1
    alpha = 0
    beta = 0
    
    N = np.arange(2, 51)

    error1 = np.zeros(49)

    y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16

    for i in range(len(N)):
        [D_N, x] = cheb2(N[i])
        p1, q1, r1 = eval_pqr3(x)
        yapp1t = spectral(p1, q1, r1, N[i], a, b, alpha, beta)
        error1[i] = la.norm(yapp1t - y(x))

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Accuracy of Spectral Collocation BVP"
    plt.title(title1)
    plt.show()

# INHOMOGENOUS DIRICHLET BC
def inhomog_dir_BC1():
    a = -1
    b = 1
    y0 = 0
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    y = lambda x: ((np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16) + lp(x)
   
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

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta) + lp(x1)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta) + lp(x2)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta) + lp(x3)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta) + lp(x4)
   
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
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def inhomog_dir_BC_conv1():
    a = -1
    b = 1
    y0 = 0
    y1 = 1
    alpha = 0
    beta = 0

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    N = np.arange(2, 51)

    error1 = np.zeros(49)

    y = lambda x: (np.exp(4*x) - x*np.sinh(4)-np.cosh(4))/16 + lp(x)

    for i in range(len(N)):
        [D_N, x] = cheb2(N[i])
        p1, q1, r1 = eval_pqr1(x)
        yapp1t = spectral(p1, q1, r1, N[i], a, b, alpha, beta) + lp(x)
        error1[i] = la.norm(yapp1t - y(x))

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Accuracy of Spectral Collocation BVP"
    plt.title(title1)
    plt.show()

def inhomog_dir_BC2(): 
    a = 0
    b = 1
    alpha = 0
    beta = 0
    y0 = 1
    y1 = np.exp(2)
    
    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
   
    y = lambda x: np.exp(2*x) + lp(x)
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)
    
    p1, q1, r1 = eval_pqr1(x1)
    p2, q2, r2 = eval_pqr1(x2)
    p3, q3, r3 = eval_pqr1(x3)
    p4, q4, r4 = eval_pqr1(x4)

    yapp1t = spectral(p1, q1, r1, N1, a, b, alpha, beta) + lp(x1)
    yapp2t = spectral(p2, q2, r2, N2, a, b, alpha, beta) + lp(x2)
    yapp3t = spectral(p3, q3, r3, N3, a, b, alpha, beta) + lp(x3)
    yapp4t = spectral(p4, q4, r4, N4, a, b, alpha, beta) + lp(x4)

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
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def inhomog_dir_BC_conv2():
    a = 0
    b = 1
    alpha = 0
    beta = 0
    y0 = 1
    y1 = np.exp(2)

    lp = lambda x: ((x - a)/(b - a))*y1 + ((x - b)/(a - b))*y0
    
    N = np.arange(2, 51)

    error1 = np.zeros(49)

    y = lambda x: np.exp(2*x) + lp(x)

    for i in range(len(N)):
        [D_N, x] = cheb2_ab(a, b, N[i])
        p1, q1, r1 = eval_pqr1(x)
        yapp1t = spectral(p1, q1, r1, N[i], a, b, alpha, beta) + lp(x)
        error1[i] = la.norm(yapp1t - y(x))

    plt.semilogy(N, error1, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("error")
    title1 = r"Accuracy of Spectral Collocation BVP"
    plt.title(title1)
    plt.show()

    return

def homog_neum_BC():
    a = -1
    b = 1
    alpha = 0
    beta = 0

    y = lambda x: (np.exp(4*x) - 4*np.exp(-4)*(x - 1) - np.exp(4))/16

    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2(N1)
    [D2_N2, x2] = cheb2(N2)
    [D2_N3, x3] = cheb2(N3)
    [D2_N4, x4] = cheb2(N4)
    
    p1, q1, r1 = eval_pqr1(x1)
    p2, q2, r2 = eval_pqr1(x2)
    p3, q3, r3 = eval_pqr1(x3)
    p4, q4, r4 = eval_pqr1(x4)

    yapp1t = spectral2(p1, q1, r1, N1, a, b, alpha, beta)
    yapp2t = spectral2(p2, q2, r2, N2, a, b, alpha, beta)
    yapp3t = spectral2(p3, q3, r3, N3, a, b, alpha, beta)
    yapp4t = spectral2(p4, q4, r4, N4, a, b, alpha, beta)
    plt.plot(x1, yapp1t)
    plt.plot(x1, y(x1), color = 'red')
    plt.show()

    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t, color = 'green')
    axs[0, 0].plot(x1, y(x1))
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t, color = 'green')
    axs[0, 1].plot(x2, y(x2))
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t, color = 'green')
    axs[1, 0].plot(x3, y(x3))
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t, color = 'green')
    axs[1, 1].plot(x4, y(x4))
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
    
    fig, axs = plt.subplots(2, 2)
    title = "Error in Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1t - y(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2t - y(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3t - y(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4t - y(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

# homog_dir_BC()
# homog_dir_BC_conv()
# inhomog_dir_BC()
# inhomog_dir_BC_conv1()
# inhomog_dir_BC2()
# inhomog_dir_BC_conv2()
homog_neum_BC()