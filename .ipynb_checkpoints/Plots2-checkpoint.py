import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import SolveBVP, ChebDiffMatrix
from SolveBVP import eval_pqr1, eval_pqr2, eval_pqr3, spectral
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cent_diff, cent_diff2
import scipy

def driver1(): # for u_xx = e^(4x)
    # Boundary data
    a1 = -1
    b1 = 1
    alpha1 = 0
    beta1 = 0
   
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
   
    # Approximate solution
    x1t, yapp1t = spectral(p1, q1, r1, x1, N1, alpha1, beta1)
    x2t, yapp2t = spectral(p2, q2, r2, x2, N2, alpha1, beta1)
    x3t, yapp3t = spectral(p3, q3, r3, x3, N3, alpha1, beta1)
    x4t, yapp4t = spectral(p4, q4, r4, x4, N4, alpha1, beta1)
   
    # Plot results
    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1t, yapp1t, color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2t, yapp2t, color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3t, yapp3t, color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4t, yapp4t, color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver2(): # for u_xx = 4e^2x
    # Boundary data
    a1 = 0
    b1 = 1
    alpha1 = 1
    beta1 = np.exp(2)
   
    # Exact Solution
    yex = lambda x: np.exp(2*x)
   
    # Chebyshev nodes
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
   
    # Approximate solution
    x1t, yapp1t = spectral(p1, q1, r1, x1, N1, alpha1, beta1)
    x2t, yapp2t = spectral(p2, q2, r2, x2, N2, alpha1, beta1)
    x3t, yapp3t = spectral(p3, q3, r3, x3, N3, alpha1, beta1)
    x4t, yapp4t = spectral(p4, q4, r4, x4, N4, alpha1, beta1)
   
    # Plot results
    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1t, yapp1t, color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2t, yapp2t, color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3t, yapp3t, color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4t, yapp4t, color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
    
# def driver3():
#     # Boundary data
   
#     a2 = 0
#     b2 = 1
#     alpha2 = 1
#     beta2 = np.exe(2)

#     p, q, r = eval_pqr1(x)
   
#     # Exact Solution
#     yex = lambda x: np.exp(2*x)

#     # Chebyshev nodes
#     N1 = 10
#     N2 = 20
#     N3 = 50
#     N4 = 100
   
   
#     # For interval [-1, 1]
#     [D2_N1, x1] = cheb2(N1)
#     [D2_N2, x2] = cheb2(N2)
#     [D2_N3, x3] = cheb2(N3)
#     [D2_N4, x4] = cheb2(N4)
   
#     # Approximate solution for interval [-1, 1]
#     x1, yapp1 = spectral(p, q, r, x1, N1, alpha2, beta2)
#     x2, yapp2 = spectral(p, q, r, x2, N2, alpha2, beta2)
#     x3, yapp3 = spectral(p, q, r, x3, N3, alpha2, beta2)
#     x4, yapp4 = spectral(p, q, r, x4, N4, alpha2, beta2)
   
   
#     # Exact solution
#     #yex = exact_solution(x)
   
#     # Plot results
   
#     fig, axs = plt.subplots(2, 2)
#     title = "Spectral Collocation Approximation"
#     fig.suptitle(title)
#     axs[0, 0].plot(x1, yapp1, color = 'green')
#     axs[0, 0].set_title("N = 10")
#     axs[0, 1].plot(x2, yapp2, color = 'green')
#     axs[0, 1].set_title("N = 20")
#     axs[1, 0].plot(x3, yapp3, color = 'green')
#     axs[1, 0].set_title("N = 50")
#     axs[1, 1].plot(x4, yapp4, color = 'green')
#     axs[1, 1].set_title("N = 100")

#     for ax in axs.flat:
#         ax.set(xlabel = 'x', ylabel = 'error')

#     for ax in fig.get_axes():
#         ax.label_outer()
       
#     plt.show()

driver1()
driver2()