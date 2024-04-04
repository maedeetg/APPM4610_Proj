import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import ChebDiffMatrix
from ChebDiffMatrix import cheb, cheb2, cheb_ab, cent_diff

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

# plots of approximation of first derivative
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
    title = r"Error in $\frac{d}{dx}f(x) = e^xsin(5x)$"
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
    title = r"Error in $\frac{d}{dx}f(x) = |x^3|$"
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
    title = r"Error in $\frac{d}{dx}f(x) = e^{-x^{-2}}$"
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
    title = r"Error in $\frac{d}{dx}f(x) = \frac{1}{1+x^2}$"
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
    title = r"Error in $\frac{d}{dx}f(x) = x^{10}$"
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

# plots of convergence of D_N
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
    title1 = r"Accuracy of $\frac{d}{dx}f(x) = e^xsin(5x)$"
    plt.title(title1)
    plt.show()

    plt.semilogy(N, error2, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title2 = r"Accuracy of $\frac{d}{dx}f(x) = |x^3|$"
    plt.title(title2)
    plt.show()

    plt.semilogy(N, error3, '-go')
    plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title3 = r"Accuracy of $\frac{d}{dx}f(x) = e^{-x^{-2}}$"
    plt.title(title3)
    plt.show()

    plt.semilogy(N, error4, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title4 = r"Accuracy of $\frac{d}{dx}f(x) = \frac{1}{1+x^2}$"
    plt.title(title4)
    plt.show()

    plt.semilogy(N, error5, '-go')
    plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title5 = r"Accuracy of $\frac{d}{dx}f(x) = x^{10}$"
    plt.title(title5)
    plt.show()

# plots of second deriv approx
def driver3():
   
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2(N1)
    [D2_N2, x2] = cheb2(N2)
    [D2_N3, x3] = cheb2(N3)
    [D2_N4, x4] = cheb2(N4)
   
    # Second Derivative Approx e^x*sin(5x)
    fig, axs = plt.subplots(2, 2)
    title = r"Error in $\frac{d^2}{dx^2}f(x) = e^xsin(5x)$"
    fig.suptitle(title)
   
    axs[0, 0].plot(x1, D2_N1@h1(x1)-hp1(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D2_N2@h1(x2)-hp1(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D2_N3@h1(x3)-hp1(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D2_N4@h1(x4)-hp1(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
   
    # Second Derivative Approx |x^3|
    fig, axs = plt.subplots(2, 2)
    title = r"Error in $\frac{d^2}{dx^2}f(x) = |x^3|$"
    fig.suptitle(title)
   
    axs[0, 0].plot(x1, D2_N1@h2(x1)-hp2(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D2_N2@h2(x2)-hp2(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D2_N3@h2(x3)-hp2(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D2_N4@h2(x4)-hp2(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
   
    # Second Derivative Approx e^-(x^(-2))
    fig, axs = plt.subplots(2, 2)
    title = r"Error in $\frac{d^2}{dx^2}f(x) = e^{-x^{-2}}$"
    fig.suptitle(title)
   
    axs[0, 0].plot(x1, D2_N1@h3(x1)-hp3(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D2_N2@h3(x2)-hp3(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D2_N3@h3(x3)-hp3(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D2_N4@h3(x4)-hp3(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
   
    # Second Derivative Approx 1/(1+x^2)
    fig, axs = plt.subplots(2, 2)
    title = r"Error in $\frac{d^2}{dx^2}f(x) = \frac{1}{1+x^2}$"
    fig.suptitle(title)
   
    axs[0, 0].plot(x1, D2_N1@h4(x1)-hp4(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D2_N2@h4(x2)-hp4(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D2_N3@h4(x3)-hp4(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D2_N4@h4(x4)-hp4(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()
   
    # Second Derivative approx x^10
    fig, axs = plt.subplots(2, 2)
    title = r"Error in $\frac{d^2}{dx^2}f(x) = x^{10}$"
    fig.suptitle(title)
    axs[0, 0].plot(x1, D2_N1@h5(x1)-hp5(x1), color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, D2_N2@h5(x2)-hp5(x2), color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, D2_N3@h5(x3)-hp5(x3), color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, D2_N4@h5(x4)-hp5(x4), color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x', ylabel = 'error')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

# convergence plots for second deriv
def driver4():
   
    N = np.arange(1, 51)

    error1 = np.zeros(50)
    error2 = np.zeros(50)
    error3 = np.zeros(50)
    error4 = np.zeros(50)
    error5 = np.zeros(50)

    for i in range(len(N)):
        [D2_N, x] = cheb2(N[i])
        error1[i] = la.norm(D2_N@h1(x) - hp1(x), np.inf)
        error2[i] = la.norm(D2_N@h2(x) - hp2(x), np.inf)
        error3[i] = la.norm(D2_N@h3(x) - hp3(x), np.inf)
        error4[i] = la.norm(D2_N@h4(x) - hp4(x), np.inf)
        error5[i] = la.norm(D2_N@h5(x) - hp5(x), np.inf)

    plt.semilogy(N, error1, '-go')
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title1 = r"Accuracy of $\frac{d^2}{dx^2}f(x) = e^xsin(5x)$"
    plt.title(title1)
    plt.show()

    plt.semilogy(N, error2, '-go')
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title2 = r"Accuracy of $\frac{d^2}{dx^2}f(x) = |x^3|$"
    plt.title(title2)
    plt.show()

    plt.semilogy(N, error3, '-go')
    #plt.ylim(10**(-17), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title3 = r"Accuracy of $\frac{d^2}{dx^2}f(x) = e^{-x^{-2}}$"
    plt.title(title3)
    plt.show()

    plt.semilogy(N, error4, '-go')
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title4 = r"Accuracy of $\frac{d^2}{dx^2}f(x) = \frac{1}{1+x^2}$"
    plt.title(title4)
    plt.show()

    plt.semilogy(N, error5, '-go')
    #plt.ylim(10**(-15), 10**(2))
    plt.xlabel("N")
    plt.ylabel("max error")
    title5 = r"Accuracy of $\frac{d^2}{dx^2}f(x) = x^{10}$"
    plt.title(title5)
    plt.show()

# plots comparing spectral to centered difference
def driver6():
    N = 50
    a = -1
    b = 1
    h = (b - a)/N

    [D_N, x] = cheb(N)
    [hp1_app, hp2_app, hp3_app, hp4_app, hp5_app] = cent_diff(x, h)

    plt.plot(x, D_N@h1(x) - hp1(x), label = 'spectral', color = 'green')
    plt.plot(x, hp1_app - hp1(x), label = 'centered', color = 'blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title(r"Error in Spectral and Centered Difference Approximations")
    plt.show()

    plt.plot(x, D_N@h2(x) - hp2(x), label = 'spectral', color = 'green')
    plt.plot(x, hp2_app - hp2(x), label = 'centered', color = 'blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title(r"Error in Spectral and Centered Difference Approximations")
    plt.show()

    plt.plot(x, D_N@h3(x) - hp3(x), label = 'spectral', color = 'green')
    plt.plot(x, hp3_app - hp3(x), label = 'centered', color = 'blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title(r"Error in Spectral and Centered Difference Approximations")
    plt.show()

    plt.plot(x, D_N@h4(x) - hp4(x), label = 'spectral', color = 'green')
    plt.plot(x, hp4_app - hp4(x), label = 'centered', color = 'blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title(r"Error in Spectral and Centered Difference Approximations")
    plt.show()

    plt.plot(x, D_N@h5(x) - hp5(x), label = 'spectral', color = 'green')
    plt.plot(x, hp5_app - hp5(x), label = 'centered', color = 'blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("error")
    plt.title(r"Error in Spectral and Centered Difference Approximations")
    plt.show()

# driver1()
# driver2()
# driver3()
# driver4()
# driver5()