def driver1(): # for u_xx = 4e^2x
    # Boundary data
    a = 0
    b = 1
    alpha = 1
    beta = np.exp(2)
   
    # Exact Solution
    yex = lambda x: np.exp(2*x)
   
    # Chebyshev nodes
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
    axs[1, 1].plot(x4, yex(x4))
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver2(): # for u_xx = 4e^2x
    # Boundary data
    a = 0
    b = 1
    alpha = 1
    beta = 2
   
    # Exact Solution
    yex = lambda x: np.exp(2)*(np.exp(4)-1)**(-1)*(np.exp(2*x)-np.exp(-2*x)) + x
   
    # Chebyshev nodes
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
    axs[1, 1].plot(x4, yex(x4))
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()

def driver3():
    # Boundary data
    a2 = -1
    b2 = 1
    alpha2 = 0
    beta2 = 0
   
    # Exact Solution
   #  yex = lambda x: np.exp(2*x)

    # Chebyshev nodes
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    # For interval [-1, 1]
    [D2_N1, x1] = cheb2(N1)
    [D2_N2, x2] = cheb2(N2)
    [D2_N3, x3] = cheb2(N3)
    [D2_N4, x4] = cheb2(N4)

    p1, q1, r1 = eval_pqr2(x1)
    p2, q2, r2 = eval_pqr2(x2)
    p3, q3, r3 = eval_pqr2(x3)
    p4, q4, r4 = eval_pqr2(x4)
   
    # Approximate solution for interval [-1, 1]
    yapp1 = spectral(p1, q1, r1, N1, a2, b2, alpha2, beta2)
    yapp2 = spectral(p2, q2, r2, N2, a2, b2, alpha2, beta2)
    yapp3 = spectral(p3, q3, r3, N3, a2, b2, alpha2, beta2)
    yapp4 = spectral(p4, q4, r4, N4, a2, b2, alpha2, beta2)
   
    # Plot results
    fig, axs = plt.subplots(2, 2)
    title = "Spectral Collocation Approximation"
    fig.suptitle(title)
    axs[0, 0].plot(x1, yapp1, color = 'green')
    axs[0, 0].set_title("N = 10")
    axs[0, 1].plot(x2, yapp2, color = 'green')
    axs[0, 1].set_title("N = 20")
    axs[1, 0].plot(x3, yapp3, color = 'green')
    axs[1, 0].set_title("N = 50")
    axs[1, 1].plot(x4, yapp4, color = 'green')
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()


def driver4(): # for u_xx = 4e^2x
    # Boundary data
    a = 0
    b = np.pi/4
    alpha = 0
    beta = 0
   
    # Exact Solution
    yex = lambda x: (-1/3)*np.cos(2*x)-(np.sqrt(2)/6)*np.sin(2*x)+(1/3)*np.cos(x)
   
    # Chebyshev nodes
    N1 = 10
    N2 = 20
    N3 = 50
    N4 = 100
   
    [D2_N1, x1] = cheb2_ab(a, b, N1)
    [D2_N2, x2] = cheb2_ab(a, b, N2)
    [D2_N3, x3] = cheb2_ab(a, b, N3)
    [D2_N4, x4] = cheb2_ab(a, b, N4)

    p1, q1, r1 = eval_pqr5(x1)
    p2, q2, r2 = eval_pqr5(x2)
    p3, q3, r3 = eval_pqr5(x3)
    p4, q4, r4 = eval_pqr5(x4)
   
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
    axs[1, 1].plot(x4, yex(x4))
    axs[1, 1].set_title("N = 100")

    for ax in axs.flat:
        ax.set(xlabel = 'x')

    for ax in fig.get_axes():
        ax.label_outer()
       
    plt.show()