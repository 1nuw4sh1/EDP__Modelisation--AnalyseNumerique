import numpy as np
from time import time


def RK2(u0, f, params, Nt, History=False):

    alpha, dx, dt = params
    r = alpha / dx**2       # sans le dt ici

    def Laplacien(u):
        return u[2:] - 2*u[1:-1] + u[:-2]
    
    def RHS(u, F):
        return r * Laplacien(u) + F[1:-1]
    
    def step(u, F):
        u_star = u.copy()
        u_next = u.copy()

        k1 = np.zeros_like(u)
        k2 = np.zeros_like(u)

        k1[1:-1] = dt *  RHS(u, F)
        u_star += k1

        k2[1:-1] = dt * RHS(u_star, F)
        u_next += 0.5 * (k1 + k2)
        return u_next


    
    t0 = time()

    if History :
        u = np.empty((Nt, u0.size), dtype=u0.dtype)
        u[0] = u0
        for n in range(1, Nt):
            u[n] = step(u[n-1], f[n-1])

    else :
        u = u0.copy()
        for n in range(1,Nt):
            u = step(u, f[n-1])

    print(f"Runge-Kutta 2: {time() - t0:.4f} s")

    return u