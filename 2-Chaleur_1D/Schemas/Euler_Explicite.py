import numpy as np
from time import time


def Explicite(u0, f, params, Nt, History=False):

    alpha, dx, dt = params
    r = alpha * dt / dx**2

    def Laplacien(u):
        return u[2:] - 2*u[1:-1] + u[:-2]
    
    def RHS(u, F):
        return r * Laplacien(u) + dt * F[1:-1]
    
    def step(u, F):
        u_next = u.copy()
        u_next[1:-1] += RHS(u, F)
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

    print(f"Euler explicite: {time() - t0:.4f} s")

    return u