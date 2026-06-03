import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from time import time


def CN(u0, f, params, Nt, History=False):
    
    alpha, dx, dt = params
    r = alpha * dt / dx**2

    def step(u, F, matrice, methode):
        b = (matrice[0] - 0.5 * r * matrice[1]) @ u
        b[1:-1] += 0.5 * dt * F[1:-1]
        return methode.solve(b)



    Nx = u0.size
    # Matrice Identite
    Id = sp.eye(Nx, format='csr')
    # Matrice Laplacien
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(Nx, Nx), format='csr')
    A[0, :2] = A[-1, -2:] = 0
    # Factorisation LU
    LU = spla.splu((Id + 0.5 * r * A).tocsc())


    t0 = time()

    if History:  
        u = np.empty((Nt, Nx), dtype=u0.dtype)
        u[0] = u0
        for n in range(1, Nt):
            u[n] = step(u[n-1], f[n] + f[n-1], [Id, A], LU)
    
    else : 
        u = u0.copy()
        for n in range(1, Nt):
            u = step(u, f[n] + f[n-1], [Id, A], LU)

    print(f"Crank-Nicolson: {time() - t0:.4f} s")

    return u