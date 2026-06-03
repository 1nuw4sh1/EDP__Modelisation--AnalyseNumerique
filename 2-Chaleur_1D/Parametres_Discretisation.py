import numpy as np
import numpy.random as rd



alpha = rd.uniform(0.85, 1.15)


# Domaine Spatial
x_min = 0.0
x_max = 1.0
Nx = 39
val_Nx = np.logspace(np.log10(3), np.log10(300), 15, dtype=int)[::-1]


# Domaine Temporel
t_min = 0.0
t_max = 0.1
Nt = 500
val_Nt = np.logspace(1, 4, 50, dtype=int)[::-1]


# Discretisation
def Discretisation(x, y, delta = None, n = None, Type = None ):

    if delta is not None:
        xi = np.arange(x, 
                       y + delta, 
                       delta)
        return xi
    
    else:
        N = int(n) + 2 if Type == "Espace" else int(n) + 1
        xi, dx = np.linspace(x, 
                             y if Type == "Espace" else y + 1e-10, 
                             N + 1 if Type == "Temps" else N + 2, 
                             retstep=True)
        return xi, dx