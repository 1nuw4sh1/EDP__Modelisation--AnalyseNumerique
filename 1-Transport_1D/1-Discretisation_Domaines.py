import numpy as np 

def Domaine_Spatial(nx, x0 = 0.0, x1 = 1.0):
    x, dx = np.linspace(x0, x1, nx + 1, retstep=True)
    return x, dx

def Domaine_Temporel(dt, t0 = 0.0, tf = 1.0 + 1e-10):
    t = np.arange(t0, tf, dt)
    return t