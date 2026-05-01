import numpy as np

def Upwind(u, param):
    c, dx, dt = param
    du = (u - np.roll(u, 1)) * (c > 0) + (np.roll(u, -1) - u) * (c < 0)
    return du / dx


def Centre(u, param):
    c, dx, dt = param
    du = (np.roll(u, -1) - np.roll(u, 1)) / 2
    return du / dx





# def Rusanov(u, param):
#     return 


# def Eno(u, param):
#     return


# def Weno(u, param):
#     return