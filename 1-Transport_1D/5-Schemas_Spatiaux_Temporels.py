import numpy as np


def Lax_Friedrichs(u, param):
    c, dx, dt = param
    u1 = 0.5 * (np.roll(u, -1) + np.roll(u, 1))
    u2 = 0.5 * c * dt / dx * (np.roll(u, -1) - np.roll(u, 1))
    return u1 - u2


def Lax_Wendroff(u, param):
    c, dx, dt = param
    u1 = 0.5 * c * dt / dx * (np.roll(u, -1) - np.roll(u, 1))
    u2 = 0.5 * (c * dt / dx)**2 * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
    return u - u1 + u2