import numpy as np


def Explicite(u, schema_spatial, param):
    c, dx, dt = param
    F = lambda u: -c * schema_spatial(u, param)
    return u + dt * F(u)


def Runge_Kutta_2(u, schema_spatial, param):
    c, dx, dt = param
    F = lambda u: -c * schema_spatial(u, param)
    k1 = dt *F(u)
    k2 = dt * F(u + 0.5 * k1)
    return u + 0.5 * (k1 + k2)


def Runge_Kutta_3(u, schema_spatial, param):
    c, dx, dt = param   
    F = lambda u: -c * schema_spatial(u, param)
    k1 = dt * F(u)
    k2 = dt * F(u + 0.5 * k1)
    k3 = dt * F(u + 2 * k2 - k1)
    return u + (k1 + 4 * k2 + k3) / 6


def Runge_Kutta_4(u, schema_spatial, param):
    c, dx, dt = param
    F = lambda u: -c * schema_spatial(u, param)
    k1 = dt * F(u)
    k2 = dt * F(u + 0.5 * k1)
    k3 = dt * F(u + 0.5 * k2)
    k4 = dt * F(u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6