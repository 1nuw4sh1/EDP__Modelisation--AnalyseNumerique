import numpy as np 
import numpy.linalg as la


def Solveur(u0, t, schema_temporel, schema_spatial, param):
    U = [u0.copy()]
    c, dx, dt = param

    for _ in t[1:]:
        U.append(schema_temporel(U[-1], schema_spatial, param))

    return U


def Solveur_bis(u0, t, schema_spatio_temporel, param):
    U = [u0.copy()]
    c, dx, dt = param

    for _ in t[1:]:
        U.append(schema_spatio_temporel(U[-1], param))

    return U


def Erreur(u_num, u_ex):
    L2 = la.norm(u_num - u_ex, ord=2)
    Linf = la.norm(u_num - u_ex, ord=np.inf)
    return (L2, Linf)