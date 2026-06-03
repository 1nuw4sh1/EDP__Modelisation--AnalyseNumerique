import numpy.linalg as la

from Fonction_Test import *
from Parametres_Discretisation import *

# Schemas Explicites
from Schemas.Euler_Explicite import Explicite
from Schemas.Runge_Kutta_2 import RK2
from Schemas.Runge_Kutta_4 import RK4

# Schemas (semi) Implicites
from Schemas.Euler_Implicite import Implicite
from Schemas.Crank_Nicolson import CN

from Graphique import *



def Simulation():

    # Discretisation Spatiale & Temporelle
    x, dx = Discretisation(x_min, x_max, n = Nx, Type = "Espace")
    t, dt = Discretisation(t_min, t_max, n = Nt, Type = "Temps")
    X, T = np.meshgrid(x, t)

    x_fin, _ = Discretisation(x_min, x_max, n = 1e4, Type = "Espace")
    X_fin, T_fin = np.meshgrid(x_fin, t)


    # Verification condition CFL
    if not (alpha * dt / dx**2 <= 0.5) :
        print(f"""
                Attention !! \n
                Condition de stabilité non respectée.
                dt = {dt : .4e} > dt_max = {0.5 * dx**2 / alpha : .4e}
               """)
        return

    # Solution exacte
    u_exacte = Sinusoidal(X_fin, T_fin)

    # Condition Initiale
    u0 = Sinusoidal(x, t[0])
    u_min, u_max = np.min(u0), np.max(u0)

    # Simulation
    History = True
    params = (alpha, dx, dt)
    F = f_Sinusoidal(X, T)

    u_Euler_Explicite = Explicite(u0, F, params, t.size, History)
    u_RK2 = RK2(u0, F, params, t.size, History)
    u_RK4 = RK4(u0, F, params, t.size, History)

    u_Euler_Implicite = Implicite(u0, F, params, t.size, History)
    u_Crank_Nicolson = CN(u0, F, params, t.size, History)


    # Graphique
    Solutions = {
        "Exact" : {"x" : x_fin, "u" : u_exacte, "linestyle": "-", "color": "black", "linewidth": 5, "marker": ""},
        "Euler Explicite" : {"x" : x, "u" : u_Euler_Explicite, "linestyle": "--", "color": "blue", "linewidth": 3, "marker": "o"},
        "Runge-Kutta 2" : {"x" : x, "u" : u_RK2, "linestyle": "--", "color": "red", "linewidth": 3, "marker": "d"},
        "Runge-Kutta 4" : {"x" : x, "u" : u_RK4, "linestyle": "--", "color": "green", "linewidth": 3, "marker": "s"},
        "Euler Implicite" : {"x" : x, "u" : u_Euler_Implicite, "linestyle": "-.", "color": "magenta", "linewidth": 3, "marker": "s"},
        "Crank-Nicolson" : {"x" : x, "u" : u_Crank_Nicolson, "linestyle": ":", "color": "cyan", "linewidth": 3, "marker": "^"},
    }

    Graphique_Simulation(t, Solutions, [u_min, u_max], params)
    







def Erreur():


    # Calculs des pas minimums
    dx_min = Discretisation(x_min, x_max, n = max(val_Nx), Type = "Espace")[1]
    dt_max = Discretisation(t_min, t_max, n = min(val_Nt), Type = "Temps")[1]

    # Preallocations
    val_dx = np.zeros_like(val_Nx, dtype = float)
    val_dt = np.zeros_like(val_Nt, dtype = float)

    Space_Euler_Explicite = np.zeros_like(val_Nx, dtype = float)
    Space_RK2 = np.zeros_like(val_Nx, dtype = float)
    Space_RK4 = np.zeros_like(val_Nx, dtype = float)
    Space_Euler_Implicite = np.zeros_like(val_Nx, dtype = float)
    Space_Crank_Nicolson = np.zeros_like(val_Nx, dtype = float)

    Temporal_Euler_Implicite = np.zeros_like(val_Nt, dtype = float)
    Temporal_Crank_Nicolson = np.zeros_like(val_Nt, dtype = float)


    # Erreur Spatiale
    dt = (0.5 * dx_min**2 / alpha)  * 0.1
    for i, Nx in enumerate(val_Nx):

        print("\n\t" + f"Nx = {Nx}")

        x, dx = Discretisation(x_min, x_max, n = Nx, Type = "Espace")
        t = Discretisation(t_min, t_max, delta = dt, Type = "Temps")
        X, T = np.meshgrid(x, t)

        params = (alpha, dx, dt)
        val_dx[i] = dx

        u_exacte = Sinusoidal(X, T)[-1, :]

        u0 = Sinusoidal(x, t[0])
        F = f_Sinusoidal(X, T)

        u_Euler_Explicite = Explicite(u0, F, params, t.size)
        u_RK2 = RK2(u0, F, params, t.size)
        u_RK4 = RK4(u0, F, params, t.size)
        u_Euler_Implicite = Implicite(u0, F, params, t.size)
        u_Crank_Nicolson = CN(u0, F, params, t.size)

        Space_Euler_Explicite[i] = la.norm(u_Euler_Explicite - u_exacte, np.inf)
        Space_RK2[i] = la.norm(u_RK2 - u_exacte, np.inf)
        Space_RK4[i] = la.norm(u_RK4 - u_exacte, np.inf)
        Space_Euler_Implicite[i] = la.norm(u_Euler_Implicite - u_exacte, np.inf)
        Space_Crank_Nicolson[i] = la.norm(u_Crank_Nicolson - u_exacte, np.inf)
    
    Erreurs_Spatiales = {
        "Euler Explicite" : {"dx" : val_dx, "E" : Space_Euler_Explicite, "linestyle": "--", "color": "blue", "linewidth": 1.5, "marker": "o"},
        "Runge-Kutta 2" : {"dx" : val_dx, "E" : Space_RK2, "linestyle": "--", "color": "red", "linewidth": 1.5, "marker": "d"},
        "Runge-Kutta 4" : {"dx" : val_dx, "E" : Space_RK4, "linestyle": "--", "color": "green", "linewidth": 1.5, "marker": "s"},
        "Euler Implicite" : {"dx" : val_dx, "E" : Space_Euler_Implicite, "linestyle": "-.", "color": "magenta", "linewidth": 1.5, "marker": "s"},
        "Crank-Nicolson" : {"dx" : val_dx, "E" : Space_Crank_Nicolson, "linestyle": ":", "color": "lightblue", "linewidth": 1.5, "marker": "^"},
    }



    # Erreur temporelle (Schemas implicites)
    dx = 1e-4
    x = Discretisation(x_min, x_max, delta = dx, Type = "Espace")
    for i, Nt in enumerate(val_Nt):

        print("\n\t" + f"Nt = {Nt}")

        t, dt = Discretisation(t_min, t_max, n = Nt, Type = "Temps")
        X, T = np.meshgrid(x, t)

        params = (alpha, dx, dt)
        val_dt[i] = dt

        u_exacte = Sinusoidal(x, t[-1])

        u0 = Sinusoidal(x, t[0])
        F = f_Sinusoidal(X, T)
        
        u_Euler_Implicite = Implicite(u0, F, params, t.size)
        u_Crank_Nicolson = CN(u0, F, params, t.size)

        Temporal_Euler_Implicite[i] = la.norm(u_Euler_Implicite - u_exacte, np.inf)
        Temporal_Crank_Nicolson[i] = la.norm(u_Crank_Nicolson - u_exacte, np.inf)

    Erreurs_Temporelles = {
        "Euler Implicite" : {"dt" : val_dt, "E" : Temporal_Euler_Implicite, "linestyle": "-.", "color": "magenta", "linewidth": 1.5, "marker": "s"},
        "Crank-Nicolson" : {"dt" : val_dt, "E" : Temporal_Crank_Nicolson, "linestyle": ":", "color": "lightblue", "linewidth": 1.5, "marker": "^"},
    }
    
    Graphique_Erreur((Erreurs_Spatiales, Erreurs_Temporelles), (val_dx, val_dt))



    
    




if __name__ == "__main__":


    print("\n\t=== Simulation ===\n")
    Simulation()
    print("\n\t=== Fin simulation ===\n\n")


    print("\n\t=== Calcul d'erreur ===\n")
    Erreur()
    print("\n\t=== Fin calcul d'erreur ===\n\n")