from Discretisation_Domaines import Domaine_Spatial, Domaine_Temporel

from Solution_Exacte import Condition_Initiale, Solution_Exacte

from Graphe import Creation_Graphe, Creation_Graphe_Erreur
from Solveur import Solveur, Solveur_bis, Erreur

from Schemas_Spatiaux import Upwind, Centre
from Schemas_Temporels import Explicite, Runge_Kutta_2, Runge_Kutta_3, Runge_Kutta_4
from Schemas_Spatiaux_Temporels import Lax_Wendroff, Lax_Friedrichs

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm



# === Paramètres de la simulation === #

# Coefficient de CFL
CFL = 0.8

# Vitesse transport
v = 1.0

# Domaine spatial
x, dx = Domaine_Spatial(nx=100)

# Domaine temporel
dt = CFL * dx / abs(v)
t = Domaine_Temporel(dt)

# Param invariants schemas
param = (v, dx, dt)

# Condition initiale
u0 = Condition_Initiale(x)
u0_min = np.min(u0)
u0_max = np.max(u0)

# Autres paramètres
eps_lim = 1e-2
eps_xlim = (x[-1] - x[0]) * 1e-2
eps_ylim = (u0_max - u0_min) * 1e-2









# === Resolutions numeriques  =>  Visualisation === #

# Upwind
u_Upw_Exp = Solveur(u0[:-1], t, Explicite, Upwind, param)
u_Upw_RK2 = Solveur(u0[:-1], t, Runge_Kutta_2, Upwind, param)
u_Upw_RK3 = Solveur(u0[:-1], t, Runge_Kutta_3, Upwind, param)
u_Upw_RK4 = Solveur(u0[:-1], t, Runge_Kutta_4, Upwind, param)

# Centre
u_cent_RK3 = Solveur(u0[1:-1], t, Runge_Kutta_3, Centre, param)
u_cent_RK4 = Solveur(u0[1:-1], t, Runge_Kutta_4, Centre, param)

# Lax-Wendroff / Lax-Friedrichs
u_Lax_Wend = Solveur_bis(u0[1:-1], t, Lax_Wendroff, param)
u_Lax_Fried = Solveur_bis(u0[1:-1], t, Lax_Friedrichs, param)


# print(t)
# print(len(t))
# print(len(u_Upw_Exp))
# print(len(u_cent_RK3))
# print(len(u_Lax_Wend))



# # Graphique 
nb_graphes = 3
fig, ax = Creation_Graphe((x[0], x[-1], eps_xlim), (u0_min, u0_max, eps_ylim), nb_graphes)

for i, ti in tqdm(enumerate(t), total=len(t)):

    fig.suptitle(r"$ u_t + v \, u_x = 0 $" + "\n" + f"t = {ti:.2f} / CFL = {CFL:.1f}", fontsize=14)

    if i == 0:
        u0_line, = ax[0].plot(x, u0, label = "C.I.", lw=3, color="black")
        u0_line, = ax[1].plot(x, u0, label = "C.I.", lw=3, color="black")
        u0_line, = ax[2].plot(x, u0, label = "C.I.", lw=3, color="black")

        u_Upw_Exp_line, = ax[0].plot(x[:-1], u_Upw_Exp[i], label="Euleur Explicite", color = "red", ls = "--", lw = 2)
        u_Upw_RK2_line, = ax[0].plot(x[:-1], u_Upw_RK2[i], label="RK2", color = "blue", ls = "-.", lw = 2)
        u_Upw_RK3_line, = ax[0].plot(x[:-1], u_Upw_RK3[i], label="RK3", color = "orange", ls = "--", lw = 2)
        u_Upw_RK4_line, = ax[0].plot(x[:-1], u_Upw_RK4[i], label="RK4", color = "green", ls = "-.", lw = 2)

        u_cent_RK3_line, = ax[1].plot(x[1:-1], u_cent_RK3[i], label="RK3", color = "orange", ls = "--", lw = 2)
        u_cent_RK4_line, = ax[1].plot(x[1:-1], u_cent_RK4[i], label="RK4", color = "green", ls = "-.", lw = 2)

        u_Lax_Wend_line, = ax[2].plot(x[1:-1], u_Lax_Wend[i], label="Lax-Wendroff", color = "purple", ls = "--", lw = 2)
        u_Lax_Fried_line, = ax[2].plot(x[1:-1], u_Lax_Fried[i], label="Lax-Friedrichs", color = "brown", ls = "-.", lw = 2)

    else:
        u_Upw_Exp_line.set_ydata(u_Upw_Exp[i])
        u_Upw_RK2_line.set_ydata(u_Upw_RK2[i])
        u_Upw_RK3_line.set_ydata(u_Upw_RK3[i])
        u_Upw_RK4_line.set_ydata(u_Upw_RK4[i])

        u_cent_RK3_line.set_ydata(u_cent_RK3[i])
        u_cent_RK4_line.set_ydata(u_cent_RK4[i])

        u_Lax_Wend_line.set_ydata(u_Lax_Wend[i])
        u_Lax_Fried_line.set_ydata(u_Lax_Fried[i])

    ax[0].set_title("Upwind", fontsize=12)
    ax[1].set_title("Centré", fontsize=12)
    ax[2].set_title("Spatio-temporels", fontsize=12)

    ax[0].legend(loc="lower right", fontsize=12)
    ax[1].legend(loc="lower right", fontsize=12)
    ax[2].legend(loc="lower right", fontsize=12)

    plt.pause(1e-3)

plt.show()















# # === Erreur === #

# CFL = 1e-2

# # Spatiale
# val_nx = np.logspace(1, 3, 5, dtype=int)
# val_dx = np.zeros_like(val_nx, dtype=np.float64)
# val_dt = np.zeros_like(val_nx, dtype=np.float64)

# Err_Upw_Exp = []
# Err_Upw_RK2 = []
# Err_Upw_RK3 = []
# Err_Upw_RK4 = []

# Err_cent_RK3 = []
# Err_cent_RK4 = []

# Err_Lax_Wend = []
# Err_Lax_Fried = []


# for i, nx in tqdm(enumerate(val_nx), total=len(val_nx)):
#     x, dx = Domaine_Spatial(nx)
#     val_dx[i] = dx

#     dt = CFL * dx / abs(v)
#     t = Domaine_Temporel(dt)
#     val_dt[i] = dt

#     param = (v, dx, dt)
#     u0 = Condition_Initiale(x)

#     u_exacte = Solution_Exacte(x, t[-1], v)

#     Err_Upw_Exp.append(Erreur(Solveur(u0[:-1], t, Explicite, Upwind, param)[-1], u_exacte[:-1]))
#     Err_Upw_RK2.append(Erreur(Solveur(u0[:-1], t, Runge_Kutta_2, Upwind, param)[-1], u_exacte[:-1]))
#     Err_Upw_RK3.append(Erreur(Solveur(u0[:-1], t, Runge_Kutta_3, Upwind, param)[-1], u_exacte[:-1]))
#     Err_Upw_RK4.append(Erreur(Solveur(u0[:-1], t, Runge_Kutta_4, Upwind, param)[-1], u_exacte[:-1]))

#     Err_cent_RK3.append(Erreur(Solveur(u0[1:-1], t, Runge_Kutta_3, Centre, param)[-1], u_exacte[1:-1]))
#     Err_cent_RK4.append(Erreur(Solveur(u0[1:-1], t, Runge_Kutta_4, Centre, param)[-1], u_exacte[1:-1]))

#     Err_Lax_Wend.append(Erreur(Solveur_bis(u0[1:-1], t, Lax_Wendroff, param)[-1], u_exacte[1:-1]))
#     Err_Lax_Fried.append(Erreur(Solveur_bis(u0[1:-1], t, Lax_Friedrichs, param)[-1], u_exacte[1:-1]))



# # Graphique

# fig, ax = Creation_Graphe_Erreur()

# for j in range(2):

#     ax[j].set_title(r"Erreur $L^2$" if j == 0 else r"Erreur $L^\infty$", fontsize=12)

#     ax[j].loglog(val_dx, val_dx, label=r"$\mathcal{O}(\Delta x)$", color = "black", ls = "--", lw = 2)
#     ax[j].loglog(val_dx, val_dx**2, label=r"$\mathcal{O}(\Delta x^2)$", color = "black", ls = "--", lw = 2)

#     # ax[j].loglog(val_dt, val_dt, label=r"$\mathcal{O}(\Delta t)$", color = "gray", ls = "-.", lw = 2)
#     # ax[j].loglog(val_dt, val_dt**2, label=r"$\mathcal{O}(\Delta t^2)$", color = "gray", ls = "-.", lw = 2)

#     ax[j].loglog(val_dx, [err[j] for err in Err_Upw_Exp], label="Upwind Explicite", color = "red")
#     ax[j].loglog(val_dx, [err[j] for err in Err_Upw_RK2], label="Upwind RK2", color = "blue")
#     ax[j].loglog(val_dx, [err[j] for err in Err_Upw_RK3], label="Upwind RK3", color = "orange")
#     ax[j].loglog(val_dx, [err[j] for err in Err_Upw_RK4], label="Upwind RK4", color = "green")

#     ax[j].loglog(val_dx, [err[j] for err in Err_cent_RK3], label="Centre RK3", color = "cyan")
#     ax[j].loglog(val_dx, [err[j] for err in Err_cent_RK4], label="Centre RK4", color = "magenta")

#     ax[j].loglog(val_dx, [err[j] for err in Err_Lax_Wend], label="Lax-Wendroff", color = "purple")
#     ax[j].loglog(val_dx, [err[j] for err in Err_Lax_Fried], label="Lax-Friedrichs", color = "brown")

#     ax[j].legend(loc="lower right", fontsize=6)


# plt.show()