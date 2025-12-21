import sys
sys.dont_write_bytecode = True

import os
import shutil
import time

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

import mpi4py.MPI as MPI

from Fonctions import Domaine, Sous_Domaines, Matrice_A, Vecteur_b, Solve_monodomaine, Graphe_Domaine_SousDomaines, Graphes



# ======================================================================================= #

    # ============================================================== #
    # === Initialisation MPI & Creation des dossiers de stockage === #
    # ============================================================== #


# MPI
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

# Stockage
Nb_proc = f"{size}_Processeurs"
if rank == 0:
    shutil.rmtree(Nb_proc, ignore_errors=True)
    os.makedirs(Nb_proc, exist_ok=True)

    Figures = [Nb_proc + "/Figures_PDF", Nb_proc + "/Figures_PNG"]
    for f in Figures:
        shutil.rmtree(f, ignore_errors=True)
        os.makedirs(f, exist_ok=True)

    Sinus = [Figures[i] + "/sinus" for i in range(len(Figures))]
    Exp   = [Figures[i] + "/exponentielle" for i in range(len(Figures))]
    Quad  = [Figures[i] + "/quadratique" for i in range(len(Figures))]
    Autre = [Figures[i] + "/autre" for i in range(len(Figures))]

    for folders in [Sinus, Exp, Quad, Autre]:
        for f in folders:
            shutil.rmtree(f, ignore_errors=True)
            os.makedirs(f, exist_ok=True)

COMM.Barrier()

Visualisation = lambda x : True if x < int(1e4) else False


# ======================================================================================= #

    # ========================== #
    # === Parametres globaux === #
    # ========================== #


# Borne domaine physique
L = 1.             

# Discretisation 
# N = int(1e2)
N = int(1e6)      

# Recouvrement
delta = N // 20   

# Parametre condition transmission de Robin
p = 2.5

# Nb iterations max
kmax = 10**5      

# Critere d'arret
eps_rel = 1e-8             # Algorithme
eps_abs = 1e-12
eps2 = 1e-2             # Graphiques


# ======================================================================================= #

    # ======================= #
    # === Discretisations === #
    # ======================= #


# Globale
x, dx, Nx = Domaine(L, N, size)

# Locale
x_loc = Sous_Domaines(x, delta, rank, size)


# ======================================================================================= #

    # ============================================= #
    # === Visualisation Domaine & Sous-domaines === #
    # ============================================= #

xi = COMM.gather(x_loc)
Graphe_Domaine_SousDomaines(x, xi, delta, Figures, Visualisation(N)) if rank == 0 else MPI.PROC_NULL


# ======================================================================================= #

    # ================================= #
    # === Approximation : sin(Pi x) === #
    # ================================= #


# Solution exacte
u_exact = np.sin(np.pi * x)

# Terme source
f = lambda X : np.pi**2 * np.sin(np.pi * X)

# Conditions limites
g1 = 0.
g2 = 0.

# Solution monodomaine
t11 = time.time() if rank == 0 else MPI.PROC_NULL
U_mono = Solve_monodomaine(x, dx, f, [True, True], [g1, g2]) if rank == 0 else MPI.PROC_NULL
Err_mono = la.norm(U_mono - u_exact)
t22 = time.time() if rank == 0 else MPI.PROC_NULL



# Verif' domaine de bord
CL_gauche_Dirichlet = abs(x[0] - x_loc[0]) < 1e-14
CL_droite_Dirichlet = abs(x[-1] - x_loc[-1]) < 1e-14


# === Algorithme Schwarz
U_loc = np.zeros_like(x_loc)
U_old = np.zeros_like(x_loc)
Ui = []

t1 = time.time() if rank == 0 else MPI.PROC_NULL

A = Matrice_A(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], dx, p)
LU = spl.factorized(A)

k, k2 = 0, 0
Err = 1.

while Err > eps_rel and k < kmax:

    # Envoi des valeurs aux voisins
    send_gauche = U_loc[delta - 2 : delta + 1].copy()
    send_droite = U_loc[-delta - 1 : -delta + 2].copy()

    recv_gauche = np.zeros(3)
    recv_droite = np.zeros(3)

    idx_gauche = rank - 1 if rank > 0 else MPI.PROC_NULL
    idx_droite = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    COMM.Sendrecv(send_gauche, idx_gauche, recvbuf=recv_gauche, source=idx_gauche)
    COMM.Sendrecv(send_droite, idx_droite, recvbuf=recv_droite, source=idx_droite)

    # MAJ Second membre
    CLg = g1 if rank == 0 else recv_gauche
    CLd = g2 if rank == size - 1 else recv_droite
    b = Vecteur_b(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], [CLg, CLd], f, dx, p)

    # Solve Poisson
    U_loc = LU(b)

    # Calcul Erreur
    err_loc = la.norm(U_loc - U_old) / (la.norm(U_loc) + eps_abs)
    Err = COMM.allreduce(err_loc, op=MPI.MAX)

    k2 = k if (Err < eps2 and k2 == 0) else k2

    U_old = U_loc.copy()
    if Visualisation and k2 == 0 :
        Ui.append(COMM.gather(U_loc))

    k += 1

t2 = time.time() if rank == 0 else MPI.PROC_NULL
if rank == 0:
    print(f"\n\n========== {size} processeurs ==========")
    print(f"\n=> Solution Monodomaine : {t22 - t11:.3f}s, erreur = {Err_mono:.2e}")
    print(f"=> Algorithme Schwarz Parallele : {t2 - t1:.3f}s, {k-1} iterations, precision eps = {eps_rel}")
    



# Animation
Graphes(x, xi, u_exact, U_mono, Ui, k2, Sinus, Visualisation(N)) if rank == 0 else MPI.PROC_NULL


# ======================================================================================= #

    # ====================================== #
    # === Approximation : e^(-x^2) - e^-1 === #
    # ====================================== #


# Solution exacte
u_exact = np.exp(-x**2) - np.exp(-1)

# Terme source
f = lambda X : (2 - 4 * X**2) * np.exp(-X**2)

# Conditions limites
g1 = 0.
g2 = 0.

# Solution monodomaine
t11 = time.time() if rank == 0 else MPI.PROC_NULL
U_mono = Solve_monodomaine(x, dx, f, [True, True], [g1, g2]) if rank == 0 else MPI.PROC_NULL
Err_mono = la.norm(U_mono - u_exact)
t22 = time.time() if rank == 0 else MPI.PROC_NULL

# Verif' domaine de bord
CL_gauche_Dirichlet = abs(x[0] - x_loc[0]) < 1e-14
CL_droite_Dirichlet = abs(x[-1] - x_loc[-1]) < 1e-14


# === Algorithme Schwarz
U_loc = np.zeros_like(x_loc)
U_old = np.zeros_like(x_loc)
Ui = []

t1 = time.time() if rank == 0 else MPI.PROC_NULL

A = Matrice_A(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], dx, p)
LU = spl.factorized(A)

k, k2 = 0, 0
Err = 1.

while Err > eps_rel and k < kmax:

    # Envoi des valeurs aux voisins
    send_gauche = U_loc[delta - 2 : delta + 1].copy()
    send_droite = U_loc[-delta - 1 : -delta + 2].copy()

    recv_gauche = np.zeros(3)
    recv_droite = np.zeros(3)

    idx_gauche = rank - 1 if rank > 0 else MPI.PROC_NULL
    idx_droite = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    COMM.Sendrecv(send_gauche, idx_gauche, recvbuf=recv_gauche, source=idx_gauche)
    COMM.Sendrecv(send_droite, idx_droite, recvbuf=recv_droite, source=idx_droite)

    # MAJ Second membre
    CLg = g1 if rank == 0 else recv_gauche
    CLd = g2 if rank == size - 1 else recv_droite
    b = Vecteur_b(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], [CLg, CLd], f, dx, p)

    # Solve Poisson
    U_loc = LU(b)

    # Calcul Erreur
    err_loc = la.norm(U_loc - U_old) / (la.norm(U_loc) + eps_abs)
    Err = COMM.allreduce(err_loc, op=MPI.MAX)

    k2 = k if (Err < eps2 and k2 == 0) else k2

    U_old = U_loc.copy()
    if Visualisation and k2 == 0 :
        Ui.append(COMM.gather(U_loc))

    k += 1

t2 = time.time() if rank == 0 else MPI.PROC_NULL
if rank == 0:
    print(f"\n=> Solution Monodomaine : {t22 - t11:.3f}s, erreur = {Err_mono:.2e}")
    print(f"=> Algorithme Schwarz Parallele : {t2 - t1:.3f}s, {k-1} iterations, precision eps = {eps_rel}")


# Animation
Graphes(x, xi, u_exact, U_mono, Ui, k2, Exp, Visualisation(N)) if rank == 0 else MPI.PROC_NULL


# ======================================================================================= #

    # =================================== #
    # === Approximation : 1 - x^2 === #
    # =================================== #


# Solution exacte
u_exact = 1 - x**2

# Terme source
f = lambda X : 2 * np.ones_like(X)

# Conditions limites
g1 = 0.
g2 = 0.

# Solution monodomaine
t11 = time.time() if rank == 0 else MPI.PROC_NULL
U_mono = Solve_monodomaine(x, dx, f, [True, True], [g1, g2]) if rank == 0 else MPI.PROC_NULL
Err_mono = la.norm(U_mono - u_exact)
t22 = time.time() if rank == 0 else MPI.PROC_NULL

# Verif' domaine de bord
CL_gauche_Dirichlet = abs(x[0] - x_loc[0]) < 1e-14
CL_droite_Dirichlet = abs(x[-1] - x_loc[-1]) < 1e-14


# === Algorithme Schwarz
U_loc = np.zeros_like(x_loc)
U_old = np.zeros_like(x_loc)
Ui = []

t1 = time.time() if rank == 0 else MPI.PROC_NULL

A = Matrice_A(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], dx, p)
LU = spl.factorized(A)

k, k2 = 0, 0
Err = 1.

while Err > eps_rel and k < kmax:

    # Envoi des valeurs aux voisins
    send_gauche = U_loc[delta - 2 : delta + 1].copy()
    send_droite = U_loc[-delta - 1 : -delta + 2].copy()

    recv_gauche = np.zeros(3)
    recv_droite = np.zeros(3)

    idx_gauche = rank - 1 if rank > 0 else MPI.PROC_NULL
    idx_droite = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    COMM.Sendrecv(send_gauche, idx_gauche, recvbuf=recv_gauche, source=idx_gauche)
    COMM.Sendrecv(send_droite, idx_droite, recvbuf=recv_droite, source=idx_droite)

    # MAJ Second membre
    CLg = g1 if rank == 0 else recv_gauche
    CLd = g2 if rank == size - 1 else recv_droite
    b = Vecteur_b(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], [CLg, CLd], f, dx, p)

    # Solve Poisson
    U_loc = LU(b)

    # Calcul Erreur
    err_loc = la.norm(U_loc - U_old) / (la.norm(U_loc) + eps_abs)
    Err = COMM.allreduce(err_loc, op=MPI.MAX)

    k2 = k if (Err < eps2 and k2 == 0) else k2

    U_old = U_loc.copy()
    if Visualisation and k2 == 0 :
        Ui.append(COMM.gather(U_loc))

    k += 1

t2 = time.time() if rank == 0 else MPI.PROC_NULL
if rank == 0:
    print(f"\n=> Solution Monodomaine : {t22 - t11:.3f}s, erreur = {Err_mono:.2e}")
    print(f"=> Algorithme Schwarz Parallele : {t2 - t1:.3f}s, {k-1} iterations, precision eps = {eps_rel}")


# Animation
Graphes(x, xi, u_exact, U_mono, Ui, k2, Quad, Visualisation(N)) if rank == 0 else MPI.PROC_NULL


# ======================================================================================= #

    # ============================================ #
    # === Approximation : x^2 e^(-x^2) sin(3x) === #
    # ============================================ #


# Solution exacte
u_exact = x**2 * np.exp(-x**2) * np.sin(3*x)

# Terme source
f = lambda x : (-4 * x**4 + 19 * x**2 - 2) * np.exp(-x**2) * np.sin(3 * x) - 12 * x * (1 - x**2) * np.exp(-x**2) * np.cos(3 * x)

# Conditions limites
g1 = u_exact[0]
g2 = u_exact[-1]

# Solution monodomaine
t11 = time.time() if rank == 0 else MPI.PROC_NULL
U_mono = Solve_monodomaine(x, dx, f, [True, True], [g1, g2]) if rank == 0 else MPI.PROC_NULL
Err_mono = la.norm(U_mono - u_exact)
t22 = time.time() if rank == 0 else MPI.PROC_NULL

# Verif' domaine de bord
CL_gauche_Dirichlet = abs(x[0] - x_loc[0]) < 1e-14
CL_droite_Dirichlet = abs(x[-1] - x_loc[-1]) < 1e-14


# === Algorithme Schwarz
U_loc = np.zeros_like(x_loc)
U_old = np.zeros_like(x_loc)
Ui = []

t1 = time.time() if rank == 0 else MPI.PROC_NULL

A = Matrice_A(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], dx, p)
LU = spl.factorized(A)

k, k2 = 0, 0
Err = 1.

while Err > eps_rel and k < kmax:

    # Envoi des valeurs aux voisins
    send_gauche = U_loc[delta - 2 : delta + 1].copy()
    send_droite = U_loc[-delta - 1 : -delta + 2].copy()

    recv_gauche = np.zeros(3)
    recv_droite = np.zeros(3)

    idx_gauche = rank - 1 if rank > 0 else MPI.PROC_NULL
    idx_droite = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    COMM.Sendrecv(send_gauche, idx_gauche, recvbuf=recv_gauche, source=idx_gauche)
    COMM.Sendrecv(send_droite, idx_droite, recvbuf=recv_droite, source=idx_droite)

    # MAJ Second membre
    CLg = g1 if rank == 0 else recv_gauche
    CLd = g2 if rank == size - 1 else recv_droite
    b = Vecteur_b(x_loc, [CL_gauche_Dirichlet, CL_droite_Dirichlet], [CLg, CLd], f, dx, p)

    # Solve Poisson
    U_loc = LU(b)

    # Calcul Erreur
    err_loc = la.norm(U_loc - U_old) / (la.norm(U_loc) + eps_abs)
    Err = COMM.allreduce(err_loc, op=MPI.MAX)

    k2 = k if (Err < eps2 and k2 == 0) else k2

    U_old = U_loc.copy()
    if Visualisation and k2 == 0 :
        Ui.append(COMM.gather(U_loc))

    k += 1

t2 = time.time() if rank == 0 else MPI.PROC_NULL
if rank == 0:
    print(f"\n=> Solution Monodomaine : {t22 - t11:.3f}s, erreur = {Err_mono:.2e}")
    print(f"=> Algorithme Schwarz Parallele : {t2 - t1:.3f}s, {k-1} iterations, precision eps = {eps_rel}")
    print("\n\n")


# Animation
Graphes(x, xi, u_exact, U_mono, Ui, k2, Autre, Visualisation(N)) if rank == 0 else MPI.PROC_NULL