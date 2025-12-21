import os
import shutil

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

colors = ['black', 
          'blue', 
          'pink', 
          'red', 
          'cyan', 
          'magenta', 
          'orange', 
          'purple', 
          'brown']




def Domaine(L, N, size):
    Nx = N + (size - (N % size)) % size             
    x, dx = np.linspace(-L, L, Nx, retstep=True)
    return x, dx, Nx


def Sous_Domaines(x, delta, rank, size):
    N_loc = x.size // size 
    if rank == 0 :
        start = 0
        end = N_loc + delta // 2
    elif rank == size - 1 :
        start = x.size - (N_loc + delta // 2)
        end = x.size
    else :
        start = rank * N_loc - delta // 2
        end = start + N_loc + 2 * (delta // 2)
    return x[start:end] 




def Matrice_A(xi, CL, dx, p):

    ssdiag = -1 * np.ones_like(xi[:-1])
    ssdiag[-1] = 0 if CL[-1] else -1

    diag = 2 * np.ones_like(xi)
    diag[0] = 1.0 if CL[0] else (1 + p * dx)
    diag[-1] = 1.0 if CL[-1] else (1 + p * dx)

    sdiag = -1 * np.ones_like(xi[1:])
    sdiag[0] = 0 if CL[0] else -1

    A = sp.diags([ssdiag, diag, sdiag], offsets=[-1, 0, 1], format='csc')

    return A




def Vecteur_b(xi, CL, CL_value, f, dx, p):

    b = f(xi) * dx**2

    CLg = CL_value[0]
    b[0] = 0 if CL[0] else (0.5 * b[0] + p * dx * CLg[1] + 0.5 * (CLg[0] - CLg[-1]))

    CLd = CL_value[-1]
    b[-1] = 0 if CL[-1] else (0.5 * b[-1] + p * dx * CLd[1] + 0.5 * (CLd[-1] - CLd[0]))

    return b



def Solve_monodomaine(x, dx, f, CL, CL_value):
    A = Matrice_A(x, CL, dx, None)
    b = Vecteur_b(x, CL, CL_value, f, dx, None)
    return spl.spsolve(A, b)





def Graphe_Domaine_SousDomaines(x, xi, delta, dossier, Visualisation, temps_affichage = 0.001):
    if not Visualisation:
        return

    plt.figure(figsize=(14, 3))

    Title = "Domaine & sous-domaines" + "\n" + rf"(recouvrement $\delta =$ {delta} pts)"
    plt.title(Title, fontsize=16, fontweight = 'bold')


    # === Domaine global

    ax1, = plt.plot(x, 
                    np.zeros_like(x) - 0.05, 
                    color = colors[0], 
                    marker = "o", 
                    label = r"$\Omega$")
    
    legend_ax1= plt.legend(handles = [ax1], 
                            title = "Domaine global :", 
                            loc = 'upper left')
    
    plt.gca().add_artist(legend_ax1)
    

    # === Sous-domaines

    ssd = []

    for i, d in enumerate(xi):
        ax2, = plt.plot(d, 
                        np.ones_like(d) * 0.0175 * (i + 1), 
                        color = colors[i + 1], 
                        marker = "o", 
                        label = rf"$\Omega_{i + 1}$")
        
        ssd.append(ax2)

    legend_ax2 = plt.legend(handles = ssd, 
                            title = "Sous-domaines :", 
                            loc = 'upper right', 
                            ncol = len(ssd))
    
    plt.gca().add_artist(legend_ax2)


    # === Embellissement graphique
            
    plt.xlabel("x")
    plt.xlim(x[0] * 1.025, x[-1] * 1.025)

    plt.ylim(-.075, .25)
    plt.yticks([])

    plt.tight_layout(h_pad=0.2,
                     w_pad=0.2)
    

    # === Sauvegarde

    Name_PDF = dossier[0] + "/Domaine.pdf"
    plt.savefig(Name_PDF, 
                bbox_inches="tight")
    
    Name_PNG = dossier[-1] + "/Domaine.png"
    plt.savefig(Name_PNG, 
                bbox_inches="tight", 
                dpi=300, 
                facecolor='white')
    

    # === Affichage
    plt.pause(temps_affichage)
    plt.close()





def Graphes(x, xi, u_ex, U_mono, Ui, kstop, dossier, Visualisation):

    if not Visualisation:
        return
    

    plt.ion()
    fig = plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax_exact = fig.add_subplot(gs[0,0])
    ax_mono = fig.add_subplot(gs[1,0])
    ax_schwarz = fig.add_subplot(gs[:,1:])

    for k, U in enumerate(Ui):

        plt.cla()

        Title = r"Résolution de l'équation de Poisson : $-\Delta u = f$"+ f"\nItération k = {k}"
        fig.suptitle(Title, 
                     fontsize=16, 
                     fontweight="bold")


        h, = ax_exact.plot(x, 
                           u_ex, 
                           color=colors[0], 
                           lw=3, 
                           label = r"$\Omega$")
        ax_exact.set_title("Solution exacte")
        ax_exact.set_ylim(np.min(u_ex) - 0.25, np.max(u_ex) + 0.25)
        ax_exact.legend(handles=[h])
        ax_exact.grid(True)


        h, = ax_mono.plot(x, 
                          U_mono, 
                          color=colors[-1], 
                          lw=3, 
                          label = r"$\Omega$")
        ax_mono.set_title("Solution monodomaine")
        ax_mono.set_ylim(np.min(u_ex) - 0.25, np.max(u_ex) + 0.25)
        ax_mono.legend(handles=[h])
        ax_mono.grid(True)


        handles = []
        for i, d in enumerate(xi):
            h, = ax_schwarz.plot(d, 
                                 U[i], 
                                 color=colors[i + 1],
                                 lw=3,
                                 label=rf"$\Omega_{i+1}$")
            handles.append(h)

        ax_schwarz.set_title("Méthode de Schwarz")
        ax_schwarz.set_ylim(np.min(u_ex) - 0.25, np.max(u_ex) + 0.25)
        ax_schwarz.legend(handles=handles, 
                          ncol=len(xi),
                          loc = 'upper right')
        ax_schwarz.grid(True)

        plt.tight_layout(h_pad=0.2,
                         w_pad=0.2)        
        plt.pause(0.1)

        plt.savefig(dossier[0] + f"/Iteration_{k}.pdf", bbox_inches="tight")
        plt.savefig(dossier[-1] + f"/Iteration_{k}.png", bbox_inches="tight", dpi=300)

        if k + 1 == kstop:
            break

    plt.ioff()
    plt.pause(2)
    plt.close()
