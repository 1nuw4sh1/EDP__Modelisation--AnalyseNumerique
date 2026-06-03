import numpy as np
import matplotlib.pyplot as plt

from Parametres_Discretisation import alpha
from pathlib import Path
from PIL import Image




def Graphique_Simulation(t, Solutions, u_lim, params, pourcentage = 5, zoom_pts = 3):

    alpha, dx, dt = params

    Dossier = rf"Graphes"
    Path(Dossier).mkdir(parents=True, exist_ok=True)

    def limites_u(u_lim, p = pourcentage):
        u_min, u_max = u_lim
        
        def intervalle_confiance(u, eps = p / 100):
            s = np.sign(u)
            return (u - s * eps * abs(u), u + s * eps * abs(u))

        return [min(intervalle_confiance(u_min)) if abs(u_min) > 1e-2 else -p / 100 ,
                max(intervalle_confiance(u_max)) if abs(u_max) > 1e-2 else  p / 100]


 
    def Visualisation(it, t):
        nt = t.size
        if nt <= 100:
            return True
        else:
            # Affiche tous les 1% du temps
            ti = it % ( (nt - 1) // 100 ) == 0
            # Affiche le dernier
            tf = it == nt
            return ti or tf



    fig, ax = plt.subplot_mosaic("112",
                                 figsize=(12, 6), layout="constrained")
    
    ax0, ax1 = ax["1"], ax["2"]    

    for it, ti in enumerate(t):

        if Visualisation(it, t): 
            ax0.cla()
            ax1.cla()

            for lab in list(Solutions.keys()):

                sol = Solutions.get(lab)
                x, u = sol["x"], sol["u"][it]
                ls, lw = sol["linestyle"], sol["linewidth"]
                c, m = sol["color"], sol["marker"]

                ax0.plot(x, 
                         u,
                         label = lab,
                         linestyle = ls,
                         color = c,
                         linewidth = lw)
                
                ax1.plot(x, 
                         u,
                         label = lab,
                         linestyle = ls,
                         color = c,
                         linewidth = lw,
                         marker = m)                
                    
            

            fig.suptitle(rf"t = {ti:.3f} s $\qquad \alpha$ = {alpha:.2f}", fontsize=16)    

            ax0.set_title("Solution complète")
            ax0.set_xlabel("x")
            ax0.set_ylabel("u")
            ax0.set_ylim(limites_u(u_lim))
            ax0.legend(loc="upper right")
            ax0.grid()
            ax0.set_facecolor("#F5F5F5")
            

            nx_zoom = x.size // 2
            u_zoom = u[nx_zoom : nx_zoom + zoom_pts]
            u_lim_zoom = [min(u_zoom) * (1 + 7.5e-3), max(u_zoom) * (1 - 2.5e-3)]

            ax1.set_title("Zoom ")
            # ax1.set_xlabel("x")
            # ax1.set_ylabel("u")
            ax1.set_xlim(x[nx_zoom] + dx * .9, x[nx_zoom + zoom_pts - 1] - dx * .9)
            ax1.set_ylim(limites_u(u_lim_zoom, p = 7.5e-2))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.legend(loc="lower left")
            ax1.grid()
            ax1.set_facecolor("#F5F5F5")


            plt.pause(1e-3)
            plt.savefig(Dossier + rf"\t={ti:.3f}.png")







def Graphique_Erreur(Erreurs, Valeurs, alpha = alpha):

    Dossier = rf"Graphes\alpha={alpha:.2f}"
    Path(Dossier).mkdir(parents=True, exist_ok=True)

    val_dx, val_dt = Valeurs
    Err_Space, Err_Tempo = Erreurs
    
    
    fig, ax = plt.subplot_mosaic("12",
                                 figsize=(12, 6), layout="constrained")
    
    fig.suptitle(rf"$\alpha$ = {alpha:.2f}", fontsize=16)

    ax0, ax1 = ax["1"], ax["2"]

    ax0.loglog(val_dx, val_dx**2, label = r"$o(\Delta x^2)$", color = "black", ls = "-", lw = 2)
    ax1.loglog(val_dt, val_dt, label = r"$o(\Delta t)$", color = "red", ls = "-", lw = 2)
    ax1.loglog(val_dt, val_dt**2, label = r"$o(\Delta t^2)$", color = "black", ls = "-", lw = 2)

    for lab in list(Err_Space.keys()):

        sol = Err_Space.get(lab)
        dx, E = sol["dx"], sol["E"]
        ls, lw = sol["linestyle"], sol["linewidth"]
        c, m = sol["color"], sol["marker"]

        ax0.plot(dx, 
                 E,
                 label = lab,
                 linestyle = ls,
                 color = c,
                 linewidth = lw)
    
    for lab in list(Err_Tempo.keys()):
        sol = Err_Tempo.get(lab)
        dt, E = sol["dt"], sol["E"]
        ls, lw = sol["linestyle"], sol["linewidth"]
        c, m = sol["color"], sol["marker"]

        ax1.plot(dt, 
                 E,
                 label = lab,
                 linestyle = ls,
                 color = c,
                 linewidth = lw,)                
                    
    
    ax0.set_title("Erreur spatiale")
    ax0.set_xlabel(r"$\Delta x$")
    ax0.set_ylabel(r"$E$")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.legend(loc="lower right")
    ax0.grid()
    ax0.set_facecolor("#F5F5F5")

    ax1.set_title("Erreur temporelle")
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$E$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc="lower right")
    ax1.grid()
    ax1.set_facecolor("#F5F5F5")

    plt.show()
    plt.savefig(rf"\erreur.png")