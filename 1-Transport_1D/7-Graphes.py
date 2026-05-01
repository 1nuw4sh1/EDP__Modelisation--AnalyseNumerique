import matplotlib.pyplot as plt


def Creation_Graphe(param_x, param_y, nb_graphes):

    # Recuperation des paramètres
    x0, x1, eps_xlim = param_x
    y0, y1, eps_ylim = param_y


    # Creation du graphique
    fig, ax = plt.subplots(1, nb_graphes,figsize=(5 * nb_graphes, 5))

    for i in range(nb_graphes):
        ax[i].set_xlabel("x", fontsize=14)
        ax[i].set_ylabel(r"$u(x,t)$", fontsize=14)

        ax[i].set_xlim(x0 - eps_xlim, x1 + eps_xlim)
        ax[i].set_ylim(y0 - eps_ylim, y1 + eps_ylim)

        ax[i].grid("both")
        ax[i].set_aspect("equal")

    return fig, ax


def Creation_Graphe_Erreur(nb_lignes = 1, nb_colonnes = 2):
    fig, ax = plt.subplots(nb_lignes, nb_colonnes, figsize=(5 * nb_colonnes, 5 * nb_lignes))

    for j in range(nb_colonnes):
        ax[j].set_xlabel(r"$\Delta x$ / $\Delta t$", fontsize=14)
        ax[j].set_ylabel(r"$L^2$" if j == 0 else r"$L^\infty$", fontsize=14)
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].grid("both")

    return fig, ax