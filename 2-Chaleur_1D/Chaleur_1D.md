# Chaleur 1D

Equation de la chaleur $\quad u_t + \alpha \ u_{xx} = f \quad$  sur $\quad \Omega = [0;1] \quad$ avec conditions de bords Dirichlet homogène.
Implémentation de divers schémas numériques temporels (différences finies centrées) :
- Explicite : Euler, RK 2-4
- Implicite : Euler
- Semi-implicite : Crank-Nicolson

Etude de l'erreur de discrétisation :
- Spatiale : tous les schémas en $o(\Delta x^2)$
- Temporelle : Euler $o(\Delta t)$, Crank-Nicolson $o(\Delta t^2)$

Pas d'étude de l'erreur pour les schémas explicites, condition de stabilité trop raide.
