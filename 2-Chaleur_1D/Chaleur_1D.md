# Chaleur 1D

Equation de la chaleur $\quad u_t + \alpha \ u_{xx} = f \quad$  sur $\quad \Omega = [0;1] \quad$ avec conditions de bords Dirichlet homogène.
Implémentation de divers schémas numériques temporels (différences finies centrées) :
- Explicite : Euler, RK 2-4
- Implicite : Euler
- Semi-implicite : Crank-Nicolson

Etude de l'erreur spatiale pour tous les schémas (vérification de l'ordre 2 en espace), et temporelle pour les schemas (semi-) implicites (Euler $o(\Delta x)$, Crank-Nicolson $o(\Delta x^2)$). Pas d'étude de l'erreur temporelle pour les schémas explicites, condition de stabilité trop raide.
