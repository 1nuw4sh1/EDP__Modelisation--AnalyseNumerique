from Parametres_Discretisation import alpha
import numpy as np

Sinusoidal = lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
f_Sinusoidal = lambda x, t: (alpha - 1) * np.pi**2 * Sinusoidal(x, t)