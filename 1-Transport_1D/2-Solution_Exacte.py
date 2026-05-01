import numpy as np

def Solution_Exacte(x, t, c):
    X = (x - c * t) % 1
    return np.exp(-50 * (X - 0.3)**2)
    # return np.where((X >= 0.2) & (X <= 0.4), 1, 0.0)

def Condition_Initiale(x, t = 0, c = 1):
    return Solution_Exacte(x, t, c)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1.5, 100)
    c = x[-1] / t[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Solution exacte du transport linéaire", fontsize=16)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u(x,t)", fontsize=14)
    ax.grid("both")


    for i, ti in enumerate(t):
        ax.clear()

        ax.plot(x, Solution_Exacte(x, ti, c), label=f"t={ti:.1f}")
        ax.legend()

        plt.pause(1e-6)
    
    plt.show()