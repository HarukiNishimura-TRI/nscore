import numpy as np 
from scipy.stats import dirichlet

if __name__ == "__main__":
    alpha = np.ones(5)
    alpha[-1] += 1

    dirichlet_posterior = dirichlet(alpha)
    print("Alpha: ")
    print(alpha)
    print("Dirichlet mean: ")
    print(dirichlet_posterior.mean())
    print("Dirichlet args: ")
    print(dirichlet_posterior.alpha)

    print()
    print()
    dummy = True 
    if dummy in np.arange(5):
        print("Yay!!!")