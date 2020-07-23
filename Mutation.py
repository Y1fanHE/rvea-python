import numpy as np

def uniform_crossover(x1, x2):
    sigma = np.random.rand()
    x1_ = sigma * x1 + (1 - sigma) * x2
    x2_ = (1 - sigma) * x1 + sigma * x2
    return x1_, x2_

def gaussian_mutation(x):
    return x + np.random.randn(len(x))