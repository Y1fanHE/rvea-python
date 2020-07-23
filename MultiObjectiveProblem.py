import numpy as np

def sch(x):
    f1 = np.power(x[0], 2)
    f2 = np.power(x[0] - 2, 2)
    return np.array([f1, f2])