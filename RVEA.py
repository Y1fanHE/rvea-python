import numpy as np
from copy import copy
from ReferenceVector import das_dennis, neighboring_angle
from AngleDistancePenalty import adp_selection
from Mutation import uniform_crossover, gaussian_mutation
from MultiObjectiveProblem import sch

n_obj = 2                                               # number of objectives
n_var = 1                                               # number of decision variable
xl, xu = -100, 100                                      # lower, upper boundaries
f = sch                                                 # objective function

n_pop = 50                                              # population size
n_part = 99                                             # number of partitions
n_eval = 25000                                          # maximum evaluation
n_gen = n_eval // n_pop                                 # maximum generation

alpha = 2.0                                             # parameter of adp selection
fr = 0.1                                                # parameter of adp selection

ref_dirs_ini = das_dennis(n_part, n_obj)
ref_dirs = copy(ref_dirs_ini[:, :])
ref_angle = neighboring_angle(ref_dirs)
X = np.random.uniform(xl, xu, (n_pop, n_var))
F = np.array([f(x) for x in X])

for c_gen in range(n_gen - 1):

    X_ = np.full(X.shape, np.nan)

    # reproduce offspring
    for i in range(X.shape[0] // 2):
        x1 = X[np.random.choice(X.shape[0]), :]
        x2 = X[np.random.choice(X.shape[0]), :]
        x1, x2 = uniform_crossover(x1, x2)
        x1, x2 = gaussian_mutation(x1), gaussian_mutation(x2)
        X_[2 * i], X_[2 * i +1] = np.clip(x1, xl, xu), np.clip(x2, xl, xu)
    if X.shape[0] % 2 != 0:
        x1 = X[np.random.choice(X.shape[0]), :]
        x2 = X[np.random.choice(X.shape[0]), :]
        x1, x2 = uniform_crossover(x1, x2)
        x1, x2 = gaussian_mutation(x1), gaussian_mutation(x2)
        X_[-1] = np.clip(x1, xl, xu)

    # evaluate and merge with parents
    F_ = np.array([f(x) for x in X_])
    X_, F_ = np.vstack([X, X_]), np.vstack([F, F_])

    # adp selection
    theta0 = (c_gen / n_gen) ** alpha * n_obj
    X, F = adp_selection(X_, F_, ref_dirs, theta0, ref_angle)

    # reference vector adaption
    if c_gen % (n_gen * fr) == 0:
        z_min = np.min(F, axis=0)
        z_max = np.max(F, axis=0)
        ref_dirs = ref_dirs_ini * (z_max - z_min)
        ref_dirs = ref_dirs / np.linalg.norm(ref_dirs, axis=1).reshape(-1, 1)
        ref_angle = neighboring_angle(ref_dirs)

print(F)

import matplotlib.pyplot as plt

plt.scatter(F[:,0], F[:,1], c="r")
plt.show()